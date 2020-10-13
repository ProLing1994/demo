#include "feat/wave-reader.h"
#include "online/online-audio-source.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <poll.h>
#include <signal.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>

namespace kaldi {

bool Write(const std::string &msg); // write to accepted client
bool WriteLn(const std::string &msg, const std::string &eol = "\n"); // write line to accepted client

std::string LatticeToString(const Lattice &lat, const fst::SymbolTable &word_syms) {
  LatticeWeight weight;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(lat, &alignment, &words, &weight);

  std::ostringstream msg;
  for (size_t i = 0; i < words.size(); i++) {
    std::string s = word_syms.Find(words[i]);
    if (s.empty()) {
      KALDI_WARN << "Word-id " << words[i] << " not in symbol table.";
      msg << "<#" << std::to_string(i) << "> ";
    } else
      msg << s << " ";
  }
  return msg.str();
}

std::string GetTimeString(int32 t_beg, int32 t_end, BaseFloat time_unit) {
  char buffer[100];
  double t_beg2 = t_beg * time_unit;
  double t_end2 = t_end * time_unit;
  snprintf(buffer, 100, "%.2f %.2f", t_beg2, t_end2);
  return std::string(buffer);
}

int32 GetLatticeTimeSpan(const Lattice& lat) {
  std::vector<int32> times;
  LatticeStateTimes(lat, &times);
  return times.back();
}

std::string LatticeToString(const CompactLattice &clat, const fst::SymbolTable &word_syms) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return "";
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);
  return LatticeToString(best_path_lat, word_syms);
}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online\n"
        "decoding with neural nets (nnet3 setup), with iVector-based\n"
        "speaker adaptation and endpointing.\n"
        "Note: some configuration values and inputs are set via config\n"
        "files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-tcp-nnet3-decode-faster [options] <nnet3-in> "
        "<fst-in> <word-symbol-table> <spk2utt-rspecifier> <wav-rspecifier>\n";

    ParseOptions po(usage);


    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;

    BaseFloat chunk_length_secs = 0.185;
    BaseFloat output_period = 1;
    BaseFloat samp_freq = 16000.0;
    bool produce_time = true;

    po.Register("samp-freq", &samp_freq,
                "Sampling frequency of the input signal (coded as 16-bit slinear).");
    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.");
    po.Register("output-period", &output_period,
                "How often in seconds, do we check for changes in output.");
    po.Register("produce-time", &produce_time,
                "Prepend begin/end times between endpoints (e.g. '5.46 6.81 <text_output>', in seconds)");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      return 1;
    }

    std::string nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        word_syms_filename = po.GetArg(3),
        spk2utt_rspecifier = po.GetArg(4),
        wav_rspecifier = po.GetArg(5);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

    BaseFloat frame_shift = feature_info.FrameShiftInSeconds();
    int32 frame_subsampling = decodable_opts.frame_subsampling_factor;

    KALDI_VLOG(1) << "Loading AM...";

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    // this object contains precomputed stuff that is used by all decodable
    // objects.  It takes a pointer to am_nnet because if it has iVectors it has
    // to modify the nnet to accept iVectors at intervals.
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);

    KALDI_VLOG(1) << "Loading FST...";

    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    fst::SymbolTable *word_syms = NULL;
    if (!word_syms_filename.empty())
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;

    KALDI_VLOG(1) << "Loading WAV...";

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);

    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!wav_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find audio for utterance " << utt;
          continue;
        }
        std::cout << "File: " << utt << std::endl;
        const WaveData &wav_data = wav_reader.Value(utt);
        OnlineVectorSource au_src(wav_data.Data().Row(0));

        int32 samp_count = 0;// this is used for output refresh rate
        size_t chunk_len = static_cast<size_t>(chunk_length_secs * samp_freq);
        int32 check_period = static_cast<int32>(samp_freq * output_period);
        int32 check_count = check_period;

        int32 frame_offset = 0;

        bool eos = false;

        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
                                            decodable_info,
                                            *decode_fst, &feature_pipeline);

        while (!eos) {
          decoder.InitDecoding(frame_offset);
          OnlineSilenceWeighting silence_weighting(
              trans_model,
              feature_info.silence_weighting_config,
              decodable_opts.frame_subsampling_factor);
          std::vector<std::pair<int32, BaseFloat>> delta_weights;

          while (true) {
            Vector<BaseFloat> wave_part(chunk_len);
            eos = !au_src.Read(&wave_part);

            if (eos) {
              feature_pipeline.InputFinished();

              if (silence_weighting.Active() &&
                  feature_pipeline.IvectorFeature() != NULL) {
                silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
                silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                                  frame_offset * decodable_opts.frame_subsampling_factor,
                                                  &delta_weights);
                feature_pipeline.UpdateFrameWeights(delta_weights);
              }

              decoder.AdvanceDecoding();
              decoder.FinalizeDecoding();
              frame_offset += decoder.NumFramesDecoded();
              if (decoder.NumFramesDecoded() > 0) {
                CompactLattice lat;
                decoder.GetLattice(true, &lat);
                std::string msg = LatticeToString(lat, *word_syms);

                // get time-span from previous endpoint to end of audio,
                if (produce_time) {
                  int32 t_beg = frame_offset - decoder.NumFramesDecoded();
                  int32 t_end = frame_offset;
                  msg = GetTimeString(t_beg, t_end, frame_shift * frame_subsampling) + " " + msg;
                }

                KALDI_VLOG(1) << "EndOfAudio, sending message: " << msg;
                WriteLn(msg);
              } else
                Write("\n");
              break;
            }

            feature_pipeline.AcceptWaveform(samp_freq, wave_part);
            samp_count += chunk_len;

            if (silence_weighting.Active() &&
                feature_pipeline.IvectorFeature() != NULL) {
              silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
              silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                                frame_offset * decodable_opts.frame_subsampling_factor,
                                                &delta_weights);
              feature_pipeline.UpdateFrameWeights(delta_weights);
            }

            decoder.AdvanceDecoding();

            if (samp_count > check_count) {
              if (decoder.NumFramesDecoded() > 0) {
                Lattice lat;
                decoder.GetBestPath(false, &lat);
                TopSort(&lat); // for LatticeStateTimes(),
                std::string msg = LatticeToString(lat, *word_syms);

                // get time-span after previous endpoint,
                if (produce_time) {
                  int32 t_beg = frame_offset;
                  int32 t_end = frame_offset + GetLatticeTimeSpan(lat);
                  msg = GetTimeString(t_beg, t_end, frame_shift * frame_subsampling) + " " + msg;
                }

                KALDI_VLOG(1) << "Temporary transcript: " << msg;
                WriteLn(msg, "\r");
              }
              check_count += check_period;
            }

            if (decoder.EndpointDetected(endpoint_opts)) {
              decoder.FinalizeDecoding();
              frame_offset += decoder.NumFramesDecoded();
              CompactLattice lat;
              decoder.GetLattice(true, &lat);
              std::string msg = LatticeToString(lat, *word_syms);

              // get time-span between endpoints,
              if (produce_time) {
                int32 t_beg = frame_offset - decoder.NumFramesDecoded();
                int32 t_end = frame_offset;
                msg = GetTimeString(t_beg, t_end, frame_shift * frame_subsampling) + " " + msg;
              }

              KALDI_VLOG(1) << "Endpoint, sending message: " << msg;
              WriteLn(msg);
              break; // while (true)
            }
          }
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
} // main()

namespace kaldi {
bool Write(const std::string &msg) {

  const char *p = msg.c_str();
  size_t to_write = msg.size();
  size_t wrote = 0;
  while (to_write > 0) {
    ssize_t ret = write(1, static_cast<const void *>(p + wrote), to_write);
    if (ret <= 0)
      return false;

    to_write -= ret;
    wrote += ret;
  }

  return true;
}

bool WriteLn(const std::string &msg, const std::string &eol) {
  if (Write(msg))
    return Write(eol);
  else return false;
}
}  // namespace kaldi
