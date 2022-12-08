import codecs
import sys

input_file  = codecs.open(sys.argv[1], 'r', 'utf-8')
output_file = codecs.open(sys.argv[2], 'w', 'utf-8')
# input_file  = codecs.open("/home/huanyuan/code/third_code/tts/TTS_Course/02_front_end/2-2_g2p/dataset/mini-train.dict", 'r', 'utf-8')
# output_file = codecs.open("/home/huanyuan/code/third_code/tts/TTS_Course/02_front_end/2-2_g2p/dataset/mini-train.formatted.corpus", 'w', 'utf-8')

for line in input_file.readlines():
    word_list = line.strip().split('\t')
    hanzi_list = word_list[0]
    pinyin_list = word_list[1].split()
    assert len(hanzi_list) == len(pinyin_list), line

    for idx in range(len(hanzi_list)):
        output_file.write(hanzi_list[idx] + "}" + pinyin_list[idx] + " ")
    output_file.write("\n")

input_file.close()
output_file.close()
