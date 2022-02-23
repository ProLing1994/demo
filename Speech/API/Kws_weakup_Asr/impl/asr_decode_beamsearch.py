import kenlm
import copy
import heapq
import math
from collections import defaultdict

NEG_INF = -float("inf")

class Ken_LM:
  def __init__(self,lmPATH):
    self.__model = kenlm.Model(lmPATH)
    self.senlen = 3

  def compute(self,state,word):
    assert isinstance(state,list)
    state.append(word)
    sentence = " ".join(state)
    prob = list(self.__model.full_scores(sentence , bos=False, eos=False))[-1][0]  #输入需要显式指定<s>起始符,不会默认添加，然后忽略eos终止符，不会为输入的sentence添加默认终止符</s>
    if len(state) < self.__model.order:
      return state, prob
    else:
      return state[1:], prob

  def score(self, modified, bos=True, eos=True):
      sentence = " ".join(modified)

      return self.__model.score(sentence, bos=bos, eos=eos)


class PrefixBeam:
    prob_nb= -float("inf")
    prob_b = -float("inf")
    prob_lm=0
    context=''
    context_score=0


def load_lexicon(asr_dict_path):
    id2word = []
    with open(asr_dict_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            id2word.append(line.strip())
    return id2word

class BeamSearch:
    def __init__(self,lexicon,beam_size=5,blankID=0):
        self.beam_size = beam_size
        self.lexicon=lexicon
        self.blankID=blankID
        self.lm_weight=0.3
        self.conetxt_bias_list = [['yin', 'liang'], ['tiao', 'jie'], ['kong', 'tiao'], ['tian', 'chuang'], ['bai', 'fen', 'zhi'],
                          ['jin', 'ji', 'lian', 'xi', 'ren'],['dao','lu','jiu','yuan']]
        self.context_head = ['yin', 'tiao', 'kong', 'tian', 'bai', 'jin','dao']
        self.context_boundary = ['da', 'kai', 'guan', 'bi', 'lian', 'jie', 'duan', 'tiao', 'yin', 'liang', 'kong', 'tian', 'wen','du', 'qian',
                        'hou', 'pai', 'chuang', 'hu', 'shou', 'ji', 'di', 'tu', 'bai', 'fen', 'zhi', 'lai', 'yi', 'er', 'san', 'si',
                        'wu', 'liu', 'qi', 'ba', 'jiu', 'shi', 'lan', 'ya', 'ta', 'ma', 'de', 'sha', 'cao', 'ni', 'qing', 'qiu',
                        'dao', 'lu', 'yuan', 'bo', 'jin', 'xi', 'ren', 'ke', 'gou', 'zu', 'mei', 'hao', 'jiao', 'tong','wei', 'dao']
    def _log_add(self,args):
        """
        Stable log add
        """
        if all(a == -float('inf') for a in args):
            return -float('inf')
        a_max = max(args)
        lsp = math.log(sum(math.exp(a - a_max) for a in args))
        return a_max + lsp

    def _log_add_lm(self,lm,args):
        prefix=args[0]
        args=args[1][:2]
        if len(prefix) == 0:
            prob_lm = self.lm_weight * lm.score(("oov",), bos=False, eos=False)
        else:
            prob_lm = self.lm_weight * lm.score(prefix, bos=False, eos=False)
        if all(a == -float('inf') for a in args):
            return -float('inf')
        a_max = max(args)
        lsp = math.log(sum(math.exp(a - a_max) for a in args))
        return a_max + lsp+self.lm_weight*prob_lm/max(1,len(prefix))

    def simple_beam_search(self,probs, lm, bswt=1.0, lmwt=1.0):
        probs = probs.cpu().detach().numpy()
        frames, dims = probs.shape
        # initilize LM score
        _, initLMscore = lm.compute(state=[], word="UNK")
        initLMscore *= lmwt
        results = [{"words": [], "ali": [], "bs": 0, "lm": initLMscore, "total": 0, "lmState": ["<s>"], "hot_word": ['',0]}]
        for i in range(frames):
            temp = []
            testList = probs[i]
            tmp = zip(range(len(testList)), testList)
            large5 = heapq.nlargest(self.beam_size, tmp, key=lambda x: x[1])
            for preResult in results:
                for (d, bsScore) in large5:
                    if (bsScore < -1.6):
                        continue
                    one = copy.deepcopy(preResult)
                    # 1. compute beam search score
                    one["bs"] += bswt * bsScore
                    # 2. compute LM score
                    if d != self.blankID:  # ignore blank
                        if len(one["ali"]) == 0 or d != one["ali"][-1]:  # ignore continuous repeated symbols
                            word = self.lexicon[d]
                            newState, lmScore = lm.compute(state=one["lmState"], word=word)
                            #one["wordscore"].append([word, bsScore])
                            one["words"].append(word)
                            one["lm"] += lmwt * lmScore
                            one["lmState"] = newState
                    # 3. record alignment
                    one["ali"].append(d)
                    # 4. compute total score with length normalization
                    numFrames = len(one["ali"])
                    numWords = len(one["words"]) + 1
                    one["total"] = one["bs"] / numFrames + one["lm"] / numWords
                    temp.append(one)
                # prune
            temp = sorted(temp, key=lambda x: x["total"], reverse=True)
            if (len(temp) == 0):
                continue
            results = temp[:self.beam_size]
            results=sorted(results, key=lambda x: x["total"], reverse=True)
        return results[0]['words']



    def simple_beam_search_context_bias(self,probs, lm, bswt=1.0, lmwt=0.3):
        probs = probs.cpu().detach().numpy()
        frames, dims = probs.shape
        bias_score=3
        # initilize LM score
        _, initLMscore = lm.compute(state=[], word="UNK")
        initLMscore *= lmwt
        results = [{"words": [], "ali": [], "bs": 0, "lm": initLMscore, "total": 0, "lmState": ["<s>"], "hot_word": {"context":[],"drawback_socre":0}}]
        for i in range(frames):
            temp = []
            testList = probs[i]
            tmp = zip(range(len(testList)), testList)
            large5 = heapq.nlargest(self.beam_size, tmp, key=lambda x: x[1])
            for preResult in results:
                for (d, bsScore) in large5:
                    if (bsScore < -2.3):
                        continue
                    one = copy.deepcopy(preResult)
                    # 1. compute beam search score
                    one["bs"] += bswt * bsScore
                    # 2. compute LM score
                    if d != self.blankID:  # ignore blank
                        if len(one["ali"]) == 0 or d != one["ali"][-1]:  # ignore continuous repeated symbols
                            word = self.lexicon[d]

                            newState, lmScore = lm.compute(state=one["lmState"], word=word)

                            if (len(one['hot_word']['context']) == 0 and word in self.context_head):
                                one['hot_word']['context'] = self.conetxt_bias_list[self.context_head.index(word)].copy()
                                one['hot_word']['drawback_socre']+=bias_score
                                one["bs"]  += bias_score
                                one['hot_word']['context'] .pop(0)
                            elif (one['hot_word']['context'] and word == one['hot_word']['context'][0]):
                                one['hot_word']['drawback_socre']+=bias_score
                                one["bs"]  += bias_score
                                one['hot_word']['context'] .pop(0)
                            elif (len(one['hot_word']['context']) and word != one['hot_word']['context'][0]):
                                one["bs"]  -= one['hot_word']['drawback_socre']
                                one['hot_word']['drawback_socre']=0.0
                                one['hot_word']['context']=[]
                            if (len(one['hot_word']['context']) == 0 and one['hot_word']['drawback_socre'] != 0):
                                one['hot_word']['drawback_socre']=0.0

                            #one["wordscore"].append([word, bsScore])
                            one["words"].append(word)
                            one["lm"] += lmwt * lmScore
                            one["lmState"] = newState
                    # 3. record alignment
                    one["ali"].append(d)
                    # 4. compute total score with length normalization
                    numFrames = len(one["ali"])
                    numWords = len(one["words"]) + 1
                    one["total"] = one["bs"] / numFrames + one["lm"] / numWords
                    temp.append(one)
                # prune
            temp = sorted(temp, key=lambda x: x["total"], reverse=True)
            if (len(temp) == 0):
                continue
            results = temp[:self.beam_size]
            results=sorted(results, key=lambda x: x["total"], reverse=True)
        return results[0]['words']


    def prefix_beam_search(self, probs, lm=None):
          blankID=0
          maxlen = probs.shape[0]
          cur_hyps = [(tuple(), (0.0, -float('inf')))]
          # 2. CTC beam search step by step
          for t in range(0, maxlen):
              logp = probs[t]  # (vocab_size,)
              # key: prefix, value (pb, pnb), default value(-inf, -inf)
              next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
              # 2.1 First beam prune: select topk best
              top_k_logp, top_k_index = logp.topk(self.beam_size)  # (beam_size,)
              for s in top_k_index:
                  s = s.item()
                  ps = logp[s].item()

                  for prefix, (pb, pnb) in cur_hyps:
                      last = prefix[-1] if len(prefix) > 0 else None
                      if s == blankID:  # blank
                          n_pb, n_pnb = next_hyps[prefix]
                          n_pb = self._log_add([n_pb, pb + ps, pnb + ps])
                          next_hyps[prefix] = (n_pb, n_pnb)
                      elif s == last:
                          #  Update *ss -> *s;
                          n_pb, n_pnb = next_hyps[prefix]
                          n_pnb = self._log_add([n_pnb, pnb + ps])
                          next_hyps[prefix] = (n_pb, n_pnb)
                          # Update *s-s -> *ss, - is for blank
                          n_prefix = prefix + (s, )
                          n_pb, n_pnb = next_hyps[n_prefix]
                          n_pnb = self._log_add([n_pnb, pb + ps])
                          next_hyps[n_prefix] = (n_pb, n_pnb)
                      else:

                          n_prefix = prefix + (s, )
                          n_pb, n_pnb = next_hyps[n_prefix]
                          n_pnb = self._log_add([n_pnb, pb + ps, pnb + ps])
                          next_hyps[n_prefix] = (n_pb, n_pnb)
              # 2.2 Second beam prune
              next_hyps = sorted(next_hyps.items(),
                                 key=lambda x: self._log_add(list(x[1])),
                                 reverse=True)
              cur_hyps = next_hyps[:self.beam_size]

          hyps = [(y[0], self._log_add([y[1][0], y[1][1]])) for y in cur_hyps]
          if(self.lexicon is not None):
            out=[]
            for id in hyps[0][0]:
              out.append(self.lexicon[id])
          else:
            out=hyps
          return out

    def prefix_beam_search_contextbias(self,ctc_probs, lm=None,lm_weight=0.3):
          bias_score=3
          maxlen = ctc_probs.shape[0]
          cur_hyps = [(tuple(), (0.0, -float('inf'),'',0.0))]
          # 2. CTC beam search step by step
          for t in range(0, maxlen):
              logp = ctc_probs[t]  # (vocab_size,)
              # key: prefix, value (pb, pnb), default value(-inf, -inf)
              next_hyps = defaultdict(lambda: (-float('inf'), -float('inf'),[],0.0))
              # 2.1 First beam prune: select topk best
              top_k_logp, top_k_index = logp.topk(self.beam_size)  # (beam_size,)
              for s in top_k_index:
                  s = s.item()
                  if(self.lexicon[s] not in self.context_boundary):
                     ps = logp[s].item()-1.0
                  else:
                     ps = logp[s].item()
                  for prefix, (pb, pnb,context,back_score) in cur_hyps:
                      last = self.lexicon.index(prefix[-1]) if len(prefix) > 0 else None
                      if s == self.blankID:  # blank
                          n_pb, n_pnb,_,_ = next_hyps[prefix]
                          n_pb = self._log_add([n_pb, pb + ps, pnb + ps])
                          next_hyps[prefix] = (n_pb, n_pnb,context,back_score)
                      elif s == last:
                          #  Update *ss -> *s;
                          n_pb, n_pnb,_,_ = next_hyps[prefix]
                          n_pnb = self._log_add([n_pnb, pnb + ps])
                          next_hyps[prefix] = (n_pb, n_pnb,context,back_score)
                          # Update *s-s -> *ss, - is for blank
                          n_prefix = prefix + (self.lexicon[s], )
                          n_pb, n_pnb,_,_ = next_hyps[n_prefix]
                          n_pnb = self._log_add([n_pnb, pb + ps])
                          next_hyps[n_prefix] = (n_pb, n_pnb,context,back_score)
                      else:
                          n_prefix = prefix + (self.lexicon[s], )
                          n_pb, n_pnb,_,_ = next_hyps[n_prefix]
                          n_pnb = self._log_add([n_pnb, pb + ps, pnb + ps])
                          if(len(context)==0 and self.lexicon[s] in self.context_head):
                              context=self.conetxt_bias_list[self.context_head.index(self.lexicon[s])].copy()
                              back_score+=bias_score
                              n_pnb=min(-0.01,n_pnb+bias_score)
                              context.pop(0)
                          elif(len(context) and self.lexicon[s]==context[0]):
                              back_score+=bias_score
                              n_pnb=min(-0.01,n_pnb+bias_score)
                              context.pop(0)
                          elif(len(context) and self.lexicon[s]!=context[0]):
                              n_pnb-=back_score
                              back_score=0.0
                              context=[]

                          if(len(context)==0 and back_score!=0):
                              back_score=0
                          next_hyps[n_prefix] = (n_pb, n_pnb,context,back_score)
              # 2.2 Second beam prune
              if(lm==None):
                next_hyps = sorted(next_hyps.items(),key=lambda x: self._log_add(list(x[1][:2])),reverse=True)
              else:
                next_hyps = sorted(next_hyps.items(), key=lambda x: self._log_add_lm(lm=lm,args=list(x)), reverse=True)
              cur_hyps = next_hyps[:self.beam_size]

          hyps = [(y[0], self._log_add([y[1][0], y[1][1]])) for y in cur_hyps]
          return hyps[0][0]



