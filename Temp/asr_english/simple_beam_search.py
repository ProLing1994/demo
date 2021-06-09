import kenlm
import numpy as np
import copy
import heapq

dim = 32
prob = np.log( np.random.random( [100,dim] ) )
print( np.argmax(prob,1) )

blankID = dim - 1
id2word = dict( (i, str(i)) for i in range(dim) )
id2word[blankID] = "<blank>"
lmPath = "./checkpoint/lm_mandarin_408.bin"

class Ken_LM:

  def __init__(self,lmPATH):
    self.__model = kenlm.Model(lmPATH)
    self.senlen=3

  def compute(self,state,word):
    assert isinstance(state,list)
    state.append( word )
    sentence = " ".join(state)
    prob = list(self.__model.full_scores( sentence , bos=False,eos=False))[-1][0]  #输入需要显式指定<s>起始符,不会默认添加，然后忽略eos终止符，不会为输入的sentence添加默认终止符</s>
    if len(state) < self.__model.order:
      return state, prob
    else:
      return state[1:], prob


def ctc_beam_search_rescore(prob, lm, beamSize, blankID, id2word, bswt=1.0, lmwt=1.0):
  frames, dims = prob.shape

  # initilize LM score (一定要用一个非0的值来初始化，我根据经验，使用未知词的概率来初始化。)
  _, initLMscore = lm.compute(state=[], word="UNK")
  initLMscore *= lmwt
  results = [{"words": [], "ali": [], "bs": 0, "lm": initLMscore, "total": 0, "lmState": ["<s>"]}]

  for i in range(frames):
    temp = []
    for preResult in results:
      for d in range(dims):

        one = copy.deepcopy(preResult)
        # 1. compute beam search score
        bsScore = prob[i, d]
        one["bs"] += bswt * bsScore
        # 2. compute LM score
        if d != blankID:  # ignore blank
          if len(one["ali"]) == 0 or d != one["ali"][-1]:  # ignore continuous repeated symbols
            word = id2word[d]
            newState, lmScore = lm.compute(state=one["lmState"], word=word)
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
    results = temp[:beamSize]

  # post-processing ( append </s> symbol and update LM score )
  for one in results:
    newState, lmScore = lm.compute(state=one["lmState"], word="</s>")
    one["lm"] += lmwt * lmScore
    numFrames = len(one["ali"])
    numWords = len(one["words"]) + 1
    one["total"] = one["bs"] / numFrames + one["lm"] / numWords
    # discard <s>
    one["words"] = " ".join(one["words"])
    one["lmState"] = newState

  return sorted(results, key=lambda x: x["total"], reverse=True)


def ctc_beam_search(prob, lm, beamSize, blankID, id2word, bswt=1.0, lmwt=1.0,results=[]):

  frames, dims = prob.shape

  # initilize LM score (一定要用一个非0的值来初始化，我根据经验，使用未知词的概率来初始化。)
  _, initLMscore = lm.compute( state=[], word="UNK" )
  initLMscore *= lmwt
  if(len(results)==0):
    results = [ { "words":[], "ali":[], "bs":0, "lm":initLMscore, "total":0, "lmState":["<s>"],"wordscore":[] } ]

  for i in range(frames):
    temp = []
    testList = prob[i]
    tmp = zip(range(len(testList)), testList)
    large5 = heapq.nlargest(5, tmp, key=lambda x: x[1])
    for preResult in results:
      for (d,bsScore) in large5:
        if(bsScore<-1.6):
          continue

        # print("id: ", d, ", bs_score: ", bsScore)
        one = copy.deepcopy( preResult )
        # 1. compute beam search score
        one["bs"] += bswt * bsScore
        # 2. compute LM score
        if d != blankID: # ignore blank
          if len(one["ali"]) == 0 or d != one["ali"][-1]: # ignore continuous repeated symbols
            word = id2word[d]
            newState, lmScore = lm.compute( state=one["lmState"], word=word )
            one["wordscore"].append([word,bsScore])
            one["words"].append(word)
            one["lm"] += lmwt * lmScore
            one["lmState"] = newState
        # 3. record alignment
        one["ali"].append(d)
        # 4. compute total score with length normalization
        numFrames = len(one["ali"])
        numWords = len(one["words"]) + 1
        one["total"] = one["bs"] / numFrames + one["lm"] / numWords

        temp.append( one )
      # prune
    temp = sorted(temp, key=lambda x:x["total"], reverse=True)
    if(len(temp)==0):
      continue
    results = temp[:beamSize]
  '''
  # post-processing ( append </s> symbol and update LM score )
  for one in results:
    newState, lmScore = lm.compute( state=one["lmState"], word="</s>" )
    one["lm"] += lmwt * lmScore
    numFrames = len(one["ali"])
    numWords = len(one["words"]) + 1
    one["total"] = one["bs"] / numFrames + one["lm"] / numWords
    # discard <s>
    one["words"] = " ".join( one["words"] )
    one["lmState"] = newState
  '''
  return sorted(results, key=lambda x:x["total"], reverse=True)

#results = ctc_beam_search(prob, lm, beamSize=5, blankID=blankID, id2word= id2word, bswt=1.0, lmwt=1.0 )
#print( results )
  








    



      





