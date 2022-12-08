import re
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
from TTS.dataset.text.pinyin import pinyin_to_symbols
from TTS.dataset.text.symbols import symbols
from TTS.dataset.text import cleaners

# Mappings from symbol to numeric ID and vice versa:
_cur_lang = None
_symbol_to_id = None
_id_to_symbol = None

def change_lang(lang):
  global _cur_lang, _symbol_to_id, _id_to_symbol
  if _cur_lang != lang:
    _symbol_to_id = {s: i for i, s in enumerate(symbols(lang))}
    _id_to_symbol = {i: s for i, s in enumerate(symbols(lang))}
    _cur_lang = lang

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names, lang):
  """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  """
  change_lang(lang)

  sequence = []

  # Chinese Pinyin symbols (initial, final, tone, etc)
  if lang == 'py':
    sequence += _symbols_to_sequence(pinyin_to_symbols(text))

  # other symbols (English characters)
  else:
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
      m = _curly_re.match(text)
      if not m:
        sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
        break
      sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
      sequence += _arpabet_to_sequence(m.group(2))
      text = m.group(3)

  # Append EOS token
  sequence.append(_symbol_to_id["~"])
  return sequence


def sequence_to_text(sequence, lang):
  """Converts a sequence of IDs back to a string"""
  change_lang(lang)
  result = ""
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == "@":
        s = "{%s}" % s[1:]
      result += s
  return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception("Unknown cleaner: %s" % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s not in ("_", "~")


if __name__ == "__main__":
  test_symbols = text_to_sequence("CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED", cleaner_names=["english_cleaners"], lang='en')
  print(test_symbols)
  print([_id_to_symbol[s] for s in test_symbols])

  test_symbols = text_to_sequence("chi2 qi3 / hong2 ying1 qiang1 / zhui1 gan3 / dui4 fang1 / ban4 gong1 li3 ", cleaner_names=["basic_cleaners"], lang='en')
  print(test_symbols)
  print([_id_to_symbol[s] for s in test_symbols])

  test_symbols = text_to_sequence("ma1-ma1 dang1-shi2 biao3-shi4, er2-zi5 kai1-xin1-de5 / xiang4-huar1 yi2-yang4.", cleaner_names=["basic_cleaners"], lang='en')
  print(test_symbols)
  print([_id_to_symbol[s] for s in test_symbols])

  test_symbols = text_to_sequence("ma1-ma1 dang1-shi2 biao3-shi4, er2-zi5 kai1-xin1-de5 / xiang4-huar1 yi2-yang4.", cleaner_names=["basic_cleaners"], lang='py')
  print(test_symbols)
  print([_id_to_symbol[s] for s in test_symbols])
