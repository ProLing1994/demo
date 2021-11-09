"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from TTS.dataset.text import cmudict
from TTS.dataset.text import pinyin

_pad        = '_'
_eos        = '~'

# 英文字符模板
# _characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? "
# 兼容中文字符模板(old)s
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!/\'\"(),-.:;?# "
# # 兼容中文字符模板(new)
# _characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? %/'
# _digits     = '0123456789'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# English characters, 字母级别的建模
en_symbols = [_pad, _eos] + list(_characters) #+ _arpabet
# en_symbols = [_pad, _eos] + list(_characters) + list(_digits) #+ _arpabet

# Chinese Pinyin symbols (intitial, final, tone, etc)
zh_py_symbols = [_pad, _eos] + pinyin.symbols

# Export symbols according to language tag
def symbols(lang):
  if lang == 'py':
    return zh_py_symbols
  elif lang == 'en':
    return en_symbols
  else:
    raise NameError('Unknown target language: %s' % str(lang))