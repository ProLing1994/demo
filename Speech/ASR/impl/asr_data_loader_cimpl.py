import _ctypes
import ctypes
import numpy as np

# dynamic library
lib = None

# function pointer dictionary
fun_dict = {}

class Wave_Data_S(ctypes.Structure):   
    _fields_ =[('data', ctypes.POINTER(ctypes.c_short)),  
                ('data_length', ctypes.c_uint),
                ('fs', ctypes.c_ushort)] 

def __load_c_functions():
    
    global lib, fun_dict
    
    lib = ctypes.cdll.LoadLibrary('/home/huanyuan/code/demo/Speech/ASR/lib/Release/libai_speech_asr.so')

    lib.ReadWaveLength.argtypes = [ctypes.c_char_p]
    lib.ReadWaveLength.restype = ctypes.c_uint
    fun_dict['ReadWaveLength'] = lib.ReadWaveLength

    lib.ReadWave.argtypes = [ctypes.c_char_p,
                            ctypes.POINTER(Wave_Data_S)]
    lib.ReadWave.restype = None
    fun_dict['ReadWave'] = lib.ReadWave

    lib.Wave_copy_data_to.argtypes = [ctypes.POINTER(Wave_Data_S),
                                      ctypes.POINTER(ctypes.c_short)]
    lib.Wave_copy_data_to.restype = None
    fun_dict['Wave_copy_data_to'] = lib.Wave_copy_data_to

def unload():
    
  global lib, fun_dict

  try:
    while lib is not None:
      if platform.system() == 'Windows':
            _ctypes.FreeLibrary(lib._handle)
      else:
        _ctypes.dlclose(lib._handle)
  except:
    lib = None
    fun_dict = {}


def load():

  unload()
  __load_c_functions()


def load_c_functions_if_necessary():
    
  if len(fun_dict) == 0:
    print('[info] bone age infer dll reloaded')
    __load_c_functions()


def call_func(func_name, *args):
    
  load_c_functions_if_necessary()

  if len(args) == 0:
    return fun_dict[func_name]()
  else:
    return fun_dict[func_name](*args)


class WavLoader(object):
  """ wav loader python wrapper """

  def ReadWave(audio_path="/home/huanyuan/share/audio_data/xiaorui_12162020_training_60_001.wav"):
    wava_data_s = Wave_Data_S()
    call_func('ReadWave', audio_path.encode(), wava_data_s)

    wava_data = np.empty((wava_data_s.data_length), dtype=np.int16)
    data_ptr = wava_data.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
    call_func('Wave_copy_data_to', wava_data_s, data_ptr)
    return wava_data

load()