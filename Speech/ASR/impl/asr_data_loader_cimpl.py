import _ctypes
import ctypes
import numpy as np

# dynamic library
lib = None

# function pointer dictionary
fun_dict = {}

# class Wave_Data_S(ctypes.Structure):   
#   _fields_ =[('data', ctypes.POINTER(ctypes.c_short)),  
#               ('data_length', ctypes.c_uint),
#               ('fs', ctypes.c_ushort)] 

def __load_c_functions():
    
  global lib, fun_dict
  
  lib = ctypes.cdll.LoadLibrary('/home/huanyuan/code/demo/Speech/ASR/lib/Release/libai_speech_asr.so')
  
  lib.Wave_Data_create.argtypes = []
  lib.Wave_Data_create.restype = ctypes.c_void_p
  fun_dict['Wave_Data_create'] = lib.Wave_Data_create
  
  lib.Wave_Data_delete.argtypes = [ctypes.c_void_p]
  lib.Wave_Data_delete.restype = None
  fun_dict['Wave_Data_delete'] = lib.Wave_Data_delete

  lib.Wave_Data_load_data.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
  lib.Wave_Data_load_data.restype = ctypes.c_int
  fun_dict['Wave_Data_load_data'] = lib.Wave_Data_load_data

  lib.Wave_Data_length.argtypes = [ctypes.c_void_p]
  lib.Wave_Data_length.restype = ctypes.c_int
  fun_dict['Wave_Data_length'] = lib.Wave_Data_length

  lib.Wave_Data_copy_data_to.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_short)]
  lib.Wave_Data_copy_data_to.restype = None
  fun_dict['Wave_Data_copy_data_to'] = lib.Wave_Data_copy_data_to

  lib.Wave_Data_clear_state.argtypes = [ctypes.c_void_p]
  lib.Wave_Data_clear_state.restype = None
  fun_dict['Wave_Data_clear_state'] = lib.Wave_Data_clear_state

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
    print('[info] dll reloaded')
    __load_c_functions()


def call_func(func_name, *args):
    
  load_c_functions_if_necessary()

  if len(args) == 0:
    return fun_dict[func_name]()
  else:
    return fun_dict[func_name](*args)


class WaveLoader(object):
  """ wav loader python wrapper """

  def __init__(self):
    self.ptr = call_func('Wave_Data_create')

  def __del__(self):
    call_func('Wave_Data_delete', self.ptr)
    self.ptr = None

  def load_data(self, audio_path="/home/huanyuan/share/audio_data/xiaorui_12162020_training_60_001.wav"):
    call_func('Wave_Data_load_data', self.ptr, audio_path.encode())
    return 

  def data_length(self):
    return call_func('Wave_Data_length', self.ptr)

  def to_numpy(self):
    data_length = self.data_length()  
    wava_data = np.empty((data_length), dtype=np.int16)
    wave_data_ptr = wava_data.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
    call_func('Wave_Data_copy_data_to', self.ptr, wave_data_ptr)
    return wava_data

load()