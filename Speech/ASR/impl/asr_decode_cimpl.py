import _ctypes
import ctypes
import numpy as np

# dynamic library
lib = None

# function pointer dictionary
fun_dict = {}

def __load_c_functions():
    
    global lib, fun_dict
    
    lib = ctypes.cdll.LoadLibrary('/home/huanyuan/code/demo/Speech/ASR/lib/Release/libai_speech_asr.so')

    lib.Decode_create.argtypes = []
    lib.Decode_create.restype = ctypes.c_void_p
    fun_dict['Decode_create'] = lib.Decode_create

    lib.Decode_delete.argtypes = [ctypes.c_void_p]
    lib.Decode_delete.restype = None
    fun_dict['Decode_delete'] = lib.Decode_delete

    lib.Decode_ctc_decoder.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    lib.Decode_ctc_decoder.restype = None
    fun_dict['Decode_ctc_decoder'] = lib.Decode_ctc_decoder

    lib.Decode_length.argtypes = [ctypes.c_void_p]
    lib.Decode_length.restype = ctypes.c_int
    fun_dict['Decode_length'] = lib.Decode_length

    lib.Decode_copy_result_id_to.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.Decode_copy_result_id_to.restype = None
    fun_dict['Decode_copy_result_id_to'] = lib.Decode_copy_result_id_to

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


class Decode(object):
  """ decode python wrapper """
  def __init__(self):
    self.ptr = call_func('Decode_create')

  def __del__(self):
    call_func('Decode_delete', self.ptr)
    self.ptr = None

  def ctc_decoder(self, input_data):
    rows = input_data.shape[0]
    cols = input_data.shape[1]
    data = np.zeros((rows*cols), dtype=np.float32)
    for idx in range(rows):
        for idy in range(cols):
            data[idx * cols + idy] = input_data[idx, idy]
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    call_func('Decode_ctc_decoder', self.ptr, data_ptr, rows, cols)
    return 

  def data_length(self):
    return call_func('Decode_length', self.ptr)

  def result_id_to_numpy(self):
    data_length = self.data_length()  
    data = np.empty((data_length), dtype=np.int32)
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    call_func('Decode_copy_result_id_to', self.ptr, data_ptr)
    return data

load()