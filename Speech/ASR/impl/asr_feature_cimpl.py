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

    lib.GetFeatureTime.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int]
    lib.GetFeatureTime.restype = ctypes.c_int
    fun_dict['GetFeatureTime'] = lib.GetFeatureTime


    lib.GetMelIntFeature.argtypes = [ctypes.POINTER(ctypes.c_short),
                                    ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_ubyte),
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int]
    lib.GetMelIntFeature.restype = None
    fun_dict['GetMelIntFeature'] = lib.GetMelIntFeature

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


class Feature(object):
    """ feature python wrapper """

    def GetMelIntFeature(data, data_len, n_fft=256, sample_rate=8000, time_seg_ms=32, time_step_ms=10, feature_freq=48, nfilt=64, scale_num=10):
        
        data_len = ctypes.c_int(data_len)
        sample_rate = ctypes.c_int(sample_rate)
        time_seg_ms = ctypes.c_int(time_seg_ms)
        time_step_ms = ctypes.c_int(time_step_ms)

        feature_time = fun_dict['GetFeatureTime'](data_len, sample_rate, time_seg_ms, time_step_ms)

        data = np.ascontiguousarray(data, dtype=np.int16)
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_short))

        feature_data = np.zeros((feature_time, feature_freq), dtype=np.uint8)
        feature_data_ptr = feature_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        n_fft = ctypes.c_int(n_fft)
        feature_freq = ctypes.c_int(feature_freq)
        nfilt = ctypes.c_int(nfilt)
        scale_num = ctypes.c_int(scale_num)

        fun_dict['GetMelIntFeature'](data_ptr, data_len, feature_data_ptr, n_fft, sample_rate, time_seg_ms, time_step_ms, feature_freq, nfilt, scale_num)
        return feature_data

load()