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

    lib.Feature_create.argtypes = []
    lib.Feature_create.restype = ctypes.c_void_p
    fun_dict['Feature_create'] = lib.Feature_create

    lib.Feature_create_samples.argtypes = [ctypes.c_int]
    lib.Feature_create_samples.restype = ctypes.c_void_p
    fun_dict['Feature_create_samples'] = lib.Feature_create_samples

    lib.Feature_delete.argtypes = [ctypes.c_void_p]
    lib.Feature_delete.restype = None
    fun_dict['Feature_delete'] = lib.Feature_delete

    lib.Feature_data_mat_time.argtypes = [ctypes.c_void_p]
    lib.Feature_data_mat_time.restype = ctypes.c_int
    fun_dict['Feature_data_mat_time'] = lib.Feature_data_mat_time

    lib.Feature_feature_time.argtypes = [ctypes.c_void_p]
    lib.Feature_feature_time.restype = ctypes.c_int
    fun_dict['Feature_feature_time'] = lib.Feature_feature_time

    lib.Feature_feature_freq.argtypes = [ctypes.c_void_p]
    lib.Feature_feature_freq.restype = ctypes.c_int
    fun_dict['Feature_feature_freq'] = lib.Feature_feature_freq

    lib.Feature_check_data_length.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.Feature_check_data_length.restype = ctypes.c_int
    fun_dict['Feature_check_data_length'] = lib.Feature_check_data_length

    lib.Feature_get_mel_int_feature.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_short), ctypes.c_int]
    lib.Feature_get_mel_int_feature.restype = None
    fun_dict['Feature_get_mel_int_feature'] = lib.Feature_get_mel_int_feature

    lib.Feature_copy_mfsc_feature_int_to.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
    lib.Feature_copy_mfsc_feature_int_to.restype = None
    fun_dict['Feature_copy_mfsc_feature_int_to'] = lib.Feature_copy_mfsc_feature_int_to

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
    def __init__(self, data_len_samples=48000):
        if data_len_samples == 48000:
            self.ptr = call_func('Feature_create')
        else:
            data_len_samples = ctypes.c_int(data_len_samples)
            self.ptr = call_func('Feature_create_samples', data_len_samples)

    def __del__(self):
        call_func('Feature_delete', self.ptr)
        self.ptr = None
  
    def data_mat_time(self):
        return call_func('Feature_data_mat_time', self.ptr)
        
    def feature_time(self):
        return call_func('Feature_feature_time', self.ptr)

    def feature_freq(self):
        return call_func('Feature_feature_freq', self.ptr)

    def check_feature_time(self, data_len_samples):
        data_len_samples = ctypes.c_int(data_len_samples)
        return call_func('Feature_check_data_length', self.ptr, data_len_samples)

    def get_mel_int_feature(self, data, data_len_samples):
        data = np.ascontiguousarray(data, dtype=np.int16)
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_short))

        data_len_samples = ctypes.c_int(data_len_samples)
        call_func('Feature_get_mel_int_feature', self.ptr, data_ptr, data_len_samples)
        return 

    def copy_mfsc_feature_int_to(self):
        feature_data = np.zeros((self.data_mat_time(), self.feature_freq()), dtype=np.uint8)
        feature_data_ptr = feature_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        call_func('Feature_copy_mfsc_feature_int_to', self.ptr, feature_data_ptr)
        return feature_data

load()