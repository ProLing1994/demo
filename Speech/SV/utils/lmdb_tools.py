import lmdb
import numpy as np
import os

def load_lmdb_env(lmdb_path):
    lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    return lmdb_env


def read_audio_lmdb(env, key):
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode())
    audio_data = np.frombuffer(buf, dtype=np.float32)
    return audio_data