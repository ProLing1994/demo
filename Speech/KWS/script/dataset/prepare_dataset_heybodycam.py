import argparse
import glob
import os
import shutil
import re


def get_hash_name(file_name):
    if 'HEYBODYCAM_' in file_name:
        hash_name = file_name.strip().split('_')[-1].split('.')[0][:6]
    else:
        hash_name = file_name.strip().split('.')[0]
    return hash_name