import sys
import os

def add_path(custom_path):
    if custom_path not in sys.path: sys.path.insert(0, custom_path)

this_dir = os.path.dirname(__file__)

lib_path = os.path.join(this_dir, '..')
add_path(lib_path)