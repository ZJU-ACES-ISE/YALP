import os


def check_dir(file):
    dir = os.path.dirname(file)
    if not os.path.exists(dir):
        os.makedirs(dir)
