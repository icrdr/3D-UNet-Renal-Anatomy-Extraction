import json
import numpy as np


def json_save(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def json_load(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
