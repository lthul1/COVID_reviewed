import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
import pandas as pd
import pickle

def load_data(name):
    try:
        with open(name, 'rb') as f:
            data = pickle.load(f)
    except:
        data = []
    return data

def save_data(data, name):
	with open(name, 'wb') as f:
		pickle.dump(data, f)