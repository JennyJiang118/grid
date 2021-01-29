import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_data(path, base_line):
    sys.path.insert(0,".")
    data = pd.read_excel(path).dropna(axis=0, how='any').reset_index()
    data['gap'] = data['spread'] - data[base_line]
    return data



def get_level(data):
    # 所有点值的平均
    level = data.

def main():
    path=r"data/IF03-12.xlsx"
    base_line = "avg3"
    data = load_data(path, base_line)