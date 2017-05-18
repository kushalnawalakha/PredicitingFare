import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

testset=pd.read_csv("Data/test.csv",sep=",")
dataset=pd.read_csv("Data/train.csv",sep=",")

