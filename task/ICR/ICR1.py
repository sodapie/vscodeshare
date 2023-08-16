import csv
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
from sklearn import preprocessing
import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/Users/kaguy/.vscode/task/ICR/train.csv'
train = pd.read_csv(file_path)

train.head()

train_x = train.drop(columns=train.columns[57])
train_y = train[train.columns[57]]

train_x.head()
train_y.head()

x = np.array([1,2,3,4])

plt.hist(x)