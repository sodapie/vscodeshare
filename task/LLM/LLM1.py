#git push origin HEAD を使ってログインした後、コミット&プッシュをする
#git config --local user.name "piyo"
#git config --local user.email piyo@example.com

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

file_path = 'C:\Users\kaguy\.vscode\vscodeshare\task\LLM\train.csv'
train = pd.read_csv(file_path)