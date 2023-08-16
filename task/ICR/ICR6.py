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
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
###変化を加えたい###

# データ準備
file_path = '/Users/kaguy/.vscode/task/ICR/train.csv'
train = pd.read_csv(file_path)

file_path = '/Users/kaguy/.vscode/task/ICR/test.csv'
test = pd.read_csv(file_path)
test_x = test.drop(columns=[test.columns[0]])
test_id = test[test.columns[0]]
label_encoder = LabelEncoder()
test_x["EJ"] = label_encoder.fit_transform(test_x["EJ"])
test_x["EJ"] = test_x["EJ"].astype("category")

train_x = train.drop(columns=[train.columns[0], train.columns[57]])
label_encoder = LabelEncoder()
train_x["EJ"] = label_encoder.fit_transform(train_x["EJ"])
train_x["EJ"] = train_x["EJ"].astype("category")
train_y = train[train.columns[57]]

scores = []
models =[]

# K分割
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=71)

for tr_idx, va_idx in kf.split(train_x, train_y):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # LightGBMの実装
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(va_x, va_y)

    categorical_features = ['EJ']

    model = LGBMClassifier(
        boosting_type='gbdt',
        max_depth=4,
        num_leaves=15,
        n_estimators=1000,
        objective='binary',
        class_weight='balanced',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

    model.fit(
    lgb_train.data,
    lgb_train.label,
    categorical_feature=categorical_features,
    eval_set=[(lgb_train.data, lgb_train.label), (lgb_eval.data, lgb_eval.label)], 
    eval_names=['train', 'valid'],
    )

    models.append(model)

    va_pre = model.predict_proba(va_x)
    va_pre[va_x["BQ"] == 0] = 0
    va_pre[va_pre > 0.98] = 1
    va_pre[va_pre < 0.02] = 0

    va_y0 = 1 - va_y
    va_y1 = np.column_stack([va_y0,va_y])

    score = log_loss(va_y1, va_pre)
    scores.append(score)

print(np.mean(scores))

pred = np.array([model.predict(test_x) for model in models])