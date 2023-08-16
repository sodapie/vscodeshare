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

params = {
    'objective': 'regression_l1',
    'seed': 71,
    'metric': 'binary_logloss',
    'max_depth': 4,
    'num_leaves':15,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1
}
num_round = 10000

models = []

# K分割
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=71)

for tr_idx, va_idx in kf.split(train_x, train_y):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # LightGBMの実装
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(va_x, va_y)

    categorical_features = ['EJ']
    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=num_round,
                      categorical_feature=categorical_features,
                      valid_names=['train', 'valid'],
                      valid_sets=[lgb_train, lgb_eval],
                      callbacks=[lgb.early_stopping(stopping_rounds=50)]
                      )
    
    models.append(model)

    va_pred = model.predict(va_x)

    va_pred[va_x["BQ"] == 0] = 0
    va_pred[va_pred > 0.98] = 1
    va_pred[va_pred < 0.02] = 0

pred = np.array([model.predict(test_x) for model in models])
# k 個のモデルの予測値の平均 shape = (N_test,).
pred = np.mean(pred, axis=0)

pred
pred0 = 1 - (pred)
test_id

submission_df = pd.DataFrame({
    "Id": test_id,
    "class_0": pred0,
    "class_1": pred
})

for i, model in enumerate(models):
    model.save_model(f'ICRmodel{i+1}.txt')

model1 = lgb.Booster(model_file='ICRmodel1.txt')

# CSVファイルとして保存
submission_df.to_csv("submission.csv", index=False)

