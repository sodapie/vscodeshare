import pandas as pd

# インストールしたいとき
# python -m pip install --user sklearn

# アンインストールしたいとき
# python -m pip uninstall transformers

#情報を知りたいとき
import numpy as np
train = np.array[1,2,3]
train.head()
train.info()
train.shape
train.columns

#データの種類数をカウントしたいとき
pathforanime = 'atmaCup/anime.csv'
anime = pd.read_csv(pathforanime)

from collections import Counter
c = Counter(anime["source"])
print(len(c))

#データの結合
df_v = pd.concat([df1, df2], axis=0)

#データの把握
import seaborn as sns

df_iris = sns.load_dataset("iris")
sns.pairplot(df_iris)



#Lightgcnを使う
#heterogeneous graphを使う
#graph neural networkが強かった
#Lightgbmが強いが、アンサンブルで上記の解放を組み込む
#重みづけはスタッキングをsklearnのLinearRegressionでもできる

#Lightgbmの特徴量
#Implicit Feedback