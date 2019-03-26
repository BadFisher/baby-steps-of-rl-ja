"""
Kerasのtutorialにを行う
tensorflow版(tf)もある
エンドポイント: ./FN/nn_tutorial/explanation_keras_*.py
"""
import numpy as np
from tensorflow.python import keras as K

#2-layers
model = K.Sequential([
    # K.layers.Dense(units=4, input_shape=((2, ))),
    K.layers.Dense(units=4, input_shape=((2, )), activation="sigmoid"),
    K.layers.Dense(units=4),
])

weight, bias = model.layers[0].get_weights()
print("Weight shape is {}".format(weight.shape))
print("Weight is {}".format(weight))
print("Bias shape is {}".format(bias.shape))
print("Bias is {}".format(bias))

x = np.random.rand(1,2)
x
y = model.predict(x)
print("shape of x is {}\nshape of y is {}".format(x.shape,y.shape))
print("x is {} \ny is {}".format(x,y))


# Batchサイズ3で入力してみる
batch = np.random.rand(3,2)
y_batch = model.predict(batch)
print("shape of batch is {}\nshape of y_batch is {}".format(batch.shape,y_batch.shape))
print("batch is {} \ny_batch is {}".format(batch,y_batch))

"""
ボストンの住宅価格予測モデルの学習
"""
import numpy as np
from sklearn.model_selection import train_test_split
# ボストン住宅データ
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.python import keras as K

dataset = load_boston()
type(dataset)

# 住宅価格
y = dataset.target
np.max(y)
np.average(y)
np.min(y)
# 13の特徴量(住宅の部屋数, 人口1人あたりの犯罪発生数)
X = dataset.data

print(type(y), y.shape)
print(type(X), X.shape)
X[0]
y[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print (X_train.shape, X_test.shape)
print (y_train.shape, y_test.shape)

model = K.Sequential([
    # 13の特徴量の平均が0, 分散が1になるよう正規化
    K.layers.BatchNormalization(input_shape=((13, ))),
    # regularization: 正則化 -> 重みの値に制限をかけて過学習防止
    K.layers.Dense(units=13, activation="softplus", kernel_regularizer="l1"),
    K.layers.Dense(units=1),
])

# 学習の基本の流れ: 勾配を計算し、Optimizerにより適用
model.compile(loss="mean_squared_error", optimizer="sgd")
# 学習スタート
model.fit(X_train, y_train, epochs=8)

# テスト
predicts = model.predict(X_test)
predicts.shape
predicts.reshape(-1,).shape

result = pd.DataFrame({
    "predict": np.reshape(predicts, (-1,)),
    "actual": y_test
})
limit = max(np.max(predicts),np.max(y_test))
limit

# y=xのグラフに近づいているほど予測精度がよい
result.plot.scatter(x="actual", y="predict", xlim=(0, limit), ylim=(0, limit))

"""
CNNを使ったMnistの学習
"""
import numpy as np
from sklearn.model_selection import train_test_split
# Mnist
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report # 最後に使用
from tensorflow.python import keras as K

dataset = load_digits()
type(dataset)

# 8x8のグレースケールへの変換用
image_shape = (8, 8, 1)
# one-hotベクトルへの変換用
num_class = 10

# 入力画像データ
X = dataset.data
X.shape # (1797, 64)
X = np.array([data.reshape(image_shape) for data in X])
X.shape # (1797, 8, 8, 1)

# 0-9の数字
y = dataset.target
y.shape # (1797,)
y = K.utils.to_categorical(y, num_class) # one-hotベクトル化
y.shape # (1797, 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print (X_train.shape, X_test.shape)
print (y_train.shape, y_test.shape)

model = K.Sequential([
    K.layers.Conv2D(
            # 3x3のフィルタを5枚
            5, kernel_size=3, strides=1, padding="same",
            input_shape=image_shape, activation="relu"),
    K.layers.Conv2D(
            # 2x2のフィルタを3枚
            3, kernel_size=2, strides=1, padding="same",
            activation="relu"),
    K.layers.Flatten(),
    K.layers.Dense(units=num_class, activation="softmax")
])
predicts1 = model.predict(X_train[0].reshape(1,8,8,1))
predicts1.shape
predicts1
# 学習の基本の流れ: 勾配を計算し、Optimizerにより適用
model.compile(loss="categorical_crossentropy", optimizer="sgd")
# 学習スタート
model.fit(X_train, y_train, epochs=8)

# テスト
predicts = model.predict(X_test)
predicts = np.argmax(predicts, axis=1)
actual = np.argmax(y_test, axis=1)
print(classification_report(actual, predicts))
