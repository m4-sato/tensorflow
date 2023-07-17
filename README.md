# TensorFlowメモ

[TensorFlow公式ドキュメント](https://www.tensorflow.org/)
[からっぽのしょこ](https://www.anarchive-beta.com/)

## tensorflow⇔numpy

- Tensor→Arrayの変換方法：

```python
変換したいTensor名.numpy()
```

- Array→Tensorの変換方法：

```python
tf.convert_to_tensor(変換したいArray名)
```

## 自動微分機能

自動微分機能を利用したい場合は````tf.GradientTape(グラディエントテープ)```を利用する。
コンテキスト内で行われる演算すべてをテープに記録して勾配を計算する。
今回は要素をスカラーで値を1としたが、行列でも良いし中身の値も何でも良い。(微分には関係ない)

```python
x = tf.Variable(1.0) # パラメータの定義
a, b = 3, 5

with tf.GradientTape() as tape:
    y = a*x + b
print(y)
```

```python
tf.Tensor(8.0, shape=(), dtype=float32)
```

## tensorを操作する関数

- tf.transpose：行列の転置

```python
x = tf.constant([[1, 2, 3],
                 [4, 5, 6]])
print(tf.transpose(x))
```
注意
NumPyのtransposeとは処理が違うので混同しないように注意が必要
Tensor名.transposeやTensor名.Tでは転置ができない

- tf.reshape：行列の形（サイズ）の変更

```python
x = tf.constant([1, 2, 3, 4, 5, 6])
print(tf.reshape(x, [2, 3]), "\n")
print(tf.reshape(x, [2, -1]))
```

tf.reshape(Tensor名, [行, 列])で形状を指定できる
行や列の引数に-1を指定することで他方の次元の長さに合わせて自動で形状変換が可能。

## 全結合層(Fully Connected Layer)

```python   
tf.keras.layers.Dense(ユニット数, activation=活性化関数)
```

例
```python
#4次元ベクトルを2次元ベクトルに変換する全結合層をクラスとして定義
fc = layers.Dense(2)
x = tf.constant([[1., 2., 3.]])
# fcの__Call__()メソッドを呼ぶことでxを伝搬できる
x = fc(x)
print(x)
```

## 活性化関数

### ReLU関数・Softmax関数・Sigmoid関数

```python
x = tf.constant([-1.0, -0.5, 0.5, 1.0])
print("relu", tf.nn.relu(x))
print("softmax", tf.nn.softmax(x))
print("sigmoid", tf.math.sigmoid(x))
```


- 他の実装方法
  - 関数ではなく層としての実装：```layers.ReLU()```
  - 全結合層とセットで実装　　：```layers.Dense(units=units, activation="relu")```

## 損失関数

### 平均二乗誤差(MSE)MeanSquaredError

2つのベクトルの**要素ごとの差の2乗**から誤差を算出する。  
TensorFlowでは、`tf.keras.losses`モジュールで`MeanSquaredError(reduction=〇〇)`としてクラスが定義されている。

- 引数`reduction`について
  - reduction = losses_utils.ReductionV2.AUTO：要素ごとの誤差の平均 **(デフォルト)**
  - reduction = losses_utils.Reduction.SUM   ：要素ごとの誤差の合計
  - reduction = losses_utils.Reduction.NONE  ：要素ごとの誤差をテンソルとして出力

```python
from tensorflow.keras import losses

## pattern1
criterion = losses.MeanSquaredError()

x = tf.constant([0, 1, 2])
y = tf.constant([1, -1, 0])

loss = criterion(x, y)
print(loss)

## pattern2
x = tf.constant([[0.2, 0.5, 0.3]])
mse_label = tf.constant([[0, 1, 0]])
cel_label = tf.constant([1])

print('MSE:', losses.MeanSquaredError()(mse_label, x))
print('CrossEntropy:', losses.SparseCategoricalCrossentropy()(cel_label, x))


```

### 交差エントロピー誤差【SparseCategoricalCrossentropy】

「正解ラベルと対応する出力との差のみ」を計算する。
正しくこれを用いるためには、モデルの出力をSoftmax関数などで確率にする必要がある。
MSEと違いCEは、正解ラベルと対応する出力の誤差 "のみ" を計算する。

## 最適化関数

TensorFlowでは```tf.keras.optimizers```モジュールに最適化関数が用意されている。

```python
## pattern1
optimizer = optimizers.Adam(learning_rate=0.1)
print(optimizer)

## pattern2
from tensorflow.keras import optimizers

sgd = optimizers.SGD(learning_rate=0.01)
momentum = optimizers.SGD(learning_rate=0.01, momentum=0.9)
adam = optimizers.Adam(learning_rate = 0.01)

print('SGD:', sgd)
print('Momentum:', momentum)
print('Adam:', adam)
```

```

- その他パラメータ：beta_1, beta_2, epsilon

## モデルの定義

- TensorFlowでのモデルの定義方法は 3種類存在する。
  - Sequential(積層型)モデル：シンプルで簡単
  - Functional(関数型)API　 ：柔軟性があり複雑なモデルも定義できる
  - Subclassing(サブクラス)モデル：難易度は高いが、フルカスタマイズできる

### Sequential(積層型)モデル

```python
seq_model = Sequential()
seq_model.add(Input(shape=(3,)))
seq_model.add(layers.Dense(5, activation="relu"))
seq_model.add(layers.Dense(2))

criterion = losses.MeanSquaredError(),
optimizer = optimizers.Adam()

seq_model.summary()
```

- 補足説明
  - モデル名.add(層)でモデルに層を追加できる
  - Inputでは入力の形状(入力層のノード数)を指定する
  - ここで、「全結合層→ReLU→全結合層」という構造を組み立てている
  - モデル名.summary()により、モデルの構造とパラメータ数を表示することができる
  - Output Shape : 出力の形状(Noneはバッチサイズ数)
  - Param # : パラメータ数 (= 重み + バイアス)

### Functional(関数型)API

```python
inputs = Input(shape=(3,))
x = layers.Dense(5, activation="relu")(inputs)
outputs = layer.Dense(2)(x)

fun_model = Model(inputs=inputs, outputs=outputs)

criterion = losses.MeanSquaredError(),
optimizer = optimizers.Adam()

fun_model.summary()
```

### Subclassing(サブクラス)モデル

```python
class SubModel(Model):
    def _init__(self, hidden, output):
        super().__init__()

        self.fc1 = layers.Dense(hidden, activation="relu")
        self.fc2 = layers.Dense(output)
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

sub_model = SubModel(5,2)

criterion = losses.MeanSquaredError(),
optimizer = optimizers.Adam()

sub_model.build(input_shape=(None, 3))
sub_model.summary()
```

## モデルの学習

```python
inputs = Input(shape=(784,))
x = layers.Dense(512, activation="relu")(inputs)
outputs = layers.Dense(10)(x)

fun_model = Model(inputs=inputs, outputs=outputs)

criterion = losses.MeanSquaredError(),
optimizer = optimizers.Adam()

metrics = metrics.CateoricalAccuracy()

fun_model.summary()
```

### train関数(スクラッチで実装)

- 画像データを2次元から1次元に変換する
- 正解ラベルのone-hotベクトル化

```python
def train(model, x_train, y_train, batch_size=100, epochs=10):
    x_train = x_train.reshape(60000, 784).astype("float32")/ 255
    y_train = to_categorical(y_train)

    for epoch in range(epochs):
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=x_train)

        for x_batch_train, y_batch_train in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(x_batch_train, training=True)

                loss = criterion(y_batch_train, predictions)
            
            grads =tape.gradient(loss, model.trainable_weights)

        optimizer.apply_gradient(zip(loss, model.trainable_weights))

        metrics.update_state(y_batch_train, predictions)

    print("Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}".format(epoch+1, loss, metrics.result()*100))

    metrics.reset_states()

train(fun_model, x_train, y_train, batch_size=100, epochs=1)

```

### fit関数(自動)

TensorFlowにはこれらの学習を自動的を行ってくれる```fit()```関数が存在する。
```モデル名.fit()```で用いる
```fit(学習データ, 正解ラベル, バッチサイズ, エポック数)```の4つの引数を設定するだけ
```fit()```を行う前に```compile()```を用いて最適化や損失関数、評価指標を定義しておく

```python
X_train = x_train.reshape(60000, 784).astype("float32")/ 255
y_train = to_categorical(y_train)

fun_model.compile(optimizer=optimizer, loss = criterion, metrics = metrics)

fun_model.fit(x_train, y_train, batch_size=100, epochs=1)
```

### test関数(スクラッチで実装)

train関数から「損失計算」や「最適化」の要素を取り除く

```python
def test(model, x_test, y_test, batch_size=100):
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    y_test = to_categorical(y_test)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=x_test.shape[0]).batch(batch_size)

    for x_batch_test, y_batch_test in test_dataset:
        pred_label = model(x_batch_test)
        loss = criterion(pred_label, y_batch_test)
        metrics.update_state(pred_label, y_batch_test)
    
    return loss, metrics.result()*100

test_loss, test_acc = test(fun_model, x_test, y_test, batch_size =100)
print("Loss: {:.4f}, Accuracy: {:.2f}%".format(test_loss, test_acc))

```

### evaluate関数(自動)

- ```evaluate()```関数を用いると、```fit()```関数と同様に「損失計算」や「評価指標の計算」を自動で行ってくれる
- `モデル名.evaluate()`で用いる
- `evaluate(学習データ, 正解ラベル, バッチサイズ)`の3つの引数を設定するだけ
- 学習時に評価指標を設定したので`compile()`は必要ない

```python
x_test = x_test.reshape(10000, 784).astype("float32") / 255
y_test = to_categorical(y_test)

fun_model.evaluate(x_test, y_test, batch_size=100)
```

## レイヤー

### 全結合層(layers.Dense)

- 全ての入力ノードに対して全ての出力ノードへ演算（アフィン変換）を行う層で、モデルの出力層付近など、多くのケースで広く使われている。
- 重みとバイアスの2種類の学習可能なパラメータを持つ
- 活性化関数：中間層はReLU, tanh、出力層はSigmoid, Softmaxなどを適用することが多い

### ドロップアウト層(layers.Dropout)

- 過学習回避のために正則化を行う層で、非常に実用性が高く、ほぼ全てのモデルに対して適用することが容易
- 機能：学習時にノードのうちのいくつかをランダムに無効にすることで過学習を緩和し、精度の向上が見込める
- いくつを無効にするかはハイパーパラメータとして定める必要がある  
- 活性化関数：適用しないことが多い

### 畳み込み層(layers.Conv2D)

- **カーネル**と呼ばれるフィルターを通して特徴量を抽出する層で、特に画像を扱うDL手法(CNN)でプーリング層とセットで用いられることが多い
- カーネルを学習可能なパラメータとして持つ
- 活性化関数：プーリング層を適用することが多い
- 第1引数には出力のチャネル数、第2引数にはカーネルの一辺のサイズを指定する

### プーリング層(layers.MaxPool2D)

- 重要な特徴量のみ抽出することでデータの圧縮を行う層、畳み込み→活性化関数→プーリングの順で利用することが多い
- 過学習緩和・ロバスト性の向上・計算量削減などの効果が見込める
- 代表的な抽出方法は「最大値プーリング・平均値プーリング」の2種類
- 活性化関数：適用しない（プーリング層は活性化関数の作用をする）

### 長短期記憶層(layers.LSTM)

- 自己ループを持つことで内部の状態を持続させることができる層、特に言語を扱うDL手法(RNN)で利用されている
- **ゲート**と呼ばれる構造により情報の流れを制御し、勾配消失問題を解消することができる
- 活性化関数：Sigmoid, tanhなど適用することが多い

```python
from tensorflow.keras import layers

conv = layers.Conv2D(5,3)# 出力チャネル数=5, カーネルのサイズ=3*3でクラスを宣言

x = tf.random.normal((1, 28, 28, 3))# 1枚の画像が3次元(=入力チャネル)で28*28のサイズ
x = conv(x) # 畳み込みの実行

print(x.shape)
```

## モデル

### VGG16

```python
from tensorflow.keras import applications

vgg_model = applications.vgg16.VGG16()

vgg_model.summary()
```

### ResNet50

```python
resnet_model = applications.resnet50.ResNet50()

resnet_model.summary()