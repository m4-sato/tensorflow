# TensorFlowメモ

[TensorFlow公式ドキュメント](https://www.tensorflow.org/)

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

### ReLU関数

```python
x = tf.constant([-1.0, -0.5,0.5, 1.0])
x = tf.nn.relu(x)
print(x)
```

- 他の実装方法
  - 関数ではなく層としての実装：```layers.ReLU()```
  - 全結合層とセットで実装　　：```layers.Dense(units=units, activation="relu")```

## 損失関数

### 平均二乗誤差(MSE)

```python
criterion = losses.MeanSquaredError()

x = tf.constant([0, 1, 2])
y = tf.constant([1, -1, 0])

loss = criterion(x, y)
print(loss)
```

### 交差エントロピー誤差

## 最適化関数

TensorFlowでは```tf.keras.optimizers```モジュールに最適化関数が用意されている。

```python
optimizer = optimizers.Adam(learning_rate=0.1)
print(optimizer)
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