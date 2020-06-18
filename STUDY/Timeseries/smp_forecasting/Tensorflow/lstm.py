import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#길이 조절
seq_length = 28
epoch_length = 500

#raw data 불러오기
raw_data = pd.read_csv('target_v2.csv', index_col=0, squeeze=True)

#numpy array
raw_data_array = raw_data.values

#normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(raw_data_array)

#속성과 클래스로 분류
X_array = data_scaled[:,:]#0:-1]
Y_array = data_scaled[:,-1]

#X, Y+1을 한 묶음으로 (미래 예측을 위해)
Y_next_array = []

for i in range(len(Y_array)-1):
  Y_next_array.append(Y_array[i+1])

#numpy로 바꿈
X_array = np.array(X_array)
Y_next_array = np.array(Y_next_array)

#28일씩 묶어 예측을 위해 데이터 전처리 수행
X_data = []
Y_data = []

for i in range(len(Y_next_array)-seq_length):
  _x = X_array[i : i+seq_length]
  _y = Y_next_array[i+seq_length]

  X_data.append(_x)
  Y_data.append(_y)

X_data = np.array(X_data)
Y_data = np.array(Y_data)

#데이터의 70%를 훈련 데이터로 30%는 테스트 데이터로 사용
train_size = int(len(Y_data)*0.7)
test_size = len(Y_data)-train_size

#훈련 데이터
X_train = np.array(X_data[0:train_size])
Y_train = np.array(Y_data[0:train_size])

#테스트 데이터
X_test = np.array(X_data[train_size:len(Y_data)])
Y_test = np.array(Y_data[train_size:len(Y_data)])

#LSTM에 들어갈 수 있는 input 크기로 만들어줌
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))

#변수 생성
global_step = tf.Variable(0, trainable=False, name='global_step')
X = tf.placeholder(tf.float32, [None, seq_length, 8])
Y = tf.placeholder(tf.float32, [None, 1])

#모델
cell = tf.nn.rnn_cell.LSTMCell(num_units=64, activation=tf.nn.tanh)
L1, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
model = tf.contrib.layers.fully_connected(L1[:, -1], 1, activation_fn=tf.identity)

cost = tf.reduce_mean(tf.square(model-Y))
optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(cost, global_step=global_step)

#신경망 모델 학습
#tf.global_variables는 앞서 정의한 변수들을 가져오는 함수
#이 변수들을 파일에 저장하거나 이전에 학습한 결과를 불려와 사용
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

#기존에 학습해둔 모델이 있으면 saver.restore함수로 학습된 값을 불러오고, 아니라면 변수를 초기화
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
  saver.restore(sess, ckpt.model_checkpoint_path)
else:
  sess.run(tf.global_variables_initializer())

#5000 epochs만큼 수행
for epoch in range(epoch_length):
  #total_cost = 0
  sess.run(train_op, feed_dict={X: X_train, Y: Y_train})

  print('Epoch:', '%d' % (epoch+1), 'Step: %d' %sess.run(global_step),
        'Loss(train): %.3f' %sess.run(cost, feed_dict={X: X_train, Y: Y_train}),
        'Loss(test): %.3f' %sess.run(cost, feed_dict={X: X_test, Y: Y_test}))

  saver.save(sess, './model/lstm.ckpt'+'%d' % (epoch+1), global_step=global_step)
  tf.train.write_graph(sess.graph, './', 'lstm_tanh.pbtxt')

output = sess.run(model, feed_dict={X:X_test})
output_train = sess.run(model, feed_dict={X:X_train})

#test 데이터 결과 출력
fig = plt.figure(1)
plt.plot(output, 'r')
plt.plot(Y_test, 'b')
plt.show()
fig.savefig('output.png', bbox_inches='tight')

#train 데이터 결과 출력
fig2 = plt.figure(2)
plt.plot(output_train, 'r')
plt.plot(Y_train, 'b')
plt.show()
fig2.savefig('output_train.png', bbox_inches='tight')
