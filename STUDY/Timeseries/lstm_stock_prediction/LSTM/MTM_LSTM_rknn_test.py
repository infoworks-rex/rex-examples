#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os.path
import csv
tf.reset_default_graph()


# In[2]:


# 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
# Min-Max scaling
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원

# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


# In[3]:


# 하이퍼파라미터
input_data_column_cnt = 6  # 입력데이터의 컬럼 개수(Variable 개수)
output_data_column_cnt = 6 # 결과데이터의 컬럼 개수

seq_length = 28            # 1개 시퀀스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 120   # 각 셀의 (hidden)출력 크기
forget_bias = 1.0          # 망각편향(기본값 1.0)
num_stacked_layers = 7     # stacked LSTM layers 개수
keep_prob = 1.0            # dropout할 때 keep할 비율

epoch_num = 1000           # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
learning_rate = 0.01       # 학습률

predict_days = 30          # 미래의 몇일을 예측할것인지 설정


# In[4]:


# 데이터를 로딩한다.
stock_file_name = 'AAPL_5Y.csv' # 주가데이터 파일
encoding = 'euc-kr' # 문자 인코딩
names = ['Date','Open','High','Low','Close','Adj Close','Volume']
raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding) #판다스이용 csv파일 로딩
raw_dataframe.info() # 데이터 정보 출력

today = raw_dataframe.values[-1, 0]
print("today = ", today)

# raw_dataframe.drop('Date', axis=1, inplace=True) # 시간열을 제거하고 dataframe 재생성하지 않기
del raw_dataframe['Date'] # 위 줄과 같은 효과

stock_info = raw_dataframe.values[1:].astype(np.float) # 금액&거래량 문자열을 부동소수점형으로 변환한다
print("stock_info.shape: ", stock_info.shape)
print("stock_info[0]: ", stock_info[0])


# In[5]:


# 데이터 전처리
# 가격과 거래량 수치의 차이가 많아나서 각각 별도로 정규화한다

# 가격형태 데이터들을 정규화한다
# ['Open','High','Low','Close','Adj Close','Volume']에서 'Adj Close'까지 취함
# 곧, 마지막 열 Volume를 제외한 모든 열
price = stock_info[:,:-1]
    
norm_price = min_max_scaling(price) # 가격형태 데이터 정규화 처리
print("price.shape: ", price.shape)
print("price[0]: ", price[0])
print("norm_price[0]: ", norm_price[0])
print("="*100) # 화면상 구분용

# 거래량형태 데이터를 정규화한다
# ['Open','High','Low','Close','Adj Close','Volume']에서 마지막 'Volume'만 취함
# [:,-1]이 아닌 [:,-1:]이므로 주의하자! 스칼라가아닌 벡터값 산출해야만 쉽게 병합 가능
volume = stock_info[:,-1:]
norm_volume = min_max_scaling(volume) # 거래량형태 데이터 정규화 처리
print("volume.shape: ", volume.shape)
print("volume[0]: ", volume[0])
print("norm_volume[0]: ", norm_volume[0])
print("="*100) # 화면상 구분용

# 행은 그대로 두고 열을 우측에 붙여 합친다
x = np.concatenate((norm_price, norm_volume), axis=1) # axis=1, 세로로 합친다
print("x.shape: ", x.shape)
print("x[0]: ", x[0])    # x의 첫 값
print("x[-1]: ", x[-1])  # x의 마지막 값
print("="*100) # 화면상 구분용

y = x # 타켓은 주식이다
print("y[0]: ",y[0])     # y의 첫 값
print("y[-1]: ",y[-1])   # y의 마지막 값


# In[6]:


# 텐서플로우 플레이스홀더 생성
# 입력 X, 출력 Y를 생성한다
X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
print("X: ", X)
Y = tf.placeholder(tf.float32, [None, output_data_column_cnt])
print("Y: ", Y)

predictions = tf.placeholder(tf.float32, [None, output_data_column_cnt])
print("predictions: ", predictions)


# In[7]:


# 모델(LSTM 네트워크) 생성
def lstm_cell(name):
    # LSTM셀을 생성
    # num_units: 각 Cell 출력 크기
    # forget_bias:  to the biases of the forget gate 
    #              (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
    # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
    # state_is_tuple: False ==> they are concatenated along the column axis.
    cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_cell_hidden_dim, 
                                             forget_bias=forget_bias, activation=tf.nn.tanh, name=name)
    
    #cell = tf.keras.layers.LSTMCell(units=rnn_cell_hidden_dim, 
    #                                         unit_forget_bias=forget_bias, activation=tf.nn.tanh)
    
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell

# num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
#stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
#multi_cells = tf.nn.rnn_cell.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()
#multi_cells = keras.layers.RNN(stackedRNNs)

#multi_cells = tf.keras.layers.StackedRNNCells(stackedRNNs)

#print(multi_cells)

# RNN Cell(여기서는 LSTM셀임)들을 연결
hypothesis, _states = tf.nn.dynamic_rnn(lstm_cell("LSTM"), X, dtype=tf.float32)
hypothesis, _states = tf.nn.dynamic_rnn(lstm_cell("LSTM_1"), hypothesis, dtype=tf.float32)
hypothesis, _states = tf.nn.dynamic_rnn(lstm_cell("LSTM_2"), hypothesis, dtype=tf.float32)
hypothesis, _states = tf.nn.dynamic_rnn(lstm_cell("LSTM_3"), hypothesis, dtype=tf.float32)
hypothesis, _states = tf.nn.dynamic_rnn(lstm_cell("LSTM_4"), hypothesis, dtype=tf.float32)
hypothesis, _states = tf.nn.dynamic_rnn(lstm_cell("LSTM_5"), hypothesis, dtype=tf.float32)
hypothesis, _states = tf.nn.dynamic_rnn(lstm_cell("LSTM_6"), hypothesis, dtype=tf.float32)

#print("hypothesis: ", hypothesis)
#hypothesis, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(stackedRNNs, stackedRNNs, X, dtype=tf.float32)


# [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
# 과거 여러 거래일의 주가를 이용해서 다음날의 주가 6개를 예측하기때문에 MANY-TO-MANY형태이다
hypothesis = tf.contrib.layers.fully_connected(hypothesis[:,-1], output_data_column_cnt, activation_fn=tf.identity)
#hypothesis = tf.keras.layers.Dense(output_data_column_cnt, activation_fn=tf.identity)
print("hypothesis: ", hypothesis)


# In[8]:


# The file path to save the data
save_file = './model.ckpt'

saver = tf.train.Saver()
sess = tf.Session()

if os.path.isfile(save_file+".meta"):
    saver.restore(sess,save_file)
    print("저장된 모델을 불러옵니다")
else:
    sess.run(tf.global_variables_initializer())
    print("존재하는 모델이 없습니다")


# In[9]:


# sequence length만큼의 가장 최근 데이터를 슬라이싱한다
recent_data = np.array([x[len(x)-seq_length : ]])
print("recent_data.shape:", recent_data.shape)
print("recent_data:", recent_data)

time = datetime.datetime.strptime(today, '%Y-%m-%d')


for i in range(predict_days):
    
    # 내일 주가를 예측해본다
    test_predict = sess.run(hypothesis, feed_dict={X: recent_data})

    print("test_predict shape = ", test_predict.shape)
    print("test_predict = ", test_predict)

    test_predict1 = reverse_min_max_scaling(price,test_predict[0, 0:5]) # 금액데이터 역정규화한다
    test_predict2 = reverse_min_max_scaling(volume,test_predict[0, 5]) # 볼륨데이터 역정규화한다
    real_test_predict = np.append(test_predict1, test_predict2)
    #real_test_predict = real_test_predict.astype(np.string_)

    print(test_predict1, test_predict2)
    print(test_predict1.shape, test_predict2.shape)
    

    time = time + datetime.timedelta(days=1)
    str_time = time.strftime("%Y-%m-%d")
    
    print(str_time, "'s stock price", real_test_predict) # 예측한 주가를 출력한다
    
    
    #csv 파일 열고 추가
    f = open("predict.csv", 'a', newline='')
    wr = csv.writer(f)
    wr.writerow([str_time, real_test_predict[0], real_test_predict[1], real_test_predict[2], real_test_predict[3]
            , real_test_predict[4], real_test_predict[5]])
    f.close()
    
    
    #후처리
    recent_data = np.squeeze(recent_data, axis=0)  # [1,28,6] -> [28,6]
    recent_data = np.concatenate((recent_data, test_predict), axis=0)  # 새로 얻은 값 추가
    recent_data = np.delete(recent_data, [0], axis=0)  # 제일 처음값 삭제
    recent_data = np.expand_dims(recent_data, axis=0)  # [28,6] -> [1,28,6]


# In[10]:


# 하이퍼파라미터 출력
print('input_data_column_cnt:', input_data_column_cnt, end='')
print(',output_data_column_cnt:', output_data_column_cnt, end='')

print(',seq_length:', seq_length, end='')
print(',rnn_cell_hidden_dim:', rnn_cell_hidden_dim, end='')
print(',forget_bias:', forget_bias, end='')
print(',num_stacked_layers:', num_stacked_layers, end='')
print(',keep_prob:', keep_prob, end='')

print(',epoch_num:', epoch_num, end='')
print(',learning_rate:', learning_rate, end='')

# 결과 그래프 출력

recent_data = np.squeeze(recent_data, axis=0)
print(recent_data.shape)

recent_data1 = reverse_min_max_scaling(price,recent_data[:, 0:5])
recent_data2 = reverse_min_max_scaling(volume,recent_data[:, 5])

# 실제 데이터를 로딩한다.
stock_file_name = 'AAPL.csv' # 주가데이터 파일
encoding = 'euc-kr' # 문자 인코딩
names = ['Date','Open','High','Low','Close','Adj Close','Volume']
raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding) #판다스이용 csv파일 로딩
raw_dataframe.info() # 데이터 정보 출력

# raw_dataframe.drop('Date', axis=1, inplace=True) # 시간열을 제거하고 dataframe 재생성하지 않기
del raw_dataframe['Date'] # 위 줄과 같은 효과

stock_info = raw_dataframe.values[1:].astype(np.float) # 금액&거래량 문자열을 부동소수점형으로 변환한다

# sequence length만큼의 가장 최근 데이터를 슬라이싱한다
recent_real = np.array([stock_info[len(stock_info)-predict_days : ]])
recent_real = np.squeeze(recent_real, axis=0)
print("recent_real = ", recent_real)
print(recent_real.shape)
#print("stock_info = ", stock_info)
#recent_real = stock_info


# In[11]:


##########################################################################

# 예측 데이터를 로딩한다.
stock_file_name = 'predict.csv' # 주가데이터 파일
encoding = 'euc-kr' # 문자 인코딩
names = ['Date','Open','High','Low','Close','Adj Close','Volume']
raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding) #판다스이용 csv파일 로딩
raw_dataframe.info() # 데이터 정보 출력

# raw_dataframe.drop('Date', axis=1, inplace=True) # 시간열을 제거하고 dataframe 재생성하지 않기
del raw_dataframe['Date'] # 위 줄과 같은 효과

stock_info = raw_dataframe.values[1:].astype(np.float) # 금액&거래량 문자열을 부동소수점형으로 변환한다

recent_data = stock_info

print("recent_real = ", recent_data)
print(recent_data.shape)


# In[12]:


plt.figure(1)
plt.plot(recent_data[:, [0]], 'b')
plt.plot(recent_real[:, [0]], 'r')
plt.xlabel('Time Period')
plt.ylabel('Open price')
plt.show()

plt.figure(2)
plt.plot(recent_data[:, [1]], 'b')
plt.plot(recent_real[:, [1]], 'r')
plt.xlabel('Time Period')
plt.ylabel('Top price')
plt.show()

plt.figure(3)
plt.plot(recent_data[:, [2]], 'b')
plt.plot(recent_real[:, [2]], 'r')
plt.xlabel('Time Period')
plt.ylabel('Low price')
plt.show()

plt.figure(4)
plt.plot(recent_data[:, [3]], 'b')
plt.plot(recent_real[:, [3]], 'r')
plt.xlabel('Time Period')
plt.ylabel('Close price')
plt.show()

plt.figure(5)
plt.plot(recent_data[:, [4]], 'b')
plt.plot(recent_real[:, [4]], 'r')
plt.xlabel('Time Period')
plt.ylabel('Adj Close price')
plt.show()

plt.figure(6)
plt.plot(recent_data[:, [5]], 'b')
plt.plot(recent_real[:, [5]], 'r')
plt.xlabel('Time Period')
plt.ylabel('Volume')
plt.show()


# In[ ]:




