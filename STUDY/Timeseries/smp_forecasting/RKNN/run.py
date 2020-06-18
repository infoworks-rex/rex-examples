import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from rknn.api import RKNN
from statsmodels.tsa.arima_model import ARIMA

seq_length = 28

#raw data 불러오기
raw_data = pd.read_csv('target_v2.csv', index_col=0, squeeze=True)

#numpy array
raw_data_array = raw_data.values

#normalize
max = np.max(raw_data_array[:,-1])
min = np.min(raw_data_array[:,-1])
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(raw_data_array)

#속성과 클래스로 분류
X_array = data_scaled[:,:]
Y_array = data_scaled[:,-1]

#X, Y+1을 한 묶음으로 (미래 예측을 위해)
Y_next_array = []

for i in range(len(Y_array)-1):
  Y_next_array.append(Y_array[i+1])

#numpy로 바꿈
X_array = np.array(X_array, dtype=np.float32)
Y_next_array = np.array(Y_next_array, dtype=np.float32)

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

#테스트 데이터
X_test = np.array(X_data[train_size:len(Y_data)])
Y_test = np.array(Y_data[train_size:len(Y_data)])

#LSTM에 들어갈 수 있는 input 크기로 만들어줌
Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))

#rknn 객체 생성
rknn = RKNN(verbose=True)

#rknn load
rknn.load_rknn('./lstm.rknn')
print('load success')

#rknn 런타임 실행
print('--> Init runtime environment')
ret = rknn.init_runtime()
if ret != 0:
  print('Init runtime environment failed')

#모델 실행
print('--> Running model')

LSTM_array = []

#데이터 갯수만큼 실행
for i in range(Y_test.shape[0]):
  outputs = rknn.inference(inputs=[X_test[i]])
  outputs = np.array(outputs)
  LSTM_array.append(outputs[0][0])
  print('현재: ', i, '/', Y_test.shape[0])
LSTM_array = np.array(LSTM_array)

ARIMA_array = []

#ARIMA 실행
for i in range(Y_test.shape[0]):
  rows = 838-243+i
  series = pd.read_csv('project.csv', header=0, nrows=rows, index_col=0, squeeze=True)
  model = ARIMA(series, order=(0,1,1))
  model_fit = model.fit(trend='c', full_output=True, disp=1)
  fore = model_fit.forecast(steps=1)
  ARIMA_array.append(fore[0])
ARIMA_array = np.array(ARIMA_array)

#스케일링 된 결과 값 본래 값으로 바꿈
LSTM_array = cv2.normalize(LSTM_array, LSTM_array, min, max, cv2.NORM_MINMAX)

#LSTM모델 예측값과 ARIMA 모델 예측값의 평균을 냄
hybrid = (LSTM_array + ARIMA_array)/2

#원래 값으로 scaling
Y_test = cv2.normalize(Y_test, Y_test, min, max, cv2.NORM_MINMAX)

#결과 plot 저장
fig = plt.figure()
#plt.plot(LSTM_array, 'y')
plt.plot(Y_test, 'b')
#plt.plot(ARIMA_array, 'g')
plt.plot(hybrid, 'r')
plt.show()
fig.savefig('output.png', bbox_inches='tight')
print('done')

