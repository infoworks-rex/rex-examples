LSTM을 이용한 간단한 내일 주식 예측   
====================================

# 1. 실행 가능 환경 정보

* Python: 3.7.3
* numpy: 1.16.2
* pandas: 0.24.2
* tensorflow :1.13.1
* matplotlib: 3.0.3

***




# 2. 사용법  

## 2.1 모델 트레이닝


* 먼저 주식정보를 가지고있는 csv파일을 준비해주세요
* 이곳에서 구하시면 편합니다 (<https://finance.yahoo.com/>)
```
    python LSTM_train.py
```

train 파일은 이전에 모델이 없으면 새로 생성되고 미리 훈련된 .ckpt파일이 있으면 그 파일을 불러옵니다.   


## 2.2 모델 평가


내일의 주식을 알고 싶으시면   

```
    python LSTM_test.py
```

하시면 됩니다.

***

# 3. rknn 변환

* Softsign 함수가 지원 안하는 관계로 Tanh 활성화 함수를 사용합니다.

```
    python converter.py
```

***
Many to Many 형태를 이용한 먼 미래의 주식 예측   
====================================

# 1. 구조

* LSTM Cell을 7층을 쌓아올리고 은닉층은 120개로 확대

***

# 2. 사용법

## 2.1 트레이닝

```
    python MTM_LSTM_rknn_train.py
```

## 2.2 미래 예측

* 지정한 일수 만큼 예측 가능

```
    python MTM_LSTM_rknn_test.py
```
