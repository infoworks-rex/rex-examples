LSTM을 이용한 간단한 내일 주식 예측   
====================================

# 1. 실행 가능 환경 정보

* Python: 3.7.3
* numpy: 1.16.2
* pandas: 0.24.2
* tensorflow :1.13.1
* matplotlib: 3.0.3


***

# 2. rknn 변환

* RNN은 구식의 모델이므로 LSTM 폴더로 이동하는걸 추천합니다
* rknn 변환이 가능한 파일은 RNN-train.py입니다.
* 기존 LSTM 모델이 지원이 안되는 관계로 static RNN, Tanh 활성화 함수를 사용합니다.

```
    python RNN-train.py
```