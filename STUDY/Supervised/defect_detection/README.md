# 표면 결함 탐지 (Surface Defect Detection)
표면의 defect를 탐지하여, segmentation 결과와 defect 유무를 출력합니다.<br>
논문: https://arxiv.org/pdf/1903.08536.pdf<br>

Test Environment
-
- Python 3.5.2
- Tensorflow-gpu 1.14.0
- CUDA 10.1

파일 설명
-
- KolektorSDD : 데이터세트
- original_code : PC에서 실행하는 코드, 모델
- rknn : rknn으로 실행하는 코드, 모델

**rknn파일로 들어간 후 아래 코드를 실행해 주세요.**

<br>

RKNN모델 변환
-
    python convert.py


코드 실행
-
    python run.py

결과
-
<img src="/rknn/output/Part3.jpg" width="20%" height="20%">
<img src="/rknn/output/seg_output_Part3.jpg" width="20%" height="20%">
<br>
<img src="/rknn/output/defect_message.jpg" width="40%" height="40%">

인용
-
https://github.com/Wslsdx/Deep-Learning-Approach-for-Surface-Defect-Detection
