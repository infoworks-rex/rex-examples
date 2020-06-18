## 분야 : Low-light image enhancement 

## 논문 : Kindling the Darkness: A Practical Low-light Image Enhancer(2019)  
> 출처 : https://github.com/zhangyhuaee/KinD


tensorflow 버전이 1.x 에서 2.x 으로 업그레이드 됨에 따라 사용하는 함수가 변경되어 논문의 소스코드 일부를 수정했습니다. 



### * 요구환경 

1. Python
2. Tensorflow >= 1.10.0
3. numpy, PIL



##### 저의 경우, 다음과 같은 환경에서 실행하였습니다.

1. Python 2.7.17

2. Tensorflow 1.14.0

3. numpy 1.16.6, PIL 6.2.2

4. CuDA 10.2, CuDNN 7.6.4

5. (utils.py를 제외한) 모든 *.py파일의 코드를 수정했습니다 (tensorflow의 버전호환 문제때문에 tf.(function-name)을 tf.compat.v1.(function-name)으로 변경)

6. GPU 환경에서 실행 (회사 서버로 원격 접속하여 실행, 터미널에서는 이미지 확인 불가)

   → 결과물 확인은 WinSCP를 이용해 local로 옮겨 확인하였음

7. *.py 파일 실행 전에 GPU 상태를 봐서 어느것을 사용할 것인지 지정해줘야합니다. 

   1번 GPU를 사용하는경우 다음의 명령어를 입력한 뒤 *.py파일을 실행하면 됩니다. 

   > $ export CUDA_VISIBLE_DEVICES=1



### * 실행 순서 


#### Train

> $ python decomposition_net_train.py   
>
> $ python adjustment_net_train.py   
>
> $ python reflectance_restoration_net_train.py   
>
> $ python evaluate_LOLdataset.py 


#### Test

> $ python evaluate.py


### * 결과 확인

1. KinD디렉토리 자체를 WinSCP를 통해 local에 옮긴다. 

2. KinD파일 하위에 있는 results\LOLdataset_eval15로 가면 15장의 최종 네트워크 output 이미지를 확인할 수 있다. 

+ 참고 : LOLdataset.zip파일은 원본 링크의 README.md파일에서 다운로드 받아야한다. (로컬에서 다운받은 후 WinSCP로 서버에 옮겼음)