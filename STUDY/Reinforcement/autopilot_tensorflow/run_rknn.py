#create RKNN object
from rknn.api import RKNN
import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call

#RKNN object 생성
rknn = RKNN(verbose=True)

#RKNN Model 로드
rknn.load_rknn('/home/yoona/rknn-toolkit/packages/final.rknn') #만들어진 rknn을 로드
print('---------------------------------------> load success') #성공 메세지 출력

#runtime 환경 init
print('---------------------------------------> Init runtime environment')
ret = rknn.init_runtime()
#오류 메세지 출력
if ret != 0:
        print('---------------------------------> Init runtime environment failed')

i = 0

while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB") #이미지 불러오기
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0      #이미지 크기 조정
    image = image.astype('float32')    #이미지 dtype을 float64  ----> float32

    degrees = rknn.inference(inputs= image)     #sess.run은 텐서플로우에서 돌리는 함수고 rknn에서는 rknn.inference로 돌려야 한다

    #degrees가 list형식으로 출력되기 때문에 arctan같은 연산을 위해서는 float으로 형식 바꿔줘야 한다
    
    for j in degrees:
        d = float(j)
        
    #각도 출력을 위해 아크탄젠트 연산
    d = d * 180 / scipy.pi


    call("clear")
    print("Predicted steering angle: " + str(d) + " degrees")

    i += 1
