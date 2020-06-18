# v4l2_stream_exam

## 필요파일
Emotion_Recognition 

emotion.rknn

emotion.rknn

rex보드로 이동 후 ./Emotion_Recognition  실행


## build 방법

cd build;

cmake ..

make

## Cam , Display 설정 파일 
include/config.h


## 보드내에 라이브러리가 없는경우 
lib 폴더 하위에 있는 해당 lib파일을 보드에 /lib 경로로 복사할것

## 추가설명

RgaSURF_FORMAT에 그레이스케일 포멧을 지원하지 않아 RK_FORMAT_RGB_888으로 사용했습니다.

