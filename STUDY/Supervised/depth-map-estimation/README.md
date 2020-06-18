Rock chip이 업데이트 된다면, 사용가능한 예제 입니다.<br>
(현재 지원되지 않는 함수 때문에 rknn 변환 실패)
-
+ Single Image Depth Map
+ Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries
+ 논문 : Junjie Hu, Mete Ozay, Yan Zhang, Takayuki Okatani https://arxiv.org/abs/1803.08673

결과
-
![](https://github.com/junjH/Revisiting_Single_Depth_Estimation/raw/master/examples/example.png)
![](https://github.com/junjH/Revisiting_Single_Depth_Estimation/raw/master/examples/results.png)


필요한 버전
-
+ python 2.7<br>
+ Pytorch 0.3.1<br>

실행
-
훈련된 모델 가져오기:
[Depth estimation networks](https://drive.google.com/file/d/1QaUkdOiGpMuzMeWCGbey0sT0wXY0xtsj/view?usp=sharing) <br>
데이터 다운로드:
[NYU-v2 dataset](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing) <br>

### Image Demo<br>
     python demo.py 

### Video Demo<br>
     python video.py 

### Test<br>
     python test.py

### Train<br>
     python train.py

Citation
-
If you use the code or the pre-processed data, please cite:

    @inproceedings{Hu2018RevisitingSI,
      title={Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps With Accurate Object Boundaries},
      author={Junjie Hu and Mete Ozay and Yan Zhang and Takayuki Okatani},
      booktitle={IEEE Winter Conf. on Applications of Computer Vision (WACV)},
      year={2019}
    }
