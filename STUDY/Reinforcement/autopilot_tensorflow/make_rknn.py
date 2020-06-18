from rknn.api import RKNN
#create RKNN object
rknn = RKNN(verbose=True)
print('--> Loading model')
#freeze된 pb파일, input 노드, output노드, input 사이즈 리스트 입력
#노드 확인은 tensorboard, netron을 활용
ret = rknn.load_tensorflow(tf_pb='/home/yoona/Autopilot-TensorFlow/freeze.pb',
                                             inputs=['Placeholder'],
                                             outputs=['add_9'],
                                             input_size_list=[[66,200,3]])

#오류 발생시 오류 메시지 출력
if ret !=0:
        print('Load failed!')
        exit(ret)
#Load 완료시 'done'출력
print('done')



print('--> Building model') #빌드하는 중임을 알려주는 출력
ret = rknn.build(do_quantization=False)
#오류 발생시 오류 메시지 출력
if ret !=0:
        print('Build failed!')
        exit(ret)
#Build 완료시 'done'출력
print('done')



print('--> Export RKNN model')
ret = rknn.export_rknn('./final.rknn') #다음과 같은 이름으로 저장됩니다.
#오류 발생시 오류 메시지 출력
if ret != 0:
        print('Export failed!')
        exit(ret)
#export 완료시 'done'출력
print('done')
