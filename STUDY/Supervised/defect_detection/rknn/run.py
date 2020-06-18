from rknn.api import RKNN
import cv2
import numpy as np

img = cv2.imread('Part3.jpg', 0)
img = cv2.resize(img, (1280, 512))

rknn = RKNN(verbose=True)

rknn.load_rknn('./segmentation.rknn')
print('--> load success')

print('--> Init runtime environment')
ret = rknn.init_runtime()

if ret != 0:
  print('Init runtime environment failed')
  exit(ret)
print('--> Running model')

outputs = rknn.inference(inputs=[img])
print('done')

a = np.array(outputs)
input = np.squeeze(a,0)
input = np.squeeze(input,0)

input.reshape(160, 64, 1)
cv2.normalize(input, input, 0, 255, cv2.NORM_MINMAX)
img = cv2.resize(input, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

#print(img)
#print(img.shape)
cv2.imwrite('seg_output_Part3.jpg', img)
print('save image')

min_pixel = np.max(img)

if min_pixel == 255:
  print('Defect가 있습니다.')
else :
  print('Defect가 없습니다.')
