import cv2
import numpy as np

noise = np.random.normal(0, 1, (224,224,3))
noise=np.clip(noise/2+0.5,0,1)
noise=noise*255
noise=np.uint8(noise)
cv2.imwrite('./noise.jpg',noise)
