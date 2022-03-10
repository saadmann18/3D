import numpy as np
import cv2
from matplotlib import pyplot as plt

left_img = cv2.imread('images/tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('images/tsukuba_r.png', cv2.IMREAD_GRAYSCALE)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

depth = stereo.compute(left_img, right_img)

cv2.imshow("Left", left_img)
cv2.imshow("Right", right_img)

plt.imshow(depth, 'gray')
plt.axis('off')
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
