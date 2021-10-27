import cv2
import numpy as np
import glob

img_dir = "dataset/Pepaya/"
ext = ['jpg', 'png']
files1 = []
[files1.extend(glob.glob(img_dir + '*.' +e))for e in ext]
images1 = [cv2.imread(file1) for file1 in files1]
i = 1
for img1 in images1:
    img_adjusted1 = cv2.addWeighted(img1, 1.5, np.zeros(img1.shape, img1.dtype), 0, -12)
    img_name1 = "dataset/img-proces-pepaya/" + str(i) + ".jpg"
    cv2.imwrite(img_name1, img_adjusted1)
    i+=1

        





