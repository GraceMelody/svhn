from scipy.io import loadmat
import cv2
import numpy as np
trainMat = loadmat("train_32x32.mat")
#print trainMat["X"][0]

# Unsharp Mask
def unsharp(image, strength):

    # Calculate the Laplacian
    lap = cv2.Laplacian(image,cv2.CV_64F)

    # Calculate the sharpened image
    sharp = image-strength*lap

    # Saturate the pixels in either direction
    sharp[sharp>255] = 255
    sharp[sharp<0] = 0
    
    sharp = np.uint8(sharp)
    return sharp

# Convert to Channel Last
trainMat["X"] = trainMat["X"].transpose((3,0,1,2))

image_count = len(trainMat["X"])

newTrainMat = {"X":[], "Y": []}

# Convert to B/W
for i in range(image_count):
    newTrainMat["X"].append(cv2.cvtColor(trainMat["X"][i], cv2.COLOR_BGR2GRAY))
    num = trainMat["y"][i]
    if num == 10:
        num = 0
    newTrainMat["Y"].append(num)


for i in range(image_count):

    # Equalize Hist
    newTrainMat["X"][i] = cv2.equalizeHist(newTrainMat["X"][i])

    # Otsu Threshold
    # ret2,newTrainMat["X"][i] = cv2.threshold(newTrainMat["X"][i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    newTrainMat["X"][i] = unsharp(newTrainMat["X"][i], 0.1)



idx = 2
print(newTrainMat["Y"][idx])
cv2.imshow("original", trainMat["X"][idx])

cv2.imshow("processed", newTrainMat["X"][idx])
cv2.waitKey(0)