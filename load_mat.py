from scipy.io import loadmat
import cv2
import numpy as np
import h5py
trainMat = loadmat("train_32x32.mat")
testMat = loadmat("test_32x32.mat")
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
    
    sharp = np.float32(sharp)
    return sharp

# Convert to Channel Last
trainMat["X"] = trainMat["X"].transpose((3,0,1,2))

image_count = len(trainMat["X"])

val_size = image_count // 10

newTrainMat = {"X":[], "y": []}

# Convert to B/W
for i in range(image_count):
    newTrainMat["X"].append(cv2.cvtColor(trainMat["X"][i], cv2.COLOR_BGR2GRAY))
    num = trainMat["y"][i]
    if num == 10:
        num = 0
    newTrainMat["y"].append(num)
newTrainMat["y"] = np.array(newTrainMat["y"], dtype=np.float32)
newTestMat = {"X":[], "y": []}
newTestMat["X"] = np.array(testMat["X"])
newTestMat["y"] = np.array(testMat["y"])
for i in range(image_count):

    # Equalize Hist
    newTrainMat["X"][i] = cv2.equalizeHist(newTrainMat["X"][i])

    # Otsu Threshold
    # ret2,newTrainMat["X"][i] = cv2.threshold(newTrainMat["X"][i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    newTrainMat["X"][i] = unsharp(newTrainMat["X"][i], 0.1)

# Saves to .h5py
f = h5py.File('SVHN_grey.h5', 'w')


#newTrainMat["X"] = np.array(newTrainMat["X"])
#f.create_dataset('X_train', newTrainMat["X"].shape, data=newTrainMat["X"])
f.create_dataset('X_train', data=newTrainMat["X"])
f.create_dataset('y_train', data=newTrainMat["y"])
f.create_dataset('X_val', data=newTrainMat["X"][:val_size])
f.create_dataset('y_val', data=newTrainMat["y"][:val_size])
f.create_dataset('X_test', data=newTestMat["X"])
f.create_dataset('y_test', data=newTestMat["y"])
f.close()
idx = 2
print(newTrainMat["y"][idx])
cv2.imshow("original", trainMat["X"][idx])

cv2.imshow("processed", newTrainMat["X"][idx])
cv2.waitKey(0)
