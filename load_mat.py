from scipy.io import loadmat
import cv2
import numpy as np
import h5py
trainMat = loadmat("train_32x32.mat")
testMat = loadmat("test_32x32.mat")
#print trainMat["X"]

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
testMat["X"] = testMat["X"].transpose((3,0,1,2))
image_count = len(trainMat["X"])

val_size = image_count // 10

newTrainMat = {"X":[], "y": []}
newTestMat = {"X":[], "y": []}

# Training Data
for i in range(len(trainMat["X"])):
    # Convert to B/W
    newTrainMat["X"].append(np.array([cv2.cvtColor(trainMat["X"][i], cv2.COLOR_BGR2GRAY)]).transpose((1,2,0)))

    num = trainMat["y"][i]
    if num == 10:
        num = 0
    newTrainMat["y"].append(num)
    newTrainMat["X"][i] = newTrainMat["X"][i].transpose(2,0,1)

    # Equalize Hist
    newTrainMat["X"][i][0] = cv2.equalizeHist(newTrainMat["X"][i][0])

    # Unsharp Mask
    newTrainMat["X"][i][0] = unsharp(newTrainMat["X"][i][0], 0.1)

    newTrainMat["X"][i] = newTrainMat["X"][i].transpose(1,2,0)
    newTrainMat["X"][i] = np.array(newTrainMat["X"][i], dtype=np.float32)
    newTrainMat["X"][i] /= 255

newTrainMat["X"] = np.array(newTrainMat["X"])

# Test data
for i in range(len(testMat["X"])):
    # Convert to B/W
    newTestMat["X"].append(np.array([cv2.cvtColor(testMat["X"][i], cv2.COLOR_BGR2GRAY)]).transpose((1,2,0)))

    num = testMat["y"][i]
    if num == 10:
        num= 0
    newTestMat["y"].append(num)
    newTestMat["X"][i] = newTestMat["X"][i].transpose(2,0,1)

    # Equalize Hist
    newTestMat["X"][i][0] = cv2.equalizeHist(newTestMat["X"][i][0])

    # Unsharp Mask
    newTestMat["X"][i][0] = unsharp(newTestMat["X"][i][0], 0.1)

    newTestMat["X"][i] = newTestMat["X"][i].transpose(1,2,0)
    newTestMat["X"][i] = np.array(newTestMat["X"][i], dtype=np.float32)
    newTestMat["X"][i] /= 255

newTestMat["X"] = np.array(newTestMat["X"])

newTrainMat["y"] = np.array(newTrainMat["y"], dtype=np.uint8)
newTrainMat["y"].flatten()
y_train_hot = np.zeros((len(newTrainMat["y"]), 10), dtype=np.float32)
y_train_hot[np.arange(len(newTrainMat["y"])), newTrainMat["y"]] = 1.0
newTrainMat["y"] = np.array(y_train_hot, dtype=np.float32)

newTestMat["y"] = np.array(newTestMat["y"], dtype=np.uint8)
newTestMat["y"].flatten()
y_test_hot = np.zeros((len(newTestMat["y"]), 10), dtype=np.float32)
y_test_hot[np.arange(len(newTestMat["y"])), newTestMat["y"]] = 1.0
newTestMat["y"] = np.array(y_test_hot, dtype=np.float32)

print(newTrainMat["y"].shape)
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

print(newTrainMat["X"].shape)
idx = 2
print(newTrainMat["y"][idx])
cv2.imshow("original", trainMat["X"][idx])

cv2.imshow("processed", newTrainMat["X"][idx])
cv2.waitKey(0)
