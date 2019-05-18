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


def preprocess(mat):
    # Convert to Channel Last
    mat["X"] = mat["X"].transpose((3,0,1,2))
    newMat = {"X": [], "y": []}

    # Training Data
    for i in range(len(mat["X"])):
        # Convert to B/W
        newMat["X"].append(np.array([cv2.cvtColor(mat["X"][i], cv2.COLOR_BGR2GRAY)]).transpose((1,2,0)))

        num = mat["y"][i]
        if num == 10:
            num = 0
        newMat["y"].append(num)
        newMat["X"][i] = newMat["X"][i].transpose(2,0,1)

        # Equalize Hist
        newMat["X"][i][0] = cv2.equalizeHist(newMat["X"][i][0])

        # Unsharp Mask
        newMat["X"][i][0] = unsharp(newMat["X"][i][0], 0.1)

        newMat["X"][i] = newMat["X"][i].transpose(1,2,0)
        newMat["X"][i] = np.array(newMat["X"][i], dtype=np.float32)
        newMat["X"][i] /= 255

    newMat["X"] = np.array(newMat["X"])
    newMat["y"] = np.array(newMat["y"], dtype=np.uint8)
    newMat["y"].flatten()
    y_hot = np.zeros((len(newMat["y"]), 10), dtype=np.float32)
    y_hot[np.arange(len(newMat["y"])), newMat["y"]] = 1.0
    newMat["y"] = np.array(y_hot, dtype=np.float32)
    return newMat

newTrainMat = preprocess(trainMat)
newTestMat = preprocess(testMat)

val_size = len(newTrainMat) // 10


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
