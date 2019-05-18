from scipy.io import savemat
import cv2
import numpy as np

def img2mat(image):
    x = []
    y = []
    img = cv2.imread(image)
    x.append(img)
    x_numpy = np.array(x,dtype = np.uint8)
    x_reshape = np.reshape(x_numpy,[x_numpy.shape[1],x_numpy.shape[2],x_numpy.shape[3],x_numpy.shape[0]])
    
    y.append([0])
    y_numpy = np.array(y,dtype =np.uint8)
    savemat('predict.mat',{'X':x_reshape,'y':y})
    print(y_numpy.shape)