import numpy as np
import cv2 as cv
import scipy.misc as misc
classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']

# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 建立索引

def image2label(im):# 将3通道的label彩图变成单通道的图，图上每个像素点的值代表属于的class
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64') # 根据索引得到 label 矩阵
def cropImg(img,size):
    pass

label_im = cv.imread('/Users/huangxiao/imgData/ADEChallengeData2016/annotations/training/ADE_train_00000002.png')
print(np.shape(label_im))
print(np.shape(misc.imread('/Users/huangxiao/imgData/ADEChallengeData2016/images/training/ADE_train_00000002.jpg')))
# label = image2label(label_im)
# print(label[150:160, 240:250])