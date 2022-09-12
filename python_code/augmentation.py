import numpy as np

def augmentation (data):
    x1 = np.fliplr(data)
    x2 = np.rot90(data,2, (1,2))
    x3 = np.rot90(x1 ,2, (1,2))
    return np.vstack((data,x1,x2,x3))

def mirror (x):
    xmirror = np.fliplr(x)
    return xmirror

def rot180 (x):
    xrot180 = np.rot90(x,2, (1,2))
    return xrot180

def mirror_rot180 (x):
    xrot_mirror = np.rot90(np.fliplr(x) ,2, (1,2))
    return xrot_mirror