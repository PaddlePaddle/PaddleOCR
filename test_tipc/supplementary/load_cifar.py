import pickle as p
import numpy as np
from PIL import Image


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='bytes')
        # 以字典的形式取出数据
        X = datadict[b'data']
        Y = datadict[b'fine_labels']
        try:
            X = X.reshape(10000, 3, 32, 32)
        except:
            X = X.reshape(50000, 3, 32, 32)
        Y = np.array(Y)
        print(Y.shape)
        return X, Y


if __name__ == "__main__":
    mode = "train"
    imgX, imgY = load_CIFAR_batch(f"./cifar-100-python/{mode}")
    with open(f'./cifar-100-python/{mode}_imgs/img_label.txt', 'a+') as f:
        for i in range(imgY.shape[0]):
            f.write('img' + str(i) + ' ' + str(imgY[i]) + '\n')

    for i in range(imgX.shape[0]):
        imgs = imgX[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB", (i0, i1, i2))
        name = "img" + str(i) + ".png"
        img.save(f"./cifar-100-python/{mode}_imgs/" + name, "png")
    print("save successfully!")
