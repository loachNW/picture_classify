import os
import pickle
import random

import numpy as np
from PIL import Image


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


c = get_imlist('C:/Users/ASUS/Desktop/train1/')
d = len(c)
id = list(range(d))  # 图像个数
# random.shuffle(id)
imag = list()
label = []
x = 0
for i in id:
    filename = c[i]
    im = Image.open(filename)
    # 显示图片
    #     im.show()
    width, height = im.size
    if width>height: #图片切割
        box = ((width-height)/2,0,width - (width-height)/2,height)
        im = im.crop(box)
        width, height = im.size
    else:
        box = (0,(height - width) / 2, width, height - (height - width) / 2)
        im = im.crop(box)
        # plt.show()  # 需要调用show()方法，不然图像只会在内存中而不显示出来
        width, height = im.size
    im = im.resize((120,120))#图片缩放
    # width, height = im.size
    im = im.convert("L")
    # im.show()
    im.save('C:/Users/ASUS/Desktop/train/'+'%d.jpg' % x)
    x += 1
pass



