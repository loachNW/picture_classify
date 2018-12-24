# coding=gbk
# import scipy
import os
import pickle
import random

import numpy as np
from PIL import Image


# from PIL import Image
# os.chdir('C:/Users/ASUS/Desktop/')
# im = Image.open("890.jpg")
# box = (10,10,100,100)
# region = im.crop(box)
# region.save("cutting.jpg")






def ImageToMatrix(filename):
    # ��ȡͼƬ
    im = Image.open(filename)
    # ��ʾͼƬ
    #     im.show()
    width, height = im.size
    if width>height: #ͼƬ�и�
        box = ((width-height)/2,0,width - (width-height)/2,height)
        im = im.crop(box)
        width, height = im.size
    else:
        box = (0,(height - width) / 2, width, height - (height - width) / 2)
        im = im.crop(box)
        # plt.show()  # ��Ҫ����show()��������Ȼͼ��ֻ�����ڴ��ж�����ʾ����
        width, height = im.size
    im = im.resize((120,120))#ͼƬ����
    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype='float') / 255.0  #�ַ���ת��Ϊmatrix,���ұ�׼����ת��Ϊ0��1֮��ľ���
    new_data = np.reshape(data, (width, height))
    return new_data


#     new_im = Image.fromarray(new_data)
#     # ��ʾͼƬ
#     new_im.show()
def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


# �˺�����ȡ�ض��ļ����µ�jpg��ʽͼ���ַ��Ϣ���洢���б���
def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]







# r""�Ƿ�ֹ�ַ���ת��
# c=['zheng_chang\\1.jpg', 'zheng_chang\\2.jpg', 'zheng_chang\\3.jpg']  ��list��ʽ���jpg��ʽ������ͼ�񣨴�·����




def main():
    rightNo = 0
    # os.chdir('C:/Users/ASUS/Desktop/train1/')
    c = get_imlist('C:/Users/ASUS/Desktop/train1/')
    d = len(c)
    id = list(range(d))# ͼ�����
    random.shuffle(id)
    imag = list()
    label = []
    for i in id:
        filename = c[i]
        data = ImageToMatrix(filename)
        imag.append(np.array(data).reshape(1,14400).tolist()[0])
        if int(c[i].split("/")[-1][:-4]) < 449 :
            label.append([0,1])
        else:
            label.append([1,0])
    imag = np.array(imag)
    lable = np.array(label)

    text_x = imag[:100]
    text_y = lable[:100]
    train_x = imag[100:]
    train_y = lable[100:]
        #��Ϊpakle�ļ�
    with open("cat&dog.pkl","wb") as f:
        pickle.dump(train_x,f)
        pickle.dump(train_y, f)
        pickle.dump(text_x, f)
        pickle.dump(text_y, f)
        # with open("cat&dog.pkl","rb") as f:
        #     data = pickle.load(data)
        rightNo += 1
        print(rightNo)
main()





# print(data)
# new_im = MatrixToImage(data)
# plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
# new_im.show()
# new_im.save('lena_1.bmp')