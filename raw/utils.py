#coding=utf-8
import numpy as np
from PIL import Image
import struct 
from array import array
import png
import os

# 获取指定路径下的所有 .png 文件
def file2list(path, prefix=".png"):
    # file_list = []
    # for filename in os.listdir(path):
    #     ele_path = os.path.join(path, filename)
    #     for imgname in os.listdir(ele_path):
    #         subele_path = os.path.join(ele_path, imgname)
    #         if (subele_path.endswith(".png")):
    #             file_list.append(subele_path)
    # return file_list
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(prefix)]
 
 
 
# 解析出 .png 图件文件的名称
def extract_name(imgPath):
    return imgPath.split(os.path.sep)[-1]
 

# 将X*Y图像数据转换成 1*XY 的 numpy 向量
# 参数：imgFile--图像名  如：0_1.png
def img2vector(imgFile):
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i')
    img_normlization = np.round(img_arr/255) # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normlization, (1,-1))
    return img_arr2

# 读取一个类别的所有数据并转换成矩阵 
# 参数：
#    basePath: 图像数据所在的基本路径
#       Mnist-image/train/
#       Mnist-image/test/
#    cla：类别名称
#       0,1,2,...,9
# 返回：某一类别的所有数据----[样本数量*(图像宽x图像高)] 矩阵和标签向量
def read2convert(imgFileList):
    dataLabel = [] # 存放类标签
    dataNum = len(imgFileList)
    dataMat = np.zeros((dataNum, 28*28)) # dataNum * 400 的矩阵
    for i in range(dataNum):
        imgNameStr = imgFileList[i]
        imgName = extract_name(imgNameStr)  # 得到 数字_实例编号.png
        #print("imgName: {}".format(imgName))
        classTag = imgName.split(".")[0].split("_")[0] # 得到 类标签(数字)
        #print("classTag: {}".format(classTag))
        dataLabel.append(classTag)
        dataMat[i,:] = img2vector(imgNameStr)
    return dataMat, dataLabel

# 读取训练数据
def read_folder_img():
    cName = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    train_data_path = "./mnist/train/0"
    flist = file2list(train_data_path,".png")
    dataMat, dataLabel = read2convert(flist)
    for c in cName:
        train_data_path_ = "./mnist/train/" + c
        flist_ = file2list(train_data_path_,".png")
        dataMat_, dataLabel_ = read2convert(flist_)
        dataMat = np.concatenate((dataMat, dataMat_), axis=0)
        dataLabel = np.concatenate((dataLabel, dataLabel_), axis=0)
    print(dataMat.shape)
    print(len(dataLabel))
    return dataMat, dataLabel

def binary2img(binfile, binlabel, folders):
    for (i, label) in enumerate(binlabel):
        #filename = os.path.join(folders[label], str(i) + ".png")
        filename = os.path.join(folders[label], str(label)+"_"+str(i)+".png")
        #print("writing " + filename)
        if not os.path.exists(filename):
            with open(filename, "wb") as img:
                image = png.Writer(28, 28, greyscale=True)
                data = [binfile[(i*28*28 + j*28) : (i*28*28 + (j+1)*28)] for j in range(28)]
                image.write(img, data)

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

def convert2csv():
    if not os.path.exists("./mnist/fashion/fashion_test.csv"):
        convert("./mnist/fashion/t10k-images-idx3-ubyte", "./mnist/fashion/t10k-labels-idx1-ubyte",
                "./mnist/fashion/fashion_test.csv", 10000) 
    if not os.path.exists("./mnist/fashion/fashion_train.csv"):
        convert("./mnist/fashion/train-images-idx3-ubyte", "./mnist/fashion/train-labels-idx1-ubyte",
                "./mnist/fashion/fashion_train.csv", 60000)

if __name__ == "__main__":
    convert2csv()
    if (int(input()) == 1):
        trainimg = './mnist/fashion/train-images-idx3-ubyte'
        trainlabel = './mnist/fashion/train-labels-idx1-ubyte'
        testimg = './mnist/fashion/t10k-images-idx3-ubyte'
        testlabel = './mnist/fashion/t10k-labels-idx1-ubyte'
        trainfolder = './mnist/train'
        testfolder = './mnist/test'
        if not os.path.exists(trainfolder): os.makedirs(trainfolder)
        if not os.path.exists(testfolder): os.makedirs(testfolder)
        # rb表示以二进制读模式打开文件
        trimg = open(trainimg, 'rb')
        teimg = open(testimg, 'rb')
        trlab = open(trainlabel, 'rb')
        telab = open(testlabel, 'rb')

        struct.unpack(">IIII", trimg.read(16))
        struct.unpack(">IIII", teimg.read(16))
        struct.unpack(">II", trlab.read(8))
        struct.unpack(">II", telab.read(8))
        # array模块是Python中实现的一种高效的数组存储类型
        # 所有数组成员都必须是同一种类型，在创建数组时就已经规定
        # B表示无符号字节型，b表示有符号字节型
        trimage = array("B", trimg.read())
        teimage = array("B", teimg.read())
        trlabel = array("b", trlab.read())
        telabel = array("b", telab.read())
        trimg.close()
        teimg.close()
        trlab.close()
        telab.close()
        # 为训练集和测试集各定义10个子文件夹，用于存放从0到9的所有数字，文件夹名分别为0-9
        trainfolders = [os.path.join(trainfolder, str(i)) for i in range(10)]
        testfolders = [os.path.join(testfolder, str(i)) for i in range(10)]
        for dir in trainfolders:
            if not os.path.exists(dir):
                os.makedirs(dir)
        for dir in testfolders:
            if not os.path.exists(dir):
                os.makedirs(dir)
        binary2img(trimage, trlabel, trainfolders)
        binary2img(teimage, telabel, testfolders)
        print("Done.\n")