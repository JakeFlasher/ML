#-*- coding=UTF-8 -*-
from PIL import Image
import os
import sys
import numpy as np
import time
from sklearn import svm
import utils 
import analysis
import matplotlib.pyplot as plt
import joblib
# create model
def create_svm(dataMat, dataLabel,path,decision='ovr'):
    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
    clf = svm.SVC(C=1.0,kernel='rbf',decision_function_shape=decision)
    rf =clf.fit(dataMat, dataLabel)
    joblib.dump(rf, path)
    return clf
'''
SVC参数
svm.SVC(C=1.0,kernel='rbf',degree=3,gamma='auto',coef0=0.0,shrinking=True,probability=False,
tol=0.001,cache_size=200,class_weight=None,verbose=False,max_iter=-1,decision_function_shape='ovr',random_state=None)
C：C-SVC的惩罚参数C?默认值是1.0
C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时
准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
       0 – 线性：u'v
 　　 1 – 多项式：(gamma*u'*v + coef0)^degree
  　　2 – RBF函数：exp(-gamma|u-v|^2)
  　　3 –sigmoid：tanh(gamma*u'*v + coef0)
degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。（没用）
gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。（没用）
probability ：是否采用概率估计？.默认为False
shrinking ：是否采用shrinking heuristic方法，默认为true
tol ：停止训练的误差值大小，默认为1e-3
cache_size ：核函数cache缓存大小，默认为200
class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
verbose ：允许冗余输出？
max_iter ：最大迭代次数。-1为无限制。
decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3（选用ovr，一对多）
random_state ：数据洗牌时的种子值，int值
主要调节的参数有：C、kernel、degree、gamma、coef0
'''
def svm_OVR_test(model_path):
    #path = sys.path[1]
    path = "E:/Statistical-Learning-Method_Code/raw"
    tbasePath = os.path.join(path, "mnist/test/")
    tcName = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tst = time.time()
    f = open("out.txt", "w")
    allErrCount = 0; allErrorRate = 0.0; allScore = 0.0
    ErrCount=np.zeros(10,int); TrueCount=np.zeros(10,int)
    predict_list = []; true_list = []
    #加载模型
    clf = joblib.load(model_path)
    for tcn in tcName:
        testPath = tbasePath + tcn
        tflist = utils.file2list(testPath)
        tdataMat, tdataLabel = utils.read2convert(tflist)
        #print("test dataMat shape: {0}, test dataLabel len: {1} ".format(tdataMat.shape, len(tdataLabel)), file=f)
        pre_st = time.time(); preResult = clf.predict(tdataMat); pre_et = time.time()
        predict_list.append(preResult); true_list.append([tcn]*len(preResult))
        print("Recognition  " + tcn + " spent {:.4f}s.".format((pre_et - pre_st)), file=f)
        # print("predict result: {}".format(len(preResult)))
        errCount = len([x for x in preResult if x != tcn])
        ErrCount[int(tcn)]=errCount
        TrueCount[int(tcn)]= len(tdataLabel)-errCount
        print("errorCount: {}.".format(errCount), file=f)
        allErrCount += errCount
        score_st = time.time()
        score = clf.score(tdataMat, tdataLabel)
        score_et = time.time()
        print("computing score spent {:.6f}s.".format(score_et - score_st), file=f)
        allScore += score
        print("score: {:.6f}.".format(score), file=f)
        print("error rate is {:.6f}.".format((1 - score)), file=f)
 
    tet = time.time()
    print("Testing All class total spent {:.6f}s.".format(tet - tst), file=f)
    print("All error Count is: {}.".format(allErrCount), file=f)
    avgAccuracy = allScore / 10.0
    print("Average accuracy is: {:.6f}.".format(avgAccuracy), file=f)
    print("Average error rate is: {:.6f}.".format(1 - avgAccuracy), file=f)
    print("number"," TrueCount"," ErrCount", file=f)
    for tcn in tcName:
        tcn=int(tcn)
        print(tcn,"     ",TrueCount[tcn],"      ",ErrCount[tcn], file=f)
    plt.figure(figsize=(12, 6))
    x=list(range(10))
    plt.plot(x,TrueCount, color='blue', label="TrueCount")  # 将正确的数量设置为蓝色
    plt.plot(x,ErrCount, color='red', label="ErrCount")  # 将错误的数量为红色
    plt.legend(loc='best')  # 显示图例的位置，这里为右下方
    plt.title('Projects')
    plt.xlabel('number')  # x轴标签
    plt.ylabel('count')  # y轴标签
    plt.xticks(np.arange(10), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.savefig('./accuracy.jpg')
    plt.show()
    f.close()

    true_list = np.array(true_list).flatten()
    predict_list = np.array(predict_list).flatten()
    return [predict_list, true_list]
 
if __name__ == '__main__':
    # clf = svm.SVC(decision_function_shape='ovr')
    st = time.time()
    dataMat, dataLabel = utils.read_folder_img()
    # path = sys.path[1]
    path = "E:/Statistical-Learning-Method_Code/raw"
    model_path=os.path.join(path,'model/svm_fashion.model')
    if not os.path.exists(model_path):
        print("start training.\n")
        create_svm(dataMat, dataLabel, model_path, decision='ovr')
        et = time.time()
        print("Training spent {:.4f}s.".format((et - st)))
    else:
        print("Model found.\n")
    y_predict, y_true = svm_OVR_test(model_path)

    #analysis.my_classification_report(y_true, y_predict, True)
    np.savetxt('y_pred.txt', y_predict, delimiter=',')
    np.savetxt('y_true.txt', y_true, delimiter=',')
    analysis.my_confusion_matrix(y_true, y_predict, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
 
 #'C:\\Miniconda3\\envs\\HybridPy\\python38.zip\\model\\svm.model'
 #FileNotFoundError: [Errno 2] No such file or directory: 'E:/Statistical-Learning-Method_Code/raw\\model/svm.model'
