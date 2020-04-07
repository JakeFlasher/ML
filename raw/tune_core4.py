#-*- coding=UTF-8 -*-
from PIL import Image
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import sys
import numpy as np
import time
from sklearn import svm
from tuning import grid_search
import utils 
import analysis
import matplotlib.pyplot as plt
import joblib

# create model
def create_svm(dataMat, dataLabel,path,decision='ovr'):

    param_grid = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],'decision_function_shape': ['ovr']}]
    print("Parameters:{}".format(param_grid))
    scores = ['precision', 'recall']
    print("Tuning parameters: %s" % param_grid)
    clf_list = grid_search(param_grid, scores, [dataMat, dataLabel], './tuning4/tuningt.txt')
    
    return clf_list

def model_test(model_path, clf=None, filename='out.txt', tcName=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
    path = sys.path[0]
    tbasePath = os.path.join(path, "mnist/test/")
    #tcName = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tst = time.time()
    f = open(filename, "w")
    feature_cnt = len(tcName)
    allErrCount = 0; allErrorRate = 0.0; allScore = 0.0
    ErrCount=np.zeros(feature_cnt,int); TrueCount=np.zeros(feature_cnt,int)
    predict_list = []; true_list = []

    if clf == None:
        clf = joblib.load(model_path)

    for tcn in tcName:
        testPath = tbasePath + tcn
        tflist = utils.file2list(testPath)
        tdataMat, tdataLabel = utils.read2convert(tflist)
        print("test dataMat shape: {0}, test dataLabel len: {1} ".format(tdataMat.shape, len(tdataLabel)), file=f)
        pre_st = time.time(); preResult = clf.predict(tdataMat); pre_et = time.time()
        predict_list.append(preResult); true_list.append([tcn]*len(preResult))
        print("Recognition  " + tcn + " spent {:.4f}s.".format((pre_et - pre_st)), file=f)
        print("predict result: {}".format(len(preResult)))
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
    avgAccuracy = allScore / (feature_cnt*1.0)
    print("Average accuracy is: {:.6f}.".format(avgAccuracy), file=f)
    print("Average error rate is: {:.6f}.".format(1 - avgAccuracy), file=f)
    print("number"," TrueCount"," ErrCount", file=f)
    for tcn in tcName:
        tcn=int(tcn)
        print(tcn,"     ",TrueCount[tcn],"      ",ErrCount[tcn], file=f)
    plt.figure(figsize=(12, 6))
    x=list(range(feature_cnt))
    plt.plot(x,TrueCount, color='blue', label="TrueCount")  # 将正确的数量设置为蓝色
    plt.plot(x,ErrCount, color='red', label="ErrCount")  # 将错误的数量为红色
    plt.legend(loc='best')  # 显示图例的位置，这里为右下方
    plt.title('Projects')
    plt.xlabel('number')  # x轴标签
    plt.ylabel('count')  # y轴标签
    plt.xticks(np.arange(feature_cnt), tcName)
    plt.savefig('./accuracy_%s.jpg' % filename.split('_')[1])
    plt.show()
    f.close()

    true_list = np.array(true_list).flatten()
    predict_list = np.array(predict_list).flatten()
    return [predict_list, true_list]

if __name__ == '__main__':

    st = time.time()
    iris = load_iris()

    dataMat, dataLabel = utils.read_folder_img(cName = ['1', '2', '3', '4', '5', '6', '7', '8', '9'], delimit=6)
    print(dataMat.shape); print(dataLabel)
    path = sys.path[0]
    model_path=os.path.join(path,'model/svm_best.model')    
    
    clf_list = create_svm(dataMat, dataLabel, model_path, decision='ovr')
    i = 0; 
    for clf in clf_list:
        scores = ['precision', 'accuracy']
        y_predict, y_true = model_test(model_path, clf, './tuning4/out_%s_test.txt' % scores[i], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        analysis.my_classification_report(y_true, y_predict, True, './tuning4/report_%s_test.txt' % scores[i])
        np.savetxt('./tuning4/y_pred_%s.txt' % scores[i], y_predict, fmt="%s", delimiter=',')
        np.savetxt('./tuning4/y_true_%s.txt' % scores[i], y_true, fmt="%s", delimiter=',')
        i += 1