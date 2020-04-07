#-*- coding=UTF-8 -*-
from PIL import Image
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
    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)

    '''
    param_grid = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [0.1, 1, 10, 100, 1000], 'decision_function_shape': ['ovr']},
                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],'decision_function_shape': ['ovr']},
                    {'kernel': ['sigmoid'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'coef0': [0.1, 0, 1, 3], 'C': [0.1, 1, 10, 100, 1000],'decision_function_shape': ['ovr']},
                    {'kernel': ['poly'], 'degree': [3,4,5,6], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'coef0': [0.1, 0, 1, 3],'decision_function_shape': ['ovr']}]
    print("Tuning parameters: %s" % tuned_params)
    scores = ['precision', 'recall']
    clf_list = grid_search(tuned_params, scores, [dataMat, dataLabel])
    if (input("Dump models now? [y/n]") == 'y'):
        for clf in clf_list:
            joblib.dump(clf, path)
    return clf_list
    Best parameters found for each kernel
    clf = svm.SVC(C=10, kernel='rbf', gamma=0.01, decision_function_shape=decision)
    clf = svm.SVC(C=10,kernel='linear',decision_function_shape=decision)
    '''
    clf = svm.SVC(C=100,kernel='sigmoid',coef0=0.1, gamma=1, decision_function_shape=decision)
    clf = clf.fit(dataMat, dataLabel)
    #if (input("Dump models now? [y/n]") == 'y'):
    #    joblib.dump(clf, path)
    return clf

def model_test(model_path, clf=None):
    path = sys.path[0]
    #path = sys.path[1]
    #path = "E:/Statistical-Learning-Method_Code/raw"
    tbasePath = os.path.join(path, "mnist/test/")
    tcName = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tst = time.time()
    f = open("out.txt", "w")
    allErrCount = 0; allErrorRate = 0.0; allScore = 0.0
    ErrCount=np.zeros(10,int); TrueCount=np.zeros(10,int)
    predict_list = []; true_list = []

    if clf == None:
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
    #dataMat, dataLabel = utils.read_folder_img()
    dataMat, dataLabel = utils.read_folder_img()
    #path = sys.path[1]
    path = sys.path[0]
    #path = "E:/Statistical-Learning-Method_Code/raw"
    model_path=os.path.join(path,'model/svm_best.model')
    print(model_path)
    if not os.path.exists(model_path):
        if not os.path.exists(os.path.join(path, 'model/')):
            os.makedirs(os.path.join(path, 'model'))
        print("start training.\n")
        clf_list = create_svm(dataMat, dataLabel, model_path, decision='ovr')

        et = time.time()
        with open("report.txt", 'w') as f:
            print("Training spent {:.4f}s.".format((et - st)), file=f)
    else:
        print("Model found.\n")


    clf = clf_list
    y_predict, y_true = model_test(model_path, clf=clf)
    
    from sklearn.metrics import classification_report
    with open("report.txt", 'a') as f:
        print("classification report(left: labels):", file=f)
        print(classification_report(y_true, y_predict), file=f)

    analysis.my_classification_report(y_true, y_predict, True)
    np.savetxt('y_pred.txt', y_predict, fmt="%s", delimiter=',')
    np.savetxt('y_true.txt', y_true, fmt="%s", delimiter=',')
    #analysis.my_confusion_matrix(y_true, y_predict)
    #analysis.my_confusion_matrix(y_true, y_predict, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
 
 #'C:\\Miniconda3\\envs\\HybridPy\\python38.zip\\model\\svm.model'
 #FileNotFoundError: [Errno 2] No such file or directory: 'E:/Statistical-Learning-Method_Code/raw\\model/svm.model'
