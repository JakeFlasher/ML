import matplotlib.pyplot as plt
import numpy as np

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    plt.savefig('./rp_thresholds.jpg')
    #plt.show()

def plot_precision_vs_recall(precisions, recalls):
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.savefig('./recall_precision.jpg')
    #plt.show()

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.savefig('./roc.jpg')
    #plt.show()

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)
     # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    #plt.show()

def FAULTYmy_preprocessing(train_data):
    from sklearn import preprocessing
    X_normalized = preprocessing.normalize(train_data ,norm = "l2",axis=0)#使用l2范式，对特征列进行正则
    return X_normalized
 
def FAULTYmy_feature_selection(data, target):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    data_new = SelectKBest(chi2, k= 50).fit_transform(data,target)
    return data_new
 
def FAULTYmy_PCA(data):#data without target, just train data, withou train target.
    from sklearn import decomposition
    pca_sklearn = decomposition.PCA()
    pca_sklearn.fit(data)
    main_var = pca_sklearn.explained_variance_
    # print sum(main_var)*0.9
    import matplotlib.pyplot as plt
    n = 15
    plt.plot(main_var[:n])
    #plt.show()
 
def FAULTYclf_train(data,target):
    from sklearn import svm
    #from sklearn.linear_model import LogisticRegression
    clf = svm.SVC(C=100,kernel="rbf",gamma=0.001)
    clf.fit(data,target)
 
    #clf_LR = LogisticRegression()
    #clf_LR.fit(x_train, y_train)
    #y_pred_LR = clf_LR.predict(x_test)
    return clf
 
def print_s(name, value, sign='=', filename='metrics.txt'):
    from contextlib import redirect_stdout
    import os
    if os.path.exists(filename):
        with open(filename, 'a') as f:
            with redirect_stdout(f):
                print(name, sign, value)
                #print("%s = " % name)
                #print(str(value).rjust(len(name)+len(str(value))))
                #print("%s = " % name, value)
    else:
        with open(filename, 'w') as f:
            with redirect_stdout(f):
                print(name, sign, value)
                #print("%s = " % name)
                #print(str(value).rjust(len(name)+len(str(value))))
                #print("%s = " % name, value)

def my_metrics(TN, FP, FN, TP):
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print_s('True positive rate', TPR)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    print_s('True Negative rate', TNR)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print_s('Positive predictive value', PPV)
    # Negative predictive value
    NPV = TN/(TN+FN)
    print_s('Negative predictive value', NPV)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print_s('False positive rate', FPR)
    # False negative rate
    FNR = FN/(TP+FN)
    print_s('False negative rate', FNR)
    # False discovery rate
    FDR = FP/(TP+FP)
    print_s('False discovery rate', FDR)

    precision = TP / (TP+FP)  # 查准率
    print_s('Precision', precision)
    recall = TP / (TP+FN)  # 查全率
    print_s('Recall', recall)

def my_confusion_matrix(y_true, y_pred, label_list=[]):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import f1_score
    if len(label_list) == 0:
        labels = list(set(y_true))
    else:
        labels = label_list
    
    conf_mat = confusion_matrix(y_true, y_pred, labels = labels)
    print_s("Confusion matrix", conf_mat, '\n')
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(conf_mat, 'conf_mat.png', 'confusion matrix')
    print_s("Normalized confusion matrix", conf_mat, '\n')

    if len(label_list) == 2:
        TN, FP, FN, TP = conf_mat.ravel()
    
    else:
        FP = conf_mat.sum(axis=0) - np.diag(conf_mat)  
        FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
        TP = np.diag(conf_mat)
        TN = conf_mat.sum() - (FP + FN + TP)

    my_metrics(TN, FP, FN, TP)

    p_class, r_class, f_class, support_micro=precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average=None)
    #print(p_class); print(r_class); print(f_class); print(support_micro)
    print_s('Macro F1 score', f_class.mean())
    print_s('Micro F1 score', f1_score(y_true, y_pred, average='micro'))
    print_s('Weighted F1 score', f1_score(y_true, y_pred, average='weighted'))

    np.savetxt('conf_mat.txt', conf_mat, delimiter=',')
    
    return [p_class,r_class, f_class]
 
def my_classification_report(y_true, y_pred, multiclass=False):
    from sklearn.metrics import precision_score, recall_score
    #from sklearn.metrics import f1_score
    if multiclass == False:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        plot_roc_curve(fpr, tpr, "ROC curve")
        from sklearn.metrics import precision_recall_curve    
        precisions,recalls,thresholds = precision_recall_curve(y_true, y_pred)
        plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    my_confusion_matrix(y_true, y_pred)

    from sklearn.metrics import classification_report
    f = open("report.txt", 'w')
    print("classification_report(left: labels):", file=f)
    print(classification_report(y_true, y_pred), file=f)
    f.close()

if __name__ == '__main__':
    classes = ['A', 'B', 'C', 'D', 'E', 'F']

    random_numbers = np.random.randint(6, size=500)  # 6个类别，随机生成50个样本
    y_true = random_numbers.copy()  # 样本实际标签
    random_numbers[:100] = np.random.randint(6, size=100)  # 将前10个样本的值进行随机更改
    y_pred = random_numbers  # 样本预测标签

    # 获取混淆矩阵
    #my_confusion_matrix(y_true, y_pred)
    my_classification_report(y_true, y_pred, True)
