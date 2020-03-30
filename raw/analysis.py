import matplotlib.pyplot as plt

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    plt.savefig('./rp_thresholds.jpg')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.savefig('./recall_precision.jpg')
    plt.show()

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.savefig('./roc.jpg')
    plt.shot()

plt.show()
def my_preprocessing(train_data):
    from sklearn import preprocessing
    X_normalized = preprocessing.normalize(train_data ,norm = "l2",axis=0)#使用l2范式，对特征列进行正则
    return X_normalized
 
def my_feature_selection(data, target):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    data_new = SelectKBest(chi2, k= 50).fit_transform(data,target)
    return data_new
 
def my_PCA(data):#data without target, just train data, withou train target.
    from sklearn import decomposition
    pca_sklearn = decomposition.PCA()
    pca_sklearn.fit(data)
    main_var = pca_sklearn.explained_variance_
    # print sum(main_var)*0.9
    import matplotlib.pyplot as plt
    n = 15
    plt.plot(main_var[:n])
    plt.show()
 
def clf_train(data,target):
    from sklearn import svm
    #from sklearn.linear_model import LogisticRegression
    clf = svm.SVC(C=100,kernel="rbf",gamma=0.001)
    clf.fit(data,target)
 
    #clf_LR = LogisticRegression()
    #clf_LR.fit(x_train, y_train)
    #y_pred_LR = clf_LR.predict(x_test)
    return clf
 
def my_confusion_matrix(y_true, y_pred, has_label=False, label_list=[]):
    from sklearn.metrics import confusion_matrix
    if has_label == False:
        labels = list(set(y_true))
    else:
        labels = label_list
    conf_mat = confusion_matrix(y_true, y_pred, labels = labels)
    #print "confusion_matrix(left labels: y_true, up labels: y_pred):"
    #print "labels\t",
    for i in range(len(labels)):
        print(labels[i],"\t")
    for i in range(len(conf_mat)):
        #print i,"\t",
        for j in range(len(conf_mat[i])):
            print(conf_mat[i][j],'\t')
        #print 
    #print 
 
def my_classification_report(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score
    #from sklearn.metrics import f1_score
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred,)
    plot_roc_curve(fpr, tpr, "ROC curve")

    from sklearn.metrics import precision_recall_curve
    precisions,recalls,thresholds =precision_recall_curve(y_true, y_pred)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    
    from sklearn.metrics import classification_report
    f = open("report.txt", 'w')
    print("classification_report(left: labels):", file=f)
    print(classification_report(y_true, y_pred), file=f)
    f.close()

