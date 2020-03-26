#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
# accuracy = (TN + TP) / (TP + FN + TN + FP)
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# F_beta = (1 + beta) * (precision + recall) / (beta * precision + recall)
# sentivity = TP / (TP + FN)      ]-
#									 ROC-curve
# 1-specificity = FP / (FP + TN)  ]-

def perf_measure(y_true, y_pred, labels, kind='binary'):
	TP, FP, TN, FN = 0, 0, 0, 0
	
	if (kind == 'binary'):
		return confusion_matrix(y_pred, y_true, labels), confusion_matrix(y_pred,y_true,labels).ravel()

	elif (kind == 'multinary'):
		cm = confusion_matrix(y_pred, y_true, labels)
		FP = cm.sum(axis=0) - np.diag(cm)
		FN = cm.sum(axis=1) - np.diag(cm)
		TP = np.diag(cm)
		TN = cm.sum() - (FP + FN + TP)
		return cm, TN, FP, FN, TP

def plot_confusion_mat(cm, savename, labels, title='Confusion Matrix'):
	np.set_printoptions(precision=2)
	plt.figure(figsize=(12, 8), dpi=100)

	ind_array = np.arrage(len(labels))
	x, y = meshgrid(ind_array, ind_array)
	for x_val, y_val in zip(x.flatten(), y.flatten()):
		c = cm[y_val][x_val]
		if c > 0.01:
			plt.text(x_val, y_val, "%0.4f" % (c,), color='red', fontsize=15, va='center', ha='center')

	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)


if __name__ == "__main__":
	real_val = [1,1,2,3,2,4,5,2,1,2,3,4,5,1,0,3,2]
	preds_val = [1,0,2,3,2,4,5,2,1,2,3,4,5,1,2,3,2]
	labels = [0,1,2,3,4,5]
	MAE = mean_absolute_error(real_val, preds_val)
	print(MAE)
	cm,tn,fp,fn,tp = perf_measure(real_val, preds_val, labels, kind='multinary')
	print(cm)
	print(tn, fp, fn, tp)