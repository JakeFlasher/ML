sklearn svm 相关参数的官方说明

Parameters:
C : float, optional (default=1.0). Penalty parameter C of the error term.
kernel : string, optional (default=’rbf’). Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
degree : int, optional (default=3). Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
gamma : float, optional (default=’auto’). Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
coef0 : float, optional (default=0.0). Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
probability : boolean, optional (default=False). Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
shrinking : boolean, optional (default=True). Whether to use the shrinking heuristic.
tol : float, optional (default=1e-3). Tolerance for stopping criterion.
cache_size : float, optional. Specify the size of the kernel cache (in MB).
class_weight : {dict, ‘balanced’}, optional. Set the parameter C of class i to class_weight[i]C for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classesnp.bincount(y))
verbose : bool, default: False. Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.
max_iter : int, optional (default=-1). Hard limit on iterations within solver, or -1 for no limit.
decision_function_shape : ‘ovo’, ‘ovr’ or None, default=None. Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). The default of None will currently behave as ‘ovo’ for backward compatibility and raise a deprecation warning, but will change ‘ovr’ in 0.19.
New in version 0.17: decision_function_shape=’ovr’ is recommended.
Changed in version 0.17: Deprecated decision_function_shape=’ovo’ and None.
random_state : int seed, RandomState instance, or None (default). The seed of the pseudo random number generator to use when shuffling the data for probability estimation.
libsvm中参数说明

因为sklearn底层是调用libsvm的，因此sklearn中svm参数说明是可以直接参考libsvm中的。

1.linear核函数:
K(xi,xj)=xTixj


2.polynomial核函数:
K(xi,xj)=(γxTixj+r)d,d>1


3.RBF核函数（高斯核函数）:
K(xi,xj)=exp(−γ||xi−xj||2),γ>0


4.sigmoid核函数:
K(xi,xj)=tanh(γxTixj+r),γ>0,r<0

 

首先介绍下与核函数相对应的参数：
1）对于线性核函数，没有专门需要设置的参数
2）对于多项式核函数，有三个参数。-d用来设置多项式核函数的最高次项次数，也就是公式中的d，默认值是3。-g用来设置核函数中的gamma参数设置，也就是公式中的gamma，默认值是1/k（特征数）。-r用来设置核函数中的coef0，也就是公式中的第二个r，默认值是0。
3）对于RBF核函数，有一个参数。-g用来设置核函数中的gamma参数设置，也就是公式中gamma，默认值是1/k（k是特征数）。
4）对于sigmoid核函数，有两个参数。-g用来设置核函数中的gamma参数设置，也就是公式中gamma，默认值是1/k（k是特征数）。-r用来设置核函数中的coef0，也就是公式中的第二个r，默认值是0。

具体来说说rbf核函数中C和gamma ：

SVM模型有两个非常重要的参数C与gamma。其中 C是惩罚系数，即对误差的宽容度。c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差

gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。

这里面大家需要注意的就是gamma的物理意义，大家提到很多的RBF的幅宽，它会影响每个支持向量对应的高斯的作用范围，从而影响泛化性能。我的理解：如果gamma设的太大，方差会很小，方差很小的高斯分布长得又高又瘦， 会造成只会作用于支持向量样本附近，对于未知样本分类效果很差，存在训练准确率可以很高，(如果让方差无穷小，则理论上，高斯核的SVM可以拟合任何非线性数据，但容易过拟合)而测试准确率不高的可能，就是通常说的过训练；而如果设的过小，则会造成平滑效应太大，无法在训练集上得到特别高的准确率，也会影响测试集的准确率。

此外，可以明确的两个结论是：
结论1：样本数目少于特征维度并不一定会导致过拟合，这可以参考余凯老师的这句评论：
“这不是原因啊，呵呵。用RBF kernel, 系统的dimension实际上不超过样本数，与特征维数没有一个trivial的关系。”

结论2：RBF核应该可以得到与线性核相近的效果（按照理论，RBF核可以模拟线性核），可能好于线性核，也可能差于，但是，不应该相差太多。
当然，很多问题中，比如维度过高，或者样本海量的情况下，大家更倾向于用线性核，因为效果相当，但是在速度和模型大小方面，线性核会有更好的表现。