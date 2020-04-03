
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC


def grid_search(parameters, scores, trainingset, filename):
    x_train = trainingset[0]; y_train = trainingset[1]
    clf_list = []
    with open(filename, 'w') as f:
        for score in scores:
            f.write("Tuning hyper-parameters for %s\n" % score)
            f.write("\tparameters = %s, cv = 10\n" % parameters)
            clf = GridSearchCV(SVC(), parameters, cv=10, scoring='%s_macro' % score)
            clf.fit(x_train, y_train)
            clf_list.append(clf)
            f.write("Best parameters set found on :\n")
            best = clf.best_params_
            print(best)
            f.write("\t%s\n" % best)
            f.write("\nGrid scores on development set:\n")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']

            for mean, std, para in zip(means, stds, clf.cv_results_['params']):
                f.write("\t%0.3f (+/-%0.03f) for %r\n" % (mean, std*2, para))
    return clf_list
            
if __name__ == '__main__':

    iris = load_iris()
    param_grid = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [0.1, 1, 10, 100, 1000], 'decision_function_shape': ['ovr']},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000],'decision_function_shape': ['ovr']},
                    {'kernel': ['sigmoid'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'coef0': [0.1, 0, 1, 3], 'C': [0.1, 1, 10, 100, 1000],'decision_function_shape': ['ovr']},
                    {'kernel': ['poly'], 'degree': [3,4,5,6], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'coef0': [0.1, 0, 1, 3],'decision_function_shape': ['ovr']}]
    print("Parameters:{}".format(param_grid))
    
    X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=10)
    if (input("test grid func? [y/n]") == 'y'):
        clf = grid_search(param_grid, ['precision', 'recall'], [X_train, y_train])

    grid_search = GridSearchCV(SVC(),param_grid,cv=5) #?????GridSearchCV?
    grid_search.fit(X_train,y_train) #???????????????????????????SVC estimator?
    print("Test set score:{:.2f}".format(grid_search.score(X_test,y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))
