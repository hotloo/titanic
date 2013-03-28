from utils import *
from sklearn import cross_validation
from sklearn import svm
import ipdb

if __name__ == '__main__':
    util = TitaticUtils()
    (train_label, train_data, test_data) = util.create_feature_vector()
    kf = cross_validation.KFold(train_label.shape[0],n_folds = 10)
    best_clf = None
    best_validation = 0.0
    for train_index,validate_index in kf:
        clf = svm.SVC(kernel = 'rbf', C = 1)
        clf.fit(train_data[train_index],train_label[train_index])
        vali_score = clf.score(train_data[validate_index],train_label[validate_index])
        train_score = clf.score(train_data[train_index],train_label[train_index])
        print "The result for SVM train[%f] & validation[%f]"%(train_score, vali_score)
        if vali_score > best_validation:
            best_clf = clf
            best_validation = vali_score
    print "----"
    print "Selected the SVC with the best validation score[%f]"%best_validation
    predict_test_label = best_clf.predict(test_data)
    util.write_file(predict_test_label, '../data/test_result.csv')
