from utils import *
from sklearn import cross_validation
from sklearn import svm

if __name__ == '__main__':
    util = TitaticUtils()
    (train_label, train_data, test_label, test_data) = util.create_feature_vector()
    import ipdb;ipdb.set_trace()
    clf = svm.SVC(kernel = 'rbf', C = 1)
    kf = cross_validation.KFold(train_label.shape[0],n_folds = 5)
    for train_index,validate_index in kf:
        clf.fit(train_data[train_index],train_label[train_index])
        vali_score = clf.score(train_data[validate_index],train_label[validate_index])
        train_score = clf.score(train_data[train_index],train_label[train_index])
        print "The result for SVM train[%f] & validation[%f]"%(train_score, vali_score)
