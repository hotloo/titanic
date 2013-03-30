from sklearn import cross_validation
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Classifier():

    N_FOLD = 20

    def classify(self, method, train_label, train_data, test_data):
        np.random.seed(12345)
        if not method:
            raise 
        if method == 'bayes':
            classifier = BernoulliNB()
        elif method == 'svm':
            classifier = svm.SVC(kernel = 'rbf', C = 1)
        elif method == 'logistic':
            classifier = LogisticRegression(C = 1, penalty = 'l1')
        elif method == 'randomforest':
            classifier = RandomForestClassifier(n_estimators=5)
        else:
            raise 'Not implemented'
        train_classifier = self.cross_validate_classify(classifier, train_label, train_data)
        prediction = classifier.predict(test_data)
        return prediction

    def get_pca_projection(self, train, test, snum = 10):
        pca = PCA(n_components=snum)
        transformed_train = pca.fit_transform(train)
        transformed_test = pca.transform(test)
        return (transformed_train, transformed_test)

    def cross_validate_classify(self, method, train_label, train_data):
        best_clf = None
        best_validation = 0.0
        kf = cross_validation.KFold(train_label.shape[0],n_folds = self.N_FOLD)
        for train_index,validate_index in kf:
            clf = method
            clf.fit(train_data[train_index],train_label[train_index])
            vali_score = clf.score(train_data[validate_index],train_label[validate_index])
            train_score = clf.score(train_data[train_index],train_label[train_index])
            print "The result train[%f] & validation[%f]"%(train_score, vali_score)
            if vali_score > best_validation:
                best_clf = clf
                best_validation = vali_score
        print "----"
        print "Selected the classifier with the best validation score[%f]"%best_validation
        return best_clf

#class Visualization():
