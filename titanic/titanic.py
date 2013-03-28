from utils import *
from sklearn import cross_validation
from sklearn import svm
from sklearn.utils import shuffle
from model import Classifier

if __name__ == '__main__':
    util = TitaticUtils()
    (train_label, train_data, test_data) = util.create_feature_vector()
    train_data, train_label = shuffle(train_data, train_label)
    predict_test_label = Classifier().classify('bayes', train_label, train_data, test_data)
    util.write_file(predict_test_label, '../data/test_result.csv')
