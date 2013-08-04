from utils import *
from sklearn import cross_validation
from sklearn import svm
from sklearn.utils import shuffle
from model import Classifier

import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load the utilities to clean up the data
    util = TitaticUtils()

    # load the feature vector
    (train_label, train_data, test_data) = util.create_feature_vector()

    # shuffle the data
    train_data, train_label = shuffle(train_data, train_label)

    # Load the classifier
    classifier = Classifier()

    # project the feature vector
    (train_pca, test_pca) = classifier.get_pca_projection(train_data, test_data, snum = 2)

    # Prediction
    predict_test_label = Classifier().classify('randomforest', train_label, train_pca, test_pca)

    util.write_file(predict_test_label, '../data/test_result.csv')

    # Visualization
    #plt.scatter(train_pca[:,0],train_pca[:,1], c = train_label)
    #plt.prism
    #plt.show()
