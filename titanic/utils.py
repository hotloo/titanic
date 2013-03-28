import csv as csv
import numpy as np
from sklearn import preprocessing
from collections import Counter
import ipdb

TRAIN_DATA = '../data/train.csv'
TEST_DATA = '../data/test.csv'

class TitaticUtils():

    def read_file(self, filename = ""):
        data = []
        if not filename:
            filename = [TRAIN_DATA,TEST_DATA]
        for filename_ in filename:
            with open(filename_,'r') as f_handle:
                csv_file = csv.reader(f_handle)
                for line in csv_file:
                    data.append(line)
                if filename_ == TRAIN_DATA:
                    train_data = np.array(data)
                else:
                    test_data = np.array(data)
                data = []
        return (train_data, test_data)

    def write_file(self, result, filename = ""):
        output_file = csv.writer(open(filename, 'wb'))
        test_file = csv.reader(open(TEST_DATA, 'rb'))
        test_file.next()
        i = 0
        for row in test_file:
            row.insert(0, result[i].astype(np.uint8))
            output_file.writerow(row)
            i += 1

    def create_feature_vector(self):
        (train_data,test_data) = self.read_file()
        # survival 
        label = np.array(map(self.get_int, train_data[1:,0]))
        # pclass
        (pclass_train_data,pclass_test_data) = self.categorize_column_info(train_data[1:,1], test_data[1:,0])
        # name We skip it for now
        # sex
        (sex_train_data, sex_test_data) = self.categorize_column_info(train_data[1:,3],test_data[1:,2])
        # age
        (age_train_data, age_test_data) = self.binize_column_info(train_data[1:,4],test_data[1:,3],10)
        # sibsp
        (sibsp_train_data, sibsp_test_data) = self.binize_column_info(train_data[1:,5],test_data[1:,4], 4)
        # parch
        (parch_train_data, parch_test_data) = self.binize_column_info(train_data[1:,6], test_data[1:,5],4)
        # ticket We skip it for now
        # fare
        (fare_train_data, fare_test_data) = self.binize_column_info(train_data[1:,8], test_data[1:,7], 10)
        # cabin We skip it for now
        # embarked
        (embarked_train_data, embarked_test_data) = self.categorize_column_info(train_data[1:,10], test_data[1:,9])

        train_data = np.concatenate((pclass_train_data, sex_train_data, age_train_data, sibsp_train_data, parch_train_data, fare_train_data, embarked_train_data), axis = 1)
        test_data = np.concatenate((pclass_test_data, sex_test_data, age_test_data, sibsp_test_data, parch_test_data, fare_test_data, embarked_test_data), axis = 1)
        return (label, train_data, test_data)

    def categorize_column_info(self, column, column_test):
        (filled_column, filled_column_test) = self.estimate_missing_values(column, column_test)
        le = preprocessing.LabelEncoder()
        le.fit(filled_column)
        transformed_column = le.transform(filled_column)
        transformed_column_test = le.transform(filled_column_test)
        output_array = np.zeros((filled_column.shape[0], le.classes_.shape[0]))
        output_array_test = np.zeros((filled_column_test.shape[0], le.classes_.shape[0]))
        for index, item in enumerate(transformed_column.tolist()):
            output_array[index, item] = 1
        for index, item in enumerate(transformed_column_test.tolist()):
            output_array_test[index, item] = 1
        return (output_array, output_array_test)

    def binize_column_info(self, column, column_test, n_bin = 5):
        (filled_column, filled_column_test) = self.estimate_missing_values(column, column_test, estimation_type = 'continuous')
        max_column = np.max(filled_column)
        int_column = np.apply_along_axis(lambda x,m: np.divide(x,m), 0, filled_column, max_column)
        int_column_test = np.apply_along_axis(lambda x,m: np.divide(x,m), 0, filled_column_test, max_column)
        bins = np.linspace(np.min(filled_column),np.max(filled_column),n_bin)
        bin_column = np.digitize(int_column, bins, right = True)
        bin_column_test = np.digitize(int_column_test, bins, right = True)
        output_array = np.zeros((filled_column.shape[0],n_bin))
        output_array_test = np.zeros((filled_column_test.shape[0],n_bin))
        for index, item in enumerate(bin_column.tolist()):
            output_array[index, item - 1] = 1
        for index, item in enumerate(bin_column_test.tolist()):
            output_array_test[index, item - 1] = 1
        return (output_array, output_array_test)

    def estimate_missing_values(self, column, column_test, estimation_type = 'category'):
        if estimation_type == 'category':
            distinct_value_count = Counter(column)
            largest_value = 0
            for key in distinct_value_count:
                if key == '':
                    next
                if largest_value < distinct_value_count[key]:
                    largest_key = key
                    largest_value = distinct_value_count[key]
            indices = self.get_index(column)
            indices_test = self.get_index(column_test)
            column[indices] = largest_key
            column_test[indices_test] = largest_key
        elif estimation_type == 'continuous':
            int_column = map(self.get_float, column)
            int_column_test = map(self.get_float, column_test)
            new_int_column = np.ma.masked_array(int_column, np.isnan(int_column))
            new_int_column_test = np.ma.masked_array(int_column_test, np.isnan(int_column_test))
            mean = np.mean(new_int_column)
            column = np.array(map(lambda x: x if not np.isnan(x) else mean, int_column))
            column_test = np.array(map(lambda x: x if not np.isnan(x) else mean, int_column_test))
        else:
            raise
        return (column, column_test)

    def get_index(self, column, value = ''):
        indices = [item for item in xrange(column.shape[0]) if column[item] == value]
        return indices
    def get_float(self,s):
        return float(s) if s else np.NaN
    def get_int(self,s):
        return int(s) if s else np.NaN

if __name__ == '__main__':
    train = TitaticUtils()
    (label, train_data, test_data)= train.create_feature_vector()
