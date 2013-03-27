import csv as csv
import numpy as np
from sklearn import preprocessing
from collections import Counter

TRAIN_DATA = '../data/train.csv'
TEST_DATA = '../data/test.csv'

class TitaticUtils():

    def read_file(self, filename = ""):
        data = []
        data_size = []
        if not filename:
            filename = [TRAIN_DATA]
        for filename_ in filename:
            with open(filename_,'r') as f_handle:
                csv_file = csv.reader(f_handle)
                for line in csv_file:
                    data.append(line)
                data_size.append(len(data))
        data = np.array(data)
        return (data,data_size)

    def write_file(self, filename = ""):
        pass

    def create_feature_vector(self):
        (data,data_size) = self.read_file()
        # survival 
        label = np.array(map(self.get_int, data[1:,0]))
        # pclass
        _data = self.categorize_column_info(data[1:,1])
        # name We skip it for now
        # sex
        _data = np.concatenate((_data,
                self.categorize_column_info(data[1:,3])), axis = 1)
        # age
        _data = np.concatenate((_data,
                self.binize_column_info(data[1:,4],10)), axis = 1)
        # sibsp
        _data = np.concatenate((_data,
                self.binize_column_info(data[1:,5],4)), axis = 1)
        # parch
        _data = np.concatenate((_data,
                self.binize_column_info(data[1:,6],4)), axis = 1)
        # ticket We skip it for now
        # fare
        _data = np.concatenate((_data,
                self.binize_column_info(data[1:,8],10)), axis = 1)
        # cabin We skip it for now
        # embarked
        _data = np.concatenate((_data,
                self.categorize_column_info(data[1:,10])), axis = 1)
        return (label[:data_size[0]], _data[:data_size[0],:],label[data_size[0]:], _data[data_size[0]:,:])

    def categorize_column_info(self, column):
        filled_column = self.estimate_missing_values(column)
        le = preprocessing.LabelEncoder()
        le.fit(column)
        transformed_column = le.transform(column)
        output_array = np.zeros((filled_column.shape[0], le.classes_.shape[0]))
        for index, item in enumerate(transformed_column.tolist()):
            output_array[index, item] = 1
        return output_array

    def binize_column_info(self, column, n_bin = 5):
        filled_column = self.estimate_missing_values(column, estimation_type = 'continuous')
        max_column = np.max(filled_column)
        int_column = np.apply_along_axis(lambda x,m: np.divide(x,m), 0, filled_column, max_column)
        bins = np.linspace(np.min(filled_column),np.max(filled_column),n_bin)
        bin_column = np.digitize(int_column, bins, right = True)
        output_array = np.zeros((filled_column.shape[0],n_bin))
        for index, item in enumerate(bin_column.tolist()):
            output_array[index, item - 1] = 1
        return output_array

    def estimate_missing_values(self, column, estimation_type = 'category'):
        if estimation_type == 'category':
            distinct_value_count = Counter(column)
            if not '' in distinct_value_count.keys():
                return column
            largest_value = 0
            for key in distinct_value_count:
                if key == '':
                    next
                if largest_value < distinct_value_count[key]:
                    largest_key = key
                    largest_value = distinct_value_count[key]
            indices = self.get_index(column)
            column[indices] = largest_key
        elif estimation_type == 'continuous':
            int_column = map(self.get_float, column)
            new_int_column = np.ma.masked_array(int_column, np.isnan(int_column))
            mean = np.mean(new_int_column)
            column = np.array(map(lambda x: x if not np.isnan(x) else mean, int_column))
        else:
            raise
        return column

    def get_index(self, column, value = ''):
        indices = [item for item in xrange(column.shape[0]) if column[item] == value]
        return indices
    def get_float(self,s):
        return float(s) if s else np.NaN
    def get_int(self,s):
        return int(s) if s else np.NaN
if __name__ == '__main__':
    train = TitaticUtils()
    (train_label, train_data, test_label, test_data) = train.create_feature_vector()
