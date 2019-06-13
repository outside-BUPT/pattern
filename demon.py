# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 10:38
# @Author  : Barry
# @File    : mnist_svm.py
# @Software: PyCharm Community Edition

import pickle
import gzip

# Third-party libraries
import numpy as np


def load_data():
    """z
    返回包含训练数据、验证数据、测试数据的元组的模式识别数据
    训练数据包含50，000张图片，测试数据和验证数据都只包含10,000张图片
    """
    f = gzip.open('./MNIST_data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)


# Third-party libraries
from sklearn import svm
import time


def svm_baseline():
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    training_data, validation_data, test_data = load_data()
    # 传递训练模型的参数，这里用默认的参数
    print("C=1, kernel='rbf', gamma=0.013, decision_function_shape='ovr'")

    clf = svm.SVC( C=1, kernel='rbf', gamma=0.013, decision_function_shape='ovr')
    # clf = svm.SVC(C=8.0, kernel='rbf', gamma=0.00,cache_size=8000,probability=False)
    # 进行模型训练
    clf.fit(training_data[0], training_data[1])
    # test
    # 测试集测试预测结果
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("%s of %s test values correct." % (num_correct, len(test_data[1])))
    print('decision_function:\n', clf.decision_function(training_data[0]))
    print(time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == "__main__":
    svm_baseline()