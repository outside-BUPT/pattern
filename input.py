import tensorflow as tf
import matplotlib.pyplot as plt

''' 读取MNIST数据方法一'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

'''1)获得数据集的个数'''
train_nums = mnist.train.num_examples
validation_nums = mnist.validation.num_examples
test_nums = mnist.test.num_examples
print('MNIST数据集的个数')
print(' >>>train_nums=%d' % train_nums,'\n',
      '>>>validation_nums=%d'% validation_nums,'\n',
      '>>>test_nums=%d' % test_nums,'\n')

'''2)获得数据值'''
train_data = mnist.train.images   #所有训练数据
val_data = mnist.validation.images  #(5000,784)
test_data = mnist.test.images       #(10000,784)
print('>>>训练集数据大小：',train_data.shape,'\n',
      '>>>一副图像的大小：',train_data[0].shape)
'''3)获取标签值label=[0,0,...,0,1],是一个1*10的向量'''
train_labels = mnist.train.labels     #(55000,10)
val_labels = mnist.validation.labels  #(5000,10)
test_labels = mnist.test.labels       #(10000,10)

print('>>>训练集标签数组大小：',train_labels.shape,'\n',
      '>>>一副图像的标签大小：',train_labels[1].shape,'\n',
      '>>>一副图像的标签值：',train_labels[0])

'''4)批量获取数据和标签【使用next_batch(batch_size)】'''
batch_size = 100    #每次批量训练100幅图像
batch_xs,batch_ys = mnist.train.next_batch(batch_size)
print('使用mnist.train.next_batch(batch_size)批量读取样本\n')
print('>>>批量读取100个样本:数据集大小=',batch_xs.shape,'\n',
      '>>>批量读取100个样本:标签集大小=',batch_ys.shape)
#xs是图像数据(100,784);ys是标签(100,10)

'''5)显示图像'''
plt.figure()
for i in range(100):
    im = train_data[i].reshape(28,28)
    im = batch_xs[i].reshape(28,28)
    plt.imshow(im,'gray')
    plt.pause(0.0000001)
plt.show()