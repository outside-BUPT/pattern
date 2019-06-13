import tensorflow as tf
import matplotlib.pyplot as plt

''' 读取MNIST数据方法一'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

sess=tf.InteractiveSession()
#构建cnn网络结构

#自定义卷积函数（后面卷积时就不用写太多）

def conv2d(x,w):
   c = tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
   return c


#自定义池化函数

def max_pool_2x2(x):
    d = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    return d

#设置占位符，尺寸为样本输入和输出的尺寸
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_img = tf.reshape(x,[-1,28,28,1])

#设置第一个卷积层和池化层
w_conv1=tf.Variable(tf.truncated_normal([4, 4, 1, 24], stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1, shape=[24]))
h_conv1=tf.nn.relu(conv2d(x_img, w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

#设置第二个卷积层和池化层
w_conv2=tf.Variable(tf.truncated_normal([4,4,24,48],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[48]))
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#设置第一个全连接层
w_fc1=tf.Variable(tf.truncated_normal([7*7*48, 1024],stddev=0.1))
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]))
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*48])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

#dropout（随机权重失活）
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#设置第二个全连接层
w_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2=tf.Variable(tf.constant(0.1,shape=[10]))
y_out=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#建立loss function，为交叉熵

loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_out),reduction_indices=[1]))
#配置Adam优化器，学习速率为1e-4
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵

#建立正确率计算表达式
#tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
# 而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
#这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
#例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#开始喂数据，训练
#该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(100) #每次批量训练50幅图像
    if i%100==0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_:batch[1], keep_prob:1})
        print ("step %d,train_accuracy= %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#训练之后，使用测试集进行测试，输出最终结果
print ("test_accuracy= %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))