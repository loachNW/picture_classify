import tensorflow as tf
import pickle
import random

def choose(number):
    num = range(len(train_y))
    id = random.sample(num,number)
    banch_x = train_x[id]
    banch_y = train_y[id]
    return banch_x,banch_y

def compute_accuracy(v_xs, v_ys):  #Correct rate 训练并给出正确率
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})#使用测试集做一个预测
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1)) #损失函数  找出预测值中的最大值，返回值为ture和false
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#正确率tf.cast转化为零一序列形式
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

#设置weight以及baises,避免重复初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#产生一个截断式正态分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)#创建一个常数张量,传入list或者数值来填充
    return tf.Variable(initial)



#设置weight以及baises,避免重复初始化



def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 3, 3, 1], padding='SAME')#W在这里为卷积核

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    #ksize  [1,pool_op_length,pool_op_width,1]
    # Must have ksize[0] = ksize[3] = 1
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 14400])    # 28x28
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 120, 120, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([15,15, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 120x120x32
h_pool1 = max_pool_2x2(h_conv1)                          # output size 60x60x32

## conv2 layer ##
W_conv2 = weight_variable([15,15, 32, 128]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 60x60x128
h_pool2 = max_pool_2x2(h_conv2)                          # output size 30x30x64

##flat h_pool2##
h_pool2_flat = tf.reshape(h_pool2, [-1, 512*4])  # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]

## fc1 layer ## 输入输出以及隐藏层
W_fc1 = weight_variable([512*4, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
predictions= tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
prediction = tf.clip_by_value(predictions, 1e-10, 1.0)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),      # 分类问题中常用交叉熵来作损失函数  reduction_indices  减少指数
                                              reduction_indices=[1]))       # loss另一种方式计算损失函数 indices为在哪一围上求解-
train_step = tf.train.AdamOptimizer(1e-8).minimize(cross_entropy)           #优化器

init = tf.global_variables_initializer()                                    #初始化所有变量

sess = tf.Session()


import matplotlib.pyplot as plt

sess.run(init)
with open("cat&dog.pkl","rb") as f:
    train_x = pickle.load(f)
    train_y = pickle.load(f)
    test_x = pickle.load(f)
    test_y = pickle.load(f)
for i in range(1000):
    batch_xs, batch_ys = choose(100)   #每次进10张照片
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})
    # datas = sess.run(x_image,feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    # data = datas[1]
    # plt.imshow(data.reshape(120,120), cmap=plt.cm.gray, interpolation='nearest')
    # plt.show()

    if i % 5 == 0:
        print(compute_accuracy(test_x, test_y))
        print(sess.run(cross_entropy,feed_dict={xs: batch_xs, ys: batch_ys,keep_prob: 0.5}))