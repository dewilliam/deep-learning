#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from PIL import Image
# import struct
# import os
PATH="/media/william/文档/深度学习/图像标注/MNIST/"
def readimage(filename,save_path):
	# 读取整个二进制文件
	file=open(PATH+filename,'rb')
	index=0
	buff=file.read()
	file.close()
	# >IIII是读取四个整数
	magic_num,images_num,row,column=struct.unpack_from('>IIII',buff,index)
	index+=struct.calcsize('>IIII')
	for i in range(images_num):
		# 先生成一个空白图片，L是灰度图片
		image=Image.new('L',(column,row))
		for x in xrange(row):
			for y in xrange(column):
				# 把每一个像素写入到空白图片中
				image.putpixel((y,x),struct.unpack_from('>B',buff,index)[0])
				index+=struct.calcsize('>B')
		if not os.path.exists(PATH+save_path):
			os.mkdir(PATH+save_path)
			pass
		image.save(PATH+save_path+str(i)+".png")
	print "all images saved..."

def readlabel(filename,save_path):
	file=open(PATH+filename,'rb')
	index=0
	buff=file.read()
	file.close()
	magic_num,labels_num=struct.unpack_from('>II',buff,index)
	index+=struct.calcsize('>II')
	label_list=[]
	for i in range(labels_num):
		# 读取一个字节 转换成int数
		label=int(struct.unpack_from('>B',buff,index)[0])
		label_list.append(label)
		index+=struct.calcsize('>B')
	if not os.path.exists(PATH+save_path):
		os.mkdir(PATH+save_path)
		pass
	save_file=open(PATH+save_path+'label','a')
	save_file.write(','.join(map(lambda x: str(x), label_list)))
	save_file.write('\n')
	save_file.close()
	print "label saved..."

# readimage('train-images.idx3-ubyte','train_image/')
# readimage('t10k-images.idx3-ubyte','test_image/')
# readlabel('train-labels.idx1-ubyte','train_label/')
# readlabel('t10k-labels.idx1-ubyte','test_label/')
# im=Image.open(PATH+'train_image/0.png')
# import numpy
# im_arr=numpy.array(im)
# print im_arr
import tensorflow as tf

#cnn...
# import tensorflow.examples.tutorials.mnist.input_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 先定义两个函数，生成权值和偏置的变量
def weight_variable(shape):
	tensor=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(tensor)
def bias_variable(shape):
	tensor=tf.constant(0.1,shape=shape)
	return tf.Variable(tensor)

def conv(x,w):
	# strides是平移步长，[a,b,c,d]，a是batch步长，b是高度方向步长，c是宽度方向步长，d是通道步长.
	#padding=SAME 感受野不一定全部要在输入矩阵内，可以一部分超出边 界，超出部分置0
	return tf.nn.conv2d(x,w,strides=([1,1,1,1]),padding='SAME')
def pooling(x):
	# ksize是pooling感受野大小，四个维度和strides一样
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# 定义输入输出的占位符
x=tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder("float", [None,10])

w1=weight_variable([5,5,1,32])
b1=bias_variable([32])
w2=weight_variable([5,5,32,64])
b2=bias_variable([64])
w3=weight_variable([7*7*64,1024])
b3=bias_variable([1024])
w4=weight_variable([1024,10])
b4=bias_variable([10])
x_image = tf.reshape(x, [-1,28,28,1])
# 第一层卷积
h_conv1 = tf.nn.relu(conv(x_image, w1) + b1)
# 第一层采样
h_pool1 = pooling(h_conv1)
# 第2层卷积
h_conv2=tf.nn.relu(conv(h_pool1,w2)+b2)
# 第2层采样
h_pool2=pooling(h_conv2)
# 把第二层采样结果转换成一维向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 第一层全链接层
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w3) + b3)
# 设置dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 输出层 使用dropout
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, w4) + b4)

# 用交叉熵来定义成本函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# 定义学习速率
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 定义和计算正确率 用来评价模型
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 启动后台的session
sess = tf.Session()
# 初始化所有变量。。
sess.run(tf.initialize_all_variables())
# with tf.device("/cpu:0"):
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	# 训练模型，送入训练集输入输出和dropout概率
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob:0.5})
	correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y_conv,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	# 计算正确率
	print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob:1.0})
# with tf.device("/cpu:0"):
# 	for i in range(20000):
# 		batch = mnist.train.next_batch(50)
# 		if i%100 == 0:
# 			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
# 			print "step %d, training accuracy %g"%(i, train_accuracy)
# 			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# 			print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
