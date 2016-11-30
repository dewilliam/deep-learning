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

#softmax....one layer
# import tensorflow.examples.tutorials.mnist.input_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x=tf.placeholder(tf.float32,[None,784])
# w=tf.Variable(tf.truncated_normal([784,10],stddev=1.0))
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,w) + b)
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
