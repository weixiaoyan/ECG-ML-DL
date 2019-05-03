# -*- coding: utf-8 -*-
"""
Created on

@author: weixiaoyan

===================基于1维CNN的ECG分类算法========================

*需要第三方工具包numpy,h5py,scikit-learn
*基于深度学习框架TensorFlow
*涉及函数的使用方法可自行查看工具包文档，baidu即可得
*ECG算法入门系列博客：https://blog.csdn.net/qq_15746879
*开源github：https://github.com/Aiwiscal

*本代码所需要的数据和标签文件来自matlab提取
*详情：https://blog.csdn.net/qq_15746879/article/details/80340958
==================================================================
"""
#载入所需工具包
import time
import numpy as np
import h5py as hp
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import tensorboard

sess=tf.InteractiveSession()

#载入.mat文件的函数,h5py解码并转换为numpy数组
def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr

#使用TensorFlow组件完成CNN网络的搭建，与教程中参数略有不同
def CNNnet(inputs,n_class):
    conv1 = tf.layers.conv1d(inputs=inputs, filters=4, kernel_size=3, strides=1, \
                             padding='same', activation = tf.nn.relu)
    avg_pool_1 = tf.layers.average_pooling1d(inputs=conv1, pool_size=5, strides=5, \
                                         padding='same')
    conv2 = tf.layers.conv1d(inputs=avg_pool_1, filters=8, kernel_size=6, strides=1,\
                             padding='same', activation = tf.nn.relu)
    avg_pool_2 = tf.layers.average_pooling1d(inputs=conv2, pool_size=5, strides=5,\
                                         padding='same')
    
    flat = tf.reshape(avg_pool_2, (-1, int(250/5/5*8)))
    
    logits=tf.layers.dense(inputs=flat, units=n_class, activation=None)
    logits=tf.nn.softmax(logits)
    return logits

#随机获取一个batch大小的数据，用于训练
def get_batch(train_x,train_y,batch_size):
    indices=np.random.choice(train_x.shape[0],batch_size,False)
    batch_x=train_x[indices]
    batch_y=train_y[indices]
    return batch_x,batch_y

#设定路径及文件名并载入，这里的心拍在Matlab下截取完成
#详情：https://blog.csdn.net/qq_15746879/article/details/80340671
Path='J:/20190428-ECG/ECG-ML-DL-Algorithm-python/ECG-ML-DL-Algorithm-Python-master/'  #自定义路径要正确
DataFile='Data_CNN.mat'
LabelFile='Label_OneHot.mat'

print("Loading data and labels...")
tic=time.time()
Data=load_mat(Path+DataFile,'Data')
Label=load_mat(Path+LabelFile,'Label')
Data=Data.T
Indices=np.arange(Data.shape[0]) #随机打乱索引并切分训练集与测试集
np.random.shuffle(Indices)

print("Divide training and testing set...")
train_x=Data[Indices[:10000]]
train_y=Label[Indices[:10000]]
test_x=Data[Indices[10000:]]
test_y=Label[Indices[10000:]]
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

print("1D-CNN setup and initialize...")
tic=time.time()
x=tf.placeholder(tf.float32, [None, 250]) #定义placeholder数据入口
x_=tf.reshape(x,[-1,250,1])
y_=tf.placeholder(tf.float32,[None,4])

logits=CNNnet(x_,4)

learning_rate=0.01
batch_size=16
maxiters=15000

#cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_))
#这里使用了自适应学习率的Adam训练方法，可以认为是SGD的高级演化版本之一
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    # 这里使用了自适应学习率的Adam训练方法，可以认为是SGD的高级演化版本之一
    tf.summary.scalar('cost', cost)
with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
# important step
sess.run(tf.initialize_all_variables())
#tf.global_variables_initializer().run()

toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

print("1D-CNN training and testing...")
tic=time.time()
for i in range(maxiters):
    batch_x,batch_y=get_batch(train_x,train_y,batch_size)
    sess.run(train_step,feed_dict={x:batch_x,y_:batch_y})
    if i%500==0:
        loss=sess.run(merged,feed_dict={x:train_x,y_:train_y})
        writer.add_summary(loss,i)

#y_pred=logits.eval(feed_dict={x:test_x,y_:test_y})
#y_pred=np.argmax(y_pred,axis=1)
#y_true=np.argmax(test_y,axis=1)
#toc=time.time()
#print("Elapsed time is %f sec."%(toc-tic))

#Acc=np.mean(y_pred==y_true)
#Conf_Mat=confusion_matrix(y_true,y_pred) #利用专用函数得到混淆矩阵
#Acc_N=Conf_Mat[0][0]/np.sum(Conf_Mat[0])
#Acc_V=Conf_Mat[1][1]/np.sum(Conf_Mat[1])
#Acc_R=Conf_Mat[2][2]/np.sum(Conf_Mat[2])
#Acc_L=Conf_Mat[3][3]/np.sum(Conf_Mat[3])
