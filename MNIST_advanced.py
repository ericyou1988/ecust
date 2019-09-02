import tensorflow as tf
import input_data #可以自己下载这个py文件放在当前目录下
import os
import time
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #不加这个，macOS系统会有好几个警告，但是不影响程序运行

start_time = time.time()
print('开始时间: ',time.strftime("%Y-%m-%d  %H:%M:%S",time.localtime()))

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

batch_size = 100 #可以自己定义此值
n_batch = 55000//batch_size #train文件有55000张图片

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

def Weight_variable(shape):
    initial = tf.truncated_normal(shape=shape,mean=0,stddev=0.1)#这个叫截断的正太分布，只取正负两倍的标准差中间的值
    return tf.Variable(initial)

def biases_variable(shape):
    #给偏置一个正数是因为后面会用到relu神经元，防止relu神经元掉入0梯度区域坏死
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


#第一层卷积,32个5x5大小的卷积核

#卷积层的输入
x_image = tf.reshape(x,[-1,28,28,1])
#卷积层的参数
W_conv1 = Weight_variable(shape=[5,5,1,32]) #输入通道为1，输出通道为32，具体表现为‘图片’变厚了
b_conv1 = biases_variable(shape=[32])
#卷积层的输出
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
#卷积层后面连接一个池化层
h_pool1 = max_pooling(h_conv1)

#第二层卷积，只用了两个卷积核
W_conv2 = Weight_variable(shape=[5,5,32,64])
b_conv2 = biases_variable(shape=[64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pooling(h_conv2)
h_pool2_out = tf.reshape(h_pool2,shape=[-1,7*7*64])#reshape一下，方便后面使用

#全连接层
W_fullconnection1 = Weight_variable(shape=[7*7*64,1024])
b_fullconnection1 = biases_variable(shape=[1024])
h_fullconnection1 = tf.nn.relu(tf.matmul(h_pool2_out,W_fullconnection1)+b_fullconnection1)

#dropout，防止过拟合
keep_prop = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fullconnection1,keep_prop)

#全连接输出层
W_fullconnection2 = Weight_variable([1024,10])
b_fullconnection2 = biases_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fullconnection2)+b_fullconnection2)

cross_entropy = -tf.reduce_sum(y*tf.log(prediction)) #交叉熵损失函数
#这里用AdamOptimizer的结果和GradientDescentOptimizer差不多。见最后一张图。
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,axis=1),tf.argmax(prediction,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))#将bool值转化成数字

init = tf.global_variables_initializer()

saver = tf.train.Saver()#添加一个对象用来保存我们的模型

with tf.Session() as sess:
    sess.run(init)
    for step in range(1,30):  #迭代29轮
        for batch in range(n_batch):
            x_data,y_label = mnist.train.next_batch(batch_size) #每次训练100张图片
            sess.run(train_step,feed_dict={x:x_data,y:y_label,keep_prop:0.5})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prop:1.0})
        print('Iter'+ str(step) + ' ,Accuracy = ' + str(acc))
    saver.save(sess,"model.ckpt") #保存到当前目录
    print('保存成功')

end_time = time.time()
print('结束时间: ',time.strftime("%Y-%m-%d  %H:%M:%S",time.localtime()))
print('总耗时：',end_time - start_time,' 秒')
