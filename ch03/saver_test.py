import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize":[],"loss":[]}

def moving_average(a,w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx , val in enumerate(a)]

train_X = np.linspace(-1,1,100) #在a,b之间生成c个点
train_Y = 2*train_X+np.random.randn(*train_X.shape) * 0.3
# plt.plot(train_X,train_Y,'ro',label = 'Original data')
# plt.legend() #显示图例
# plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float") #真实值

W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.zeros([1]),name="bias")

z = tf.multiply(X,W) + b #预测值

cost = tf.reduce_mean(tf.square(Y-z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #梯度下降算法

init = tf.global_variables_initializer() #初始化所有变量
#定义参数
training_epochs = 20
display_step = 2

saver=tf.train.Saver()
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2,"log/linermodel.cpkt")
    print("x=0.2,z=",sess2.run(z,feed_dict={X:0.2}))