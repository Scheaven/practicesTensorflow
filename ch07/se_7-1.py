import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#生成数据，指定生成数据的大小，均值，方差，
def generate(sample_size, mean, cov, diff,regression):
    num_classes = 2
    sample_per_class = int(sample_size/2)

    X0 = np.random.multivariate_normal(mean,cov,sample_per_class) #获取正太分布的数据
    Y0 = np.zeros(sample_per_class)

    for ci , d in enumerate(diff): #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        X1 = np.random.multivariate_normal(mean+d,cov,sample_per_class)
        Y1 = (ci+1)*np.ones(sample_per_class)

        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))

    if regression==False:
        class_ind = [Y==class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind),dtype=np.float32)

    X,Y = shuffle(X0,Y0)

    return X,Y

input_dim = 2
np.random.seed(10)
num_classes =2
mean = np.random.randn(num_classes)
cov = np.eye(num_classes)
X,Y = generate(1000,mean,cov,[3.0],True)
colors = ['r' if l==0 else 'b' for l in Y[:]]
plt.scatter(X[:,0],X[:,1],c = colors)
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumer size (in cm)")
plt.show()
lab_dim=1

input_features = tf.placeholder(tf.float32,[None,input_dim])
input_lables = tf.placeholder(tf.float32,[None,lab_dim])

W = tf.Variable(tf.random_normal([input_dim,lab_dim]),name="weight")
b = tf.Variable(tf.zeros([lab_dim]),name="bias")

output = tf.nn.sigmoid(tf.matmul(input_features,W)+b)
cross_entropy = -(input_lables * tf.log(output) + (1 - input_lables)*tf.log(1-output))
ser = tf.square(input_lables - output)
loss = tf.reduce_mean(cross_entropy)
err = tf.reduce_mean(ser)
optimizer = tf.train.AdamOptimizer(0.04)
train = optimizer.minimize(loss)

maxEpochs = 50
minibatchSize = 25

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch in range(maxEpochs):
        sumerr =0
        for i in range(np.int32(len(Y)/minibatchSize)):
            x1 = X[i*minibatchSize:(i+1)*minibatchSize]
            y1 = np.reshape(Y[i*minibatchSize:(i+1)*minibatchSize],[-1,1])
            tf.reshape(y1,[-1,1])
            _,lossval,outputval,errval = sess.run([train,loss,output,err],feed_dict={input_features:x1,input_lables:y1})
            sumerr = sumerr + errval
        print("Epoch:",'%04d' % (epoch+1),"cost=","{:09f}".format(lossval),"err=",sumerr/minibatchSize)

    train_X,train_Y = generate(100,mean,cov,[3.0],True)
    colors = ['r' if l == 0 else 'b' for l in train_Y[:]]
    plt.scatter(train_X[:,0],train_X[:,1], c = colors)
    x = np.linspace(-1,8,200)
    y = -x*(sess.run(W)[0]/sess.run(W)[1])-sess.run(b)/sess.run(W)[1]
    plt.plot(x,y,label='Fitted line')
    plt.legend()
    plt.show()