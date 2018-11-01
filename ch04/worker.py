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
#
# X = tf.placeholder("float")
# Y = tf.placeholder("float") #真实值
#
# W = tf.Variable(tf.random_normal([1]),name="weight")
# b = tf.Variable(tf.zeros([1]),name="bias")
#
# z = tf.multiply(X,W) + b #预测值
# tf.summary.histogram('z',z) #将预测值以直方图的形式显示
#
# #反向优化
# cost = tf.reduce_mean(tf.square(Y-z))
# tf.summary.scalar('loss_function',cost) #将损失以标量的形式显示
# learning_rate = 0.01
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #梯度下降算法
#
# init = tf.global_variables_initializer() #初始化所有变量
# #定义参数
# training_epochs = 20
# display_step = 2
# saver = tf.train.Saver()
# saverdir= "log/"

# #启动 session
# with tf.Session() as sess:
#     sess.run(init)
#
#     merged_summary_op = tf.summary.merge_all() #合并所有summary
#     #创建summary_writer,用于写文件
#     summary_writer = tf.summary.FileWriter('log/mnist_with_summaries',sess.graph)
#
#     #向模型中传入参数
#     for epoch in range(training_epochs):
#         for (x,y) in zip(train_X,train_Y):
#             sess.run(optimizer,feed_dict={X:x,Y:y})
#
#         summary_str = sess.run(merged_summary_op,feed_dict={X:x,Y:y})
#         summary_writer.add_summary(summary_str,epoch);#将summary写入文件
#
#     plotdata = {"batchsize":[],"loss":[]} #用于存放批次值以及损失值
#     #向模型输入数据
#     for epoch in range(training_epochs):
#         for (x,y) in zip(train_X,train_Y):
#             sess.run(optimizer,feed_dict={X:x,Y:y})
#
#         #显示训练中的详细信息
#         if epoch % display_step == 0:
#             loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
#             print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
#             if not (loss=="NA"):
#                 plotdata["batchsize"].append(epoch)
#                 plotdata["loss"].append(loss)
#
#     print("Finished!")
#     saver.save(sess,saverdir+"linermodel.cpkt")
#     print("cost=", sess.run(cost,feed_dict={X:train_X,Y:train_Y}), "W=", sess.run(W), "b=", sess.run(b))
#
#
#     plt.plot(train_X,train_Y,'ro',label = 'Original data')
#     plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label = 'Fittedline')
#
#     plt.legend()
#     plt.show()
#
#     plotdata["avgloss"] = moving_average(plotdata["loss"])
#     plt.figure(1)
#     plt.subplot(211)
#     plt.plot(plotdata["batchsize"],plotdata["avgloss"],'b--')
#     plt.xlabel('Minibatch number')
#     plt.ylabel('Loss')
#     plt.title('Minibatch run vs. Training loss')
#
#     plt.show()
#
#     print("x=0.2,z=", sess.run(z, feed_dict={X: 0.2}))
#
strps_hosts = "localhost:1681"
strworker_hosts="localhost:1682,localhost:1683"

strjob_name = "worker"
task_index = 0

ps_hosts = strps_hosts.split(",")
worker_hosts = strworker_hosts.split(',')
cluster_spec = tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts}) #创建集群

server = tf.train.Server({'ps':ps_hosts,'worker':worker_hosts},job_name=strjob_name,task_index=task_index)

if strjob_name == 'ps':
    print("wait")
    server.join()

with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index,cluster=cluster_spec)):
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    W = tf.Variable(tf.random_normal([1]),name="weight")
    b = tf.Variable(tf.zeros([1]),name="bias")

    global_step = tf.train.get_or_create_global_step() #获得迭代次数

    z = tf.multiply(X,W)+b
    tf.summary.histogram('z',z) #将预测值以直方图的形式显示

    #反向优化
    cost = tf.reduce_mean(tf.square(Y - z))
    tf.summary.scalar('loss_function',cost)
    learning_rate = 0.01

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)

    saver = tf.train.Saver(max_to_keep=1)
    merged_summary_op = tf.summary.merge_all() #合并所有的summary

    init = tf.initialize_all_variables()

training_epochs = 2200
display_step = 2

sv = tf.train.Supervisor(is_chief=(task_index == 0),
                         logdir="log/super/",
                         init_op=init,
                         summary_op=None,
                         saver=saver,
                         global_step=global_step,
                         save_model_secs=5)

with sv.managed_session(server.target) as sess:
    print("sess ok")
    print(global_step.eval(session=sess))
    for epoch in range(global_step.eval(session=sess),training_epochs*len(train_X)):
        for (x,y) in zip(train_X,train_Y):
            _,epoch = sess.run([optimizer,global_step],feed_dict={X:x,Y:y})
            summary_str = sess.run(merged_summary_op,feed_dict={X:x,Y:y})
            #将summary写入文件
            sv.summary_computed(sess,summary_str,global_step=epoch)
            if epoch % display_step == 0:
                loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
                print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
                if not (loss == "NA"):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)

    print("Finished!")
    sv.saver.save(sess,"log/minist_with_summaries/"+"sv.cpk",global_step=epoch)

sv.stop()