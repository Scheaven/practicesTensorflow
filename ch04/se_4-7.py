import tensorflow as tf
tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step() #保存步数
step =tf.assign_add(global_step,1)
with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpoints',save_checkpoint_secs=2) as sess:
    print(sess.run([global_step]))
    while not sess.should_stop(): #如果线程应该停止则返回True。
        i=sess.run(step)
        print(i)