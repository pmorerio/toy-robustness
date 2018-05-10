from dataset import toyDataSet
import numpy as np
import tensorflow as tf
import  tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt


def plot_decision_boundary(pred_func, X, y, counter):
    plt.figure(counter)
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.02
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.stack([xx.ravel(), yy.ravel()], axis=-1))
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, marker='x' )
    #~ plt.show()


data_handler = toyDataSet()
sess = tf.InteractiveSession()
    #(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    #config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)))

# Model
num_classes = 3
l_rate_min = 0.001
l_rate_max = 0.01

_x = tf.placeholder(tf.float32, shape=[None, 2]) 
_y = tf.placeholder(tf.int64, shape=[None])


_h1 = slim.fully_connected(_x, 5, activation_fn=tf.nn.relu,  scope='hidden1')
_h2 = slim.fully_connected(_h1, 10, activation_fn=tf.nn.relu, scope='hidden2')
_h3 = slim.fully_connected(_h2, 2, activation_fn=tf.nn.relu, scope='hidden3')
_logits = slim.fully_connected(_h3, num_classes, activation_fn=None,scope='logits')


_x_hat = tf.get_variable('x_hat', [data_handler.batch_size, 2]) # 128 is the size of the dataset, should be a variable! 
_x_hat_assign_op = _x_hat.assign(_x) # to assign a value to the variable

_h1_hat = slim.fully_connected(_x_hat, 5, activation_fn=tf.nn.relu,  scope='hidden1', reuse=True)
_h2_hat = slim.fully_connected(_h1_hat, 10, activation_fn=tf.nn.relu, scope='hidden2', reuse=True)
_h3_hat = slim.fully_connected(_h2_hat, 2, activation_fn=tf.nn.relu, scope='hidden3', reuse=True)
_logits_hat = slim.fully_connected(_h3_hat, num_classes, activation_fn=None,scope='logits', reuse=True)

_t_vars = tf.trainable_variables()
_min_vars = [var for var in _t_vars if 'hat' not in var.name]
_max_vars = [var for var in _t_vars if 'hat' in var.name]

_min_loss = tf.losses.sparse_softmax_cross_entropy(labels=_y, logits=_logits)
_max_loss = tf.losses.sparse_softmax_cross_entropy(labels=_y, logits=_logits_hat)

_min_optimizer = tf.train.AdamOptimizer(l_rate_min) 
_max_optimizer = tf.train.GradientDescentOptimizer(l_rate_max) 
_min_train_op = slim.learning.create_train_op(_min_loss, _min_optimizer, variables_to_train = _min_vars)
_max_train_op = slim.learning.create_train_op(-_max_loss, _max_optimizer, variables_to_train = _max_vars) # max == min with minus
	


# EVALUATE THE MODEL
_pred = tf.argmax(_logits, 1)
_correct_prediction = tf.equal(_pred, _y)
_accuracy = tf.reduce_mean(tf.cast(_correct_prediction, tf.float32))

############## just to eval features
_h3_in = tf.placeholder(tf.float32, shape=[None, 2]) 
_logits2 = slim.fully_connected(_h3_in, num_classes, activation_fn=None,scope='logits', reuse=True)
_pred2 = tf.argmax(_logits2, 1)
################

tf.global_variables_initializer().run()

T_adv = 5 # number of SGA steps before running a SGD step

for i in range(20000):
    
    points, labels = data_handler.next_batch()
    feed_dict = {_x:points, _y:labels}
    
    sess.run(_x_hat_assign_op, feed_dict=feed_dict)
        
    for t in range(T_adv):
	_, max_l = sess.run([_max_train_op,_max_loss], feed_dict=feed_dict)
	#~ print max_l
		
    feed_dict[_x] = sess.run(_x_hat, feed_dict=feed_dict)
        
    sess.run(_min_train_op, feed_dict=feed_dict)
    
    if i%100==0:
	xentr, acc = sess.run([_min_loss, _accuracy], feed_dict)
	print('Iter', i, 'Loss:', str(xentr), 'Acc:', str(acc))
    

acc_test = sess.run(_accuracy, feed_dict={_x:data_handler.x_test, _y:data_handler.y_test})
print('Test Acc:', str(acc_test))

prediction_func = lambda x:sess.run(_pred,feed_dict={_x:x})
plot_decision_boundary(prediction_func,data_handler.x_train, data_handler.y_train,0 )
plot_decision_boundary(prediction_func,data_handler.x_test, data_handler.y_test,1 )

prediction_func2 = lambda x:sess.run(_pred2,feed_dict={_h3_in:x})
h3 = sess.run(_h3, feed_dict={_x:data_handler.x_train})
plot_decision_boundary(prediction_func2,h3, data_handler.y_train,2 )
h3 = sess.run(_h3, feed_dict={_x:data_handler.x_test})
plot_decision_boundary(prediction_func2,h3, data_handler.y_test,3 )

plt.show()
