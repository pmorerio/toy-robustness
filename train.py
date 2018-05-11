from dataset import toyDataSet
import numpy as np
import tensorflow as tf
import  tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt


def plot_decision_boundary(pred_func, X, y, counter, save=True):
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
    if save:
        #~ fig = plt.figure()
        plt.savefig(str(counter), bbox_inches='tight')
        plt.close() # avoid memory leak
        

data_handler = toyDataSet()
sess = tf.InteractiveSession()
    #(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    #config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)))

# Model
num_classes = 3
l_rate = 0.001
_x = tf.placeholder(tf.float32, shape=[None, 2]) 
_y = tf.placeholder(tf.int64, shape=[None])

with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu, 
                    weights_initializer=tf.contrib.layers.xavier_initializer(), 
                    biases_initializer = tf.constant_initializer(0.001)): 
    _h1 = slim.fully_connected(_x, 5, scope='hidden1')
    _h2 = slim.fully_connected(_h1, 10, scope='hidden2')
    #~ _h3 = slim.fully_connected(_h2, 2, scope='hidden3')
    _h3 = slim.fully_connected(_h2, 2, activation_fn=None, scope='hidden3')
_logits = slim.fully_connected(_h3, num_classes, activation_fn=None,scope='logits')
_loss = tf.losses.sparse_softmax_cross_entropy(labels=_y, logits=_logits)
_train_step = tf.train.AdamOptimizer(l_rate).minimize(_loss)
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

prediction_func = lambda x:sess.run(_pred,feed_dict={_x:x})
prediction_func2 = lambda x:sess.run(_pred2,feed_dict={_h3_in:x})

for i in range(20000):
    
    points, labels = data_handler.next_batch()
    feed_dict = {_x:points, _y:labels}
    sess.run(_train_step, feed_dict=feed_dict)
    
    if i %10 == 0 or i == 0:
        xentr, acc = sess.run([_loss, _accuracy], feed_dict)
        print('Iter', i, 'Loss:', str(xentr), 'Acc:', str(acc))
        
        
        #~ plot_decision_boundary(prediction_func,data_handler.x_train, data_handler.y_train,0 )
        plot_decision_boundary(prediction_func,data_handler.x_test, data_handler.y_test, i )

        #~ h3 = sess.run(_h3, feed_dict={_x:data_handler.x_train})
        #~ plot_decision_boundary(prediction_func2,h3, data_handler.y_train,2 )
        #~ h3 = sess.run(_h3, feed_dict={_x:data_handler.x_test})
        #~ plot_decision_boundary(prediction_func2,h3, data_handler.y_test,3 )
        

        acc_test = sess.run(_accuracy, feed_dict={_x:data_handler.x_test, _y:data_handler.y_test})
        print('Test Acc:', str(acc_test))




