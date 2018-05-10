import numpy as np
import numpy.random as npr
import cPickle
import matplotlib.pyplot as plt

class toyDataSet(object):
    
    def __init__(self,r_train = 1.5,r_test = 5.0, base_size=1000, batch_size=128):
	
	npr.seed(1234)
	assert r_train < r_test
	self.r_train=r_train
	self.r_test=r_test
	self.base_size=base_size
	self.batch_size=batch_size
	 
	self.mean_0 = np.array([-5, 0])
	self.cov_0 = np.array([[4, 0], [0, 4.]])

	self.mean_1 = np.array([5, 0])
	self.cov_1 = np.array([[4., 0], [0, 4.]])
 
	self.mean_2 = np.array([0, 5*np.sqrt(3.)])
	self.cov_2 = np.array([[4.0, 0], [0, 4.0]])

	self.x_0 = npr.multivariate_normal(mean=self.mean_0, cov=self.cov_0, size=self.base_size)
	self.x_1 = npr.multivariate_normal(mean=self.mean_1, cov=self.cov_1, size=self.base_size)
	self.x_2 = npr.multivariate_normal(mean=self.mean_2, cov=self.cov_2, size=self.base_size)

	self.x_0_train = self.x_0[np.sum((self.x_0 - self.mean_0) ** 2, 1) <= (self.r_train ** 2)]
	self.x_1_train = self.x_1[np.sum((self.x_1 - self.mean_1) ** 2, 1) <= (self.r_train ** 2)]
	self.x_2_train = self.x_2[np.sum((self.x_2 - self.mean_2) ** 2, 1) <= (self.r_train ** 2)]

	self.x_0_test = self.x_0[np.all((np.sum((self.x_0 - self.mean_0) ** 2, 1) > (self.r_train ** 2), np.sum((self.x_0 - self.mean_0) ** 2, 1) <= (self.r_test ** 2)), 0)]
	self.x_1_test = self.x_1[np.all((np.sum((self.x_1 - self.mean_1) ** 2, 1) > (self.r_train ** 2), np.sum((self.x_1 - self.mean_1) ** 2, 1) <= (self.r_test ** 2)), 0)]
	self.x_2_test = self.x_2[np.all((np.sum((self.x_2 - self.mean_2) ** 2, 1) > (self.r_train ** 2), np.sum((self.x_2 - self.mean_2) ** 2, 1) <= (self.r_test ** 2)), 0)]

	self.x_train = np.vstack((self.x_0_train, self.x_1_train, self.x_2_train))
	self.y_train = np.concatenate((0*np.ones(len(self.x_0_train), dtype=np.int),1*np.ones(len(self.x_1_train), dtype=np.int) ,2*np.ones(len(self.x_2_train), dtype=np.int) ),axis=0 )
	self.x_test = np.vstack((self.x_0_test, self.x_1_test, self.x_2_test))
	self.y_test = np.concatenate((0*np.ones(len(self.x_0_test), dtype=np.int),1*np.ones(len(self.x_1_test), dtype=np.int) ,2*np.ones(len(self.x_2_test), dtype=np.int) ),axis=0 )
	
	
	self._epochs_completed = 0
	self._index_in_epoch = 0
	self._num_train_examples = len(self.x_train)
	self._num_test_examples = len(self.x_test)
	
	perm = np.arange(self._num_train_examples)
	np.random.shuffle(perm)
	self.x_train = self.x_train[perm]
	self.y_train = self.y_train[perm]
	
	perm = np.arange(self._num_test_examples)
	np.random.shuffle(perm)
	self.x_test = self.x_test[perm]
	self.y_test = self.y_test[perm]
    
    def next_batch(self):
	start = self._index_in_epoch
	self._index_in_epoch += self.batch_size
	if self._index_in_epoch > self._num_train_examples:
	    # Finished epoch
	    self._epochs_completed += 1
	    # Shuffle the data
	    perm = np.arange(self._num_train_examples)
	    np.random.shuffle(perm)
	    self.x_train = self.x_train[perm]
	    self.y_train = self.y_train[perm]
	    # Start next epoch
	    start = 0
	    self._index_in_epoch = self.batch_size
	    assert self.batch_size <= self._num_train_examples
	end = self._index_in_epoch
	return self.x_train[start:end], self.y_train[start:end]
	
	
if __name__=='__main__':

    datapoints = toyDataSet()

    plt.scatter(datapoints.x_train[datapoints.y_train==0,0], datapoints.x_train[datapoints.y_train==0,1], c='b', marker='o')
    plt.scatter(datapoints.x_train[datapoints.y_train==1,0], datapoints.x_train[datapoints.y_train==1,1], c='g', marker='o')
    plt.scatter(datapoints.x_train[datapoints.y_train==2,0], datapoints.x_train[datapoints.y_train==2,1], c='r', marker='o')
    plt.scatter(datapoints.x_test[datapoints.y_test==0,0], datapoints.x_test[datapoints.y_test==0,1], c='b', marker='x')
    plt.scatter(datapoints.x_test[datapoints.y_test==1,0], datapoints.x_test[datapoints.y_test==1,1], c='g', marker='x')
    plt.scatter(datapoints.x_test[datapoints.y_test==2,0], datapoints.x_test[datapoints.y_test==2,1], c='r', marker='x')
    plt.show()
    
    # testing next_batch() method
    points, labels  = datapoints.next_batch()
    plt.scatter(points[labels==0,0], points[labels==0,1], c='b', marker='o')
    plt.scatter(points[labels==1,0], points[labels==1,1], c='g', marker='o')
    plt.scatter(points[labels==2,0], points[labels==2,1], c='r', marker='o')
    plt.show()

    
