import numpy as np
from dataset import toyDataSet
from scipy.spatial.distance import cdist

data_handler = toyDataSet()

correct_predictions=0
i=0
k=5 #nearest neighbour

for x, y in zip(data_handler.x_test, data_handler.y_test):
    print str(i)+'\r',
    dist = cdist(data_handler.x_train, np.expand_dims(x, axis=0), 'cosine')
    rank = np.argsort(dist.ravel())
    y=np.asarray([y for i in range(k)])
    pred = y == data_handler.y_train[rank[:k]]
    #print pred
    if np.sum(pred) > k/2:
        correct_predictions += 1
    i+=1

print 'Accuracy = ' + str(float(correct_predictions)/float(len(data_handler.y_test) ))
