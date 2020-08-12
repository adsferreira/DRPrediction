from sklearn.model_selection import ShuffleSplit
import numpy as np
X = np.arange(5)
ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=None)

for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))