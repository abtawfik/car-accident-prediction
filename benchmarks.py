####################
# Standard library #
####################

######################
# 3rd party packages #
######################
import numpy as np

#####################
# Internal Specific #
#####################



class AlwaysMean(object):
    '''A simple always guess mean model to test other models against as a benchmark
    '''
    def fit(self, X, y):
        self.mean = y.mean()
        return self
    def predict(self, df):
        return np.array( [self.mean]*df.shape[0] )

