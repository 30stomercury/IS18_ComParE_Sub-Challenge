# selfAssessed
import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

# load data: eGmaps_lld selfAssesed
def load_data(feature_path='dataset/'):
    '''
    shape: [num_data, sequence_length, dimension]
    '''
    #X_train = joblib.load(feature_path+'X_train.pkl')
    #X_devel = joblib.load(feature_path+'X_devel.pkl')
    X_test = np.load(feature_path+'X_test.npy')
    X_train = np.load(feature_path+'X_train.npy')
    X_devel = np.load(feature_path+'X_devel.npy')
    # define label
    df = pd.read_csv(feature_path+'ComParE2018_SelfAssessedAffect.csv')
    y_train = df['Valence'][:846].values
    y_devel = df['Valence'][846:].values
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_devel = le.transform(y_devel)
    y_train = y_train.reshape(len(y_train), 1)
    y_devel = y_devel.reshape(len(y_devel), 1)
    # convert into one-hot label
    enc = preprocessing.OneHotEncoder(sparse=False)
    y_train = enc.fit_transform(y_train)
    y_devel = enc.transform(y_devel)
    return X_train, X_devel, y_train, y_devel

class BatchGenerator:
    def __init__(self, X, y, hparams):
        self.hps = hparams
        D = self.hps.input_dim
        long_ = self.hps.seq_length
        batch_size = self.hps.BATCH_SIZE
        n = len(X)
        self.batch_xs, self.batch_ys = [], []
        for i in range(n//batch_size+1):
            if i != (n//batch_size):
                batch_x = np.zeros((batch_size, long_, D))
                batch_y = np.zeros((batch_size, 3))
                for j in range(batch_size):
                    words = X[i*batch_size+j]
                    for k in range(len(words)):
                        batch_x[j][k] = words[k]
                        #print(k)
                    for k in range(k, long_):
                        batch_x[j][k] = np.zeros(D) # padding with 0
                    batch_y[j] = y[i*batch_size+j] # 1-hot vector
            else:
                batch_x = np.zeros((len(X) % batch_size, long_, D))
                batch_y = np.zeros((len(y) % batch_size, 3))
                for j in range((len(y) % batch_size)):
                    words = X[i*batch_size+j]
                    for k in range(len(words)):
                        batch_x[j][k] = words[k]
                    for k in range(k, long_):
                        batch_x[j][k] = np.zeros(D) # padding with 0

                    batch_y[j] = y[i*batch_size+j] # 1-hot vector
            self.batch_xs.append(batch_x)
            self.batch_ys.append(batch_y)

    def get(self, batch_id):
        return self.batch_xs[batch_id], self.batch_ys[batch_id]
