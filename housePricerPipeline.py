from sklearn.model_selection import train_test_split 
import sklearn.ensemble
import numpy as np,scipy as sp
from sklearn.datasets import load_boston
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error

## Set ML - (training, validation, test ) - Sets ratio - splits

VALIDATION_PERCENTAGE_SPLIT = 0.2    
TEST_PERCENTAGE_SPLIT       = 0.2
TRAINING_PERCENTAGE_SPLIT   = 1.0 - ( VALIDATION_PERCENTAGE_SPLIT + TEST_PERCENTAGE_SPLIT)

## These constants are fixed to the boston dataset , which is known in advance , but can be a variable number that is read in at runtime (cmd line param or config file etc.) \ 
#  just as easily - for genralization to other datasets going forward
BOSTON_NUM_SAMPLES  = 506
BOSTON_NUM_FEATURES = 13



class ml_pipeline:

    def __init__(self):
        self.num_samples                 = BOSTON_NUM_SAMPLES
        self.feature_vec_length          = BOSTON_NUM_FEATURES
        self.feature_vector              = np.empty(self.feature_vec_length)
        self.feature_matrix              = np.zeros((self.num_samples, self.feature_vec_length))
        self.input_raw_data_mat          = np.zeros((BOSTON_NUM_SAMPLES, BOSTON_NUM_FEATURES))
        self.stdized_input_raw_data_mat  = np.empty(shape=(1,1))

    def readInput(self):

        print('in readInput() member function')

        input , output =  sklearn.datasets.load_boston(return_X_y=True)
        print('input shape of bostin housing data = ' + str(input.shape))

        self.input_raw_data_mat = input
        self.output             = output

    def preprocessData(self):

        print('in clean_data() member function')

        # calc max value in the dataset and scale - minimax
        Max = float(np.max(self.input_raw_data_mat))
        Min = float(np.min(self.input_raw_data_mat))

        if (Max != Min) :
            self.stdized_input_raw_data_mat = self.input_raw_data_mat/(Max - Min)
        else:
            self.stdized_input_raw_data_mat = self.input_raw_data_mat/ Max


    def trainModel(self ): #, input_file=self.stdized_input_raw_data_mat):   

        # split the data randomly (but in a reproducible way[same random seed] via separation along the indices) into train, validate and test sets
        # idx                = np.arange(self.stdized_input_raw_data_mat.shape[0])
        self.Xtrain,self.Xtest,self.Ytrain,self.Ytest = train_test_split(self.stdized_input_raw_data_mat,self.output,test_size=TEST_PERCENTAGE_SPLIT,random_state=42)
        self.Xtrain,self.Xval,self.Ytrain,self.Yval  = train_test_split(self.Xtrain,self.Ytrain,test_size=VALIDATION_PERCENTAGE_SPLIT,random_state=42)

    def train_and_writeOutput(self):  # features_data_file):

        print('in predict_and_writeOutput() ml_pipeline class member function')

        print("Fitting model...")
        params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 2,
                'learning_rate': 0.01, 'loss': 'ls'}
        model    = sklearn.ensemble.GradientBoostingRegressor(**params)

        model.fit(self.Xtrain, self.Ytrain)
        train_mse = mean_squared_error(self.Ytrain, model.predict(self.Xtrain))
        print("training mean squared error is : ",train_mse)
        test_mse  = mean_squared_error(self.Ytest, model.predict(self.Xtest))
        print("testing mean squared error is : ",test_mse)
        filename = 'finalized_model.pkl'
        pickle.dump(model, open(filename, 'wb'))

if __name__ == '__main__' :

    mlpipe  = ml_pipeline()
    mlpipe.readInput()
    mlpipe.preprocessData()
    mlpipe.trainModel()
    mlpipe.train_and_writeOutput()