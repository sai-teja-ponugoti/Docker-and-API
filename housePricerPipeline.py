
#import luigi
import sklearn 
from sklearn import utils
from sklearn.model_selection import train_test_split 
from sklearn.metrics import average_precision_score,classification_report 
import sklearn.ensemble
#import datetime,
import os,sys
import unittest
#import keras
import numpy as np,scipy as sp
from sklearn.datasets import load_boston
import pickle
#   import full_all_bookies_allLeagues_match_data

## Set a default file to store the input read in from the bodton housing data set via sci-kit learn  in 
curr_dir = os.getcwd()
in_file = curr_dir + '/' + 'input_file.txt'
if not os.path.isfile(  in_file ):
    print('copy nul ' + str(in_file))
    os.system('copy nul "' + in_file + '"')

clean_in_data_file = curr_dir + '/' + 'clean_input_data_file.txt'
if not os.path.isfile(  clean_in_data_file ):
    os.system('copy nul ' + clean_in_data_file)

features_data_file = curr_dir + '/' + 'featurized_data_file.txt'
if not os.path.isfile(  features_data_file ):
    os.system('copy nul ' + features_data_file)

# same for the final outpu file which will store the results in a .csv format
out_file = curr_dir + '/' + 'output_file.csv'
if not os.path.isfile(  out_file ):
    os.system('copy nul ' + out_file)

## Set ML - (training, validation, test ) - Sets ratio - splits

VALIDATION_PERCENTAGE_SPLIT = 0.2    
TEST_PERCENTAGE_SPLIT       = 0.2
TRAINING_PERCENTAGE_SPLIT   = 1.0 - ( VALIDATION_PERCENTAGE_SPLIT + TEST_PERCENTAGE_SPLIT)

## These constants are fixed to the boston dataset , which is known in advance , but can be a variable number that is read in at runtime (cmd line param or config file etc.) \ 
#  just as easily - for genralization to other datasets going forward
BOSTON_NUM_SAMPLES  = 506
BOSTON_NUM_FEATURES = 13


# Define the overall modell - pipeline Class  which will have all the internal variables, data structures and most importantly eacvh stage of the pipeline :
## from MOdel raw data input reading, through data cleaning and standardization and feautre extraction finally ->  output prdictions.


class ml_pipeline:

    def __init__(self):

        self.num_samples                 = BOSTON_NUM_SAMPLES
        self.input_file                  = in_file
        self.clean_input_data_file       = clean_in_data_file
        self.featurized_data_file        = features_data_file
        self.output_file                 = out_file
        self.feature_vec_length          = BOSTON_NUM_FEATURES
        self.feature_vector              = np.empty(self.feature_vec_length)
        self.feature_matrix              = np.zeros((self.num_samples, self.feature_vec_length))
        self.input_raw_data_mat          = np.zeros((BOSTON_NUM_SAMPLES, BOSTON_NUM_FEATURES))
        self.stdized_input_raw_data_mat  = np.empty(shape=(1,1))

        
        self.Xtrain = np.empty(shape=(1,1))
        self.Ytrain = np.empty(shape=(1,1))
        self.Xval   = np.empty(shape=(1,1))
        self.Yval   = np.empty(shape=(1,1))
        self.Xtest  = np.empty(shape=(1,1))
        self.Ytest  = np.empty(shape=(1,1))


    def readInput(self, data_source = in_file):

        print('in readInput() member function')
        #read in via sklearn boston data set

        input_ , output =  sklearn.datasets.load_boston(return_X_y=True)
        print('input shape of bostin housing data = ' + str(input_.shape))

        self.input_raw_data_mat = input_
        self.output             = output

        # store the raw data - each intermediate step should be saved if needed in future as stages can change and step should be trackable, also may be used rather than features for a future model
        try:
            with open(self.input_file, "wb") as File:
                ret_pickle_dump = pickle._dump( self.input_raw_data_mat,File , protocol=pickle.HIGHEST_PROTOCOL )

        except NameError:
            print('Error in Pickling input raw data, continuing regardless but must debug and fix ...')
            pass    

        # write to .txt inuot file created above.
        return self.input_file


    def preprocessData(self, input_file='./input_file.txt'):

        print('in clean_data() member function')

        # calc max value in the dataset and scale - minimax
        Max = float(np.max(self.input_raw_data_mat))
        Min = float(np.min(self.input_raw_data_mat))

        if (Max != Min) :
            self.stdized_input_raw_data_mat = self.input_raw_data_mat/(Max - Min)
        else:
            self.stdized_input_raw_data_mat = self.input_raw_data_mat/ Max

        return self.stdized_input_raw_data_mat

    def trainModel(self ): #, input_file=self.stdized_input_raw_data_mat):   

        # split the data randomly (but in a reproducible way[same random seed] via separation along the indices) into train, validate and test sets
        idx                = np.arange(self.stdized_input_raw_data_mat.shape[0])
        idx_train,idx_test = train_test_split(idx,test_size=TEST_PERCENTAGE_SPLIT,random_state=42)
        idx_train,idx_val  = train_test_split(idx_train,test_size=VALIDATION_PERCENTAGE_SPLIT,random_state=42)

        self.Xtrain.resize((len(idx_train), self.input_raw_data_mat.shape[1]),refcheck=False)
        self.Ytrain.resize((len(idx_train), self.input_raw_data_mat.shape[1]),refcheck=False)
        self.Xval.resize((len(idx_val),     self.input_raw_data_mat.shape[1]),refcheck=False)
        self.Yval.resize((len(idx_val),     self.input_raw_data_mat.shape[1]),refcheck=False)
        self.Xtest.resize((len(idx_test),   self.input_raw_data_mat.shape[1]),refcheck=False)
        self.Ytest.resize((len(idx_test),   self.input_raw_data_mat.shape[1]),refcheck=False)


        self.Xtrain = self.input_raw_data_mat[idx_train,:]
        self.Ytrain = self.output[idx_train]
        self.Xval   = self.input_raw_data_mat[idx_val,:]
        self.Yval   = self.output[idx_val]
        self.Xtest  = self.input_raw_data_mat[idx_test,:]
        self.Ytest  = self.output[idx_test] 

        #max_Xtrain = 1.0 #np.max(Xtrain)
        #,max_Xval,max_Xtest =  np.max(Xtrain),np.max(Xval),np.max(Xtest)  # 1.0,1.0,1.0
            
        ## pertform max-min scaling of the data    
        # if (Max != Min) :
        #     self.Xtrain,self.Xval,self.Xtest  = self.Xtrain / (Max - Min) , self.Xval / (Max - Min) , self.Xtest / (Max - Min) 
        # else:
        #     self.Xtrain,self.Xval,self.Xtest  = self.Xtrain / Max  , self.Xval / Max , self.Xtest / Max
        # remove large outliers and replace with means

        return self.Xtrain,self.Xval,self.Xtest


    # def featureizeData(self, input = clean_in_data_file):

    #     print('in featureizeData() ml_pipeline class member function')

    #     return self.featurized_data_file


    def train_and_writeOutput(self, input_data = clean_in_data_file ):  # features_data_file):

        print('in predict_and_writeOutput() ml_pipeline class member function')

        print("Fitting model...")
        params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 2,
                'learning_rate': 0.01, 'loss': 'ls'}
        model    = sklearn.ensemble.GradientBoostingRegressor(**params)

        classifier = model.fit(self.Xtrain, self.Ytrain)
        train_mse = mean_squared_error(self.y_train, model.predict(Xtrain))
        test_mse  = mean_squared_error(self.y_test, model.predict(Xtest))
        metadata  = {
            "train_mean_square_error": train_mse,
            "test_mean_square_error": test_mse  
        }
        # print('results of model on test set = ' + str(metadata))

        #pa ss
        #result = {}
        #result['pre']

        return classifier  #self.output_file




# class FileInput(luigi.ExternalTask):
#     # '''
#     # Define the input file for our job
#     # The output method of this class defines
#     # the input file of the class in which FileInput is
#     # referenced in &quot;requires&quot
#     # '''
    
#     # Parameter definition: input file path
#     input_path = luigi.Parameter()


#     input_ , output =  skl.load_boston(return_X_y=True)
#     print(input_.shape)

#     def output(self):
#         # '''
#         # As stated: the output method defines a path.
#         # If the FileInput  class is referenced in a
#         # &quot;requires&quot; method of another task class, the
#         # file can be used with the &quot;input&quot; method in that
#         # class.
#         # '''
#         return luigi.LocalTarget(self.input_path)
    

#     def requires(self):
#         # '''
#         # Requires the output of the previously defined class.
#         # Can be used as input in this class.
#         # '''
#         return FileInput(self.input_path)
    
#     def output(self):
#         # '''
#         # count.txt is the output file of the job. In a more
#         # close-to-reality job you would specify a parameter for
#         # this instead of hardcoding it.
#         # '''
#         return luigi.LocalTarget('count.txt')
    
#     def run(self):
#         # '''
#         # This method opens the input file stream, counts the
#         # words, opens the output file stream and writes the number.
#         # '''
#         # word_count = 0
#         # with self.input().open('r') as ifp:
#         #     for line in ifp:
#         #         word_count += len(line.split(' '))

#         # with self.output().open('w') as ofp:
#         #     ofp.write(unicode(word_count))

#         global input_

#         print(input_.shape)




# class TrainModelTask(luigi.Task):
#     """ Trains a classifier to predict negative, neutral, positive
#         based only on the input city.
#         Output file should be the pickle'd model.
#     """
#     features_file = luigi.Parameter(default='features.csv')
#     output_file = luigi.Parameter(default='model.pkl')

#     # TODO...
    
#     def requires(self):
#         return TrainingDataTask()    
    
#     #split the datset :    
    
#     def run(self):  
        
         
#         # clf_door = svm.SVC(probability=True)    
    
#         # # fit SVM model:
#         # svm_model=clf_door.fit(X, Y) 
     
#         # open the file for writing
#         fileObject = open(self.output_file,'wb') 
        
#         # this writes the object a to the
#         pickle.dump(svm_model,fileObject)   
        
#         # here we close the fileObject
#         fileObject.close()    
    
#     def complete(self):
#         if((os.path.getsize(self.output_file) > 1000)):
#             return True        
#         else:
#             return False        



# class ScoreTask(luigi.Task):
#     """ Uses the scored model to compute the sentiment for each city.
#         Output file should be a four column CSV with columns:
#         - city name
#         - negative probability
#         - neutral probability
#         - positive probability
#     """
#     tweet_file  = luigi.Parameter('cities.csv')
#     #city_mapping_dict= luigi.Parameter('city_mapping.mat')
#     output_file = luigi.Parameter(default='scores.csv')
    
#     model_file  = luigi.Parameter(default='model.pkl')

#     # TODO...
    
#     #dependency requirements of tasks should cascade back along the pipeline from this final 'scoring' task...
#     def requires(self):
#         return TrainModelTask()    
    
#     #split the datset :       
#     def run(self):     
                
#         fileObject = open(self.model_file,'r')     
#         model=pickle.load( open( self.model_file ,"rb" ) )                
        
#         datafile.close()
#         numbersFile.close()
        
#     #check for completeness of the task(s) by checking that the output file is non empty    
#     def complete(self):
#         if(os.path.getsize(self.output_file ) > 2000): 
#             return True   
#         else:
#             return False


# if __name__ == "__main__":
#     luigi.run()

    
if __name__ == '__main__' :
    #luigi.run(main_task_cls=CountIt)

    # input_ , output =  skl.load_boston(return_X_y=True)
    # print(input_.shape)

    ML_pipe_1  = ml_pipeline()
    ML_pipe_1.readInput()

    check = 0
