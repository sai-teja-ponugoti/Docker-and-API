
import luigi
import sklearn as skl
import datetime,os,sys
import unittest
import numpy as np,scipy as sp
from sklearn.datasets import load_boston
import pickle

## Set a default file to store the input read in from the bodton housing data set via sci-kit learn  in 
curr_dir = os.getcwd()
in_file = curr_dir + '/' + 'input_file.txt'
if not os.path.isfile(  in_file ):
    os.system('copy nul ' + in_file)

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
TRAINING_PERCENTAGE_SPLIT   = 0.6
VALIDATION_PERCENTAGE_SPLIT = 0.2    
TEST_PERCENTAGE_SPLIT       = 0.2


# Define the overall modell - pipeline Class  which will have all the internal variables, data structures and most importantly eacvh stage of the pipeline :
## from MOdel raw data input reading, through data cleaning and standardization and feautre extraction finally ->  output prdictions.
class ml_pipeline:

    def __init__(self):

        self.num_samples           = 500
        self.input_file            = in_file
        self.clean_input_data_file = clean_in_data_file
        self.featurized_data_file  = features_data_file
        self.output_file           = out_file
        self.feature_vec_length    = 8
        self.feature_vector        = np.empty(self.feature_vec_length)
        self.feature_matrix        = np.zeros((self.num_samples, self.feature_vec_length))


    def readInput(self, data_source = in_file):

        print('in readInput() member function')
        #read in via sklearn boston data set

        # write to .txt inuot file created above.

        return self.input_file


    def cleanData(self, input_file='./input_file.txt'):

        print('in clean_data() member function')


        return self.clean_input_data_file


    def featureizeData(self, input = clean_in_data_file):

        print('in featureizeData() ml_pipeline class member function')

        return self.featurized_data_file


    def predict_and_writeOutput(self, input_data = features_data_file):

        print('in predict_and_writeOutput() ml_pipeline class member function')

        pass

        return self.output_file



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
    
if __name__ == __main__:
    #luigi.run(main_task_cls=CountIt)


    input_ , output =  skl.load_boston(return_X_y=True)
    print(input_.shape)

    check = 0