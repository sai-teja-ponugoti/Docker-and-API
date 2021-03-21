from api import api
import sys
sys.path.append('../')
from flask import request, jsonify
from flask_restful import reqparse
from housePricerPipeline import *

parser = reqparse.RequestParser()
# parser.add_argument('val00', type=str)
# parser.add_argument('val01', type=str)
# parser.add_argument('val11', type=str)
# parser.add_argument('val12', type=str)
# parser.add_argument('val3', type=str)
# parser.add_argument('val4', type=str)
# parser.add_argument('val5', type=str)
# parser.add_argument('val6', type=str)
# parser.add_argument('val7', type=str)
# parser.add_argument('val8', type=str)
# parser.add_argument('val9', type=str)
# parser.add_argument('val10', type=str)
# parser.add_argument('val13', type=str)


@api.route("/")
def hello():

    return "Hello - welcome to my house pricing API!"

#def get():

@api.route("/predict_house_value", methods=["POST"])
def predict_house_value():
    #house_features = request.json.get('house_features')           #request.json.get('house_features')
    #print("username : \n")

    #uploaded_data = request.get_data(as_text = True) #['output_file.txt']
    uploaded_data = request.get_json(force=True)
    #print('full json retrieved  = ' + str(uploaded_data))

    args = parser.parse_args()

    #value1 = float(str(args['val00']))
    unseen_house_features = [48.0] 
    for arg in args:
        print('args type = ' + str(type(args)))
        print('arg is actully ->' + str(arg) + '<-')
        #unseen_house_features.append(float(arg))

    # unseen_house_features.append(float(args['val01']))
    # unseen_house_features.append(float(args['val3']))
    # unseen_house_features.append(float(args['val4']))
    # unseen_house_features.append(float(args['val5']))
    # unseen_house_features.append(float(args['val6']))
    # unseen_house_features.append(float(args['val7']))
    # unseen_house_features.append(float(args['val8']))
    # unseen_house_features.append(float(args['val9']))
    # unseen_house_features.append(float(args['val10']))
    # unseen_house_features.append(float(args['val11']))
    # unseen_house_features.append(float(args['val12']))
    # unseen_house_features.append(float(args['val13']))

    #value4 = str(args['val13'])

    print('value4 = '  + str(value4))
    
    #unseen_house_features_json  = jsonify(  uploaded_data )  #  request.args.get('ipnuts'))

    #print('type of  object retrieved  = ' + str(type(unseen_house_features_json)))
    # unseen_house_features = []
    # for key,val in unseen_house_features_json:

    #     unseen_house_features.append(val)

    #predicts = input_list['val4']
    unseen_house_features = list(value1) 
    
    # for indx in range(BOSTON_NUM_FEATURES):
        
    #     , value2 
    #     ,value3 
    #     ,value4 
    #     ,value5 
    #     ,value6 
    #     ,value7 
    #     ,value8 
    #     ,value9 
    #     ,value10
    #     ,value11
    #     ,value12
    #     ,value13)


    #return jsonify(uploaded_data)

    ML_pipe_1  = ml_pipeline()
    ML_pipe_1.readInput()

    ML_pipe_1.preprocessData()
    clasifier = ML_pipe_1.trainModel()

    #clasifier = ML_pipe_1.cleanData() # readInput()

    # preprocess inpit test data in prep. for inference of trained model.
    Max = float(np.max(unseen_house_features))
    Min = float(np.min(unseen_house_features))

    if (Max != Min) :
        stdized_unseen_house_features  = unseen_house_features/(Max - Min)
    else:
        stdized_unseen_house_features = unseen_house_features/ Max

    # ML_pipe_1.train_and_writeOutput()  # readInput()
    predicttions = clasifier.predict(stdized_unseen_house_features)

    print("Predicted house price = " + str(predicttions))
    # if not predicttions: # or not name:
    #     return jsonify({'error': 'Please provide house_features'}), 400

    return jsonify(predicttions)

    # # resp = client.put_item(
    # #     TableName=USERS_TABLE,
    # #     Item={
    # #         'userId': {'S': user_id },
    # #         'name': {'S': name }
    # #     }
    # # )

    # return jsonify({
    #     'value': predicttions
    # })
   