# from api import api
import sys
import pickle
sys.path.append('../')
# import flask
from flask import Flask, request, jsonify, render_template, url_for
from flask import request, jsonify
# from flask_restful import reqparse
from housePricerPipeline import *


api = Flask(__name__)

@api.route("/")
def hello():
    return "Hello - welcome to my house pricing API!"

#def get():

@api.route("/predict_house_value", methods=["POST"])
def predict_house_value():
    uploaded_data = request.get_json(force=True)
    #print('full json retrieved  = ' + str(uploaded_data))
    print("  data recieved is : ")
    print(uploaded_data.values())
    input = np.array([float(value) for value in uploaded_data.values()])
    # print(uploaded_data['parama1'])
    print("input : ",input)
    print("input shape " , input.shape)
    print("input type " , type(input))
    input = input.reshape(1, -1)
    print("input before min max : ",input)
    # using the same scaler that is used on the training data
    input = mlpipe.scaler.transform(input)
    print("input after min max : ",input)
    predictions = loaded_model.predict(input.reshape(1, -1))
    print(predictions)

    return jsonify(predictions[0])

if __name__ == '__main__':

    mlpipe  = ml_pipeline()
    mlpipe.readInput()
    mlpipe.preprocessData()
    mlpipe.trainModel()
    mlpipe.train_and_writeOutput()
    loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
    api.run(debug=True,port=8080) 
   