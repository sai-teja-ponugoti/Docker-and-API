from flask import Flask,jsonify,request
#from housePricer import *
#import flask_restful 
#from flask_restful import Api,reqparse

api = Flask(__name__)

from api import route

# @api.route("/")
# def hello():

#     return "Hello World!"

# #def get():

# @api.route("/predict", methods=["POST"])
# def send_house_value():
#     house_features = request.json.get('house_features')
#     #name = request.json.get('name')
#     if not house_features: # or not name:
#         return jsonify({'error': 'Please provide house_features'}), 400

#     # resp = client.put_item(
#     #     TableName=USERS_TABLE,
#     #     Item={
#     #         'userId': {'S': user_id },
#     #         'name': {'S': name }
#     #     }
#     # )

#     return jsonify({
#         'value': house_features
#     })
   
#if __name__ == '__main__':

api.run(debug=False) 
crap = 1
