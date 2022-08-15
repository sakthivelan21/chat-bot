from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
import tensorflow as tf
from flask import Flask,request,make_response,jsonify
from flask_cors import CORS

# setting the flask class as app
app=Flask(__name__)

#adding cross origin to flask app
cors=CORS(resources={
    r'/*': {
        'origins': [
            'http://localhost:3000',
        ]
    }
	})
cors.init_app(app)

# calling the pretrained models for NLP
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

passage:str = "Hi I am Jarvis. Nice To meet you. I am a digital assitant to simplify your work"

@app.route("/getChatBotMessage",methods=['POST'])
def getChatBotMessage():
    requestData = request.json
    inputs = tokenizer(requestData.get('question'), passage, return_tensors="tf")
    outputs = model(**inputs)
    answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    return make_response(jsonify({'result':tokenizer.decode(predict_answer_tokens)}))

# to start the server in debug mode
if __name__=="__main__":
	app.run(debug=True)


