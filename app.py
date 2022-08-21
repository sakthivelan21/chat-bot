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

passages={

'passage': """Hi My name is I am Jarvis, a Digital Assitant. Nice To meet you. My purpose is to help you at any situtations. Our company name is Planner's Hub.The need of Employee portal is to have quick and easy access to HR-related transactions and services.
Our company target is to help you to Plan and Utilize Resources. Our products are reliable and more secure to use.vendor portal is to access and view the complete transaction between the company and vendor. 
Our products are customer portal , vendor portal and employee portal. our mail id is incrediblesakthi21@gmail.com.Customer portal is used to access and view the complete transaction between the company and customer. bye!! have a nice day!""",

'customerPortalPassage':"""The need of Customer portal is to access and view the complete transaction between the company and customer.
Inquiry data is a request to a company that they provide a quotation or sales information without obligation.
A customer inquiry comprises one or more items that contain the quantity of a material or service that the customer asked for.
Sale order is a request from a customer to a company to deliver a defined quantity of products or provide a service at a certain time.
The delivery document is the central object in the Shipping component. Delivery Document contains info like product name, scale unit price, time, date, address, movement type, storage type etc. 
Invoice is a document which holds the amount to be paid back to the vendor. 
Payment and aging is about classifying the unpaid credit memos and invoices from customers by date periods.
A debit memo is a transaction that reduces Amounts Payable to a vendor.
A transaction that reduces Amounts Receivable from a customer is a credit memo.
The ShortCut keys for table are perform Customize Table(Alt + c),Download as Excel(Alt + j),Global Search (Alt + s) and Reset Table(Alt + r). 
To Perform Commands on the table try saying "search hi","sort column name" like that.
Commands in table are search, sort, filter,download etc. if you need help in commands, type  'command help' to get more details.
""",


'vendorPortalPassage':"""The need of vendor portal is to access and view the complete transaction between the company and vendor. 
Request for quotation is a form of invitation that is sent to the vendors to submit a quotation indicating their pricing and terms and conditions. 
A formal purchase order will be generated which include material details, quantity and other related details.
Goods receipt refers to the physical movement of goods into the warehouse from external vendors.
Invoice is a document which holds the amount to be paid back to the vendor. 
Payment and aging is about classifying the unpaid credit memos and invoices from customers by date periods.
A debit memo is a transaction that reduces Amounts Payable to a vendor.
A transaction that reduces Amounts Receivable from a customer is a credit memo.
The ShortCut keys for table are perform Customize Table(Alt + c),Download as Excel(Alt + j),Global Search (Alt + s) and Reset Table(Alt + r). 
To Perform Commands on the table try saying "search hi","sort column name" like that.
Commands in table are search, sort, filter,download etc. if you need help in commands, type  'command help' to get more details.""",

'employeePortalPassage':"""The need of Employee portal is to have quick and easy access to HR-related transactions and services.Employee leave data needs to have the provision to display the complete leave data of the employee. 
Pay slip contains employee payroll-related information like payment type, start date, and pay date.
The ShortCut keys for table are perform Customize Table(Alt + c),Download as Excel(Alt + j),Global Search (Alt + s) and Reset Table(Alt + r). 
To Perform Commands on the table try saying "search hi","sort column name" like that.
Commands in table are search, sort, filter,download etc. if you need help in commands, type  'command help' to get more details.
"""
}



@app.route("/getChatBotMessage",methods=['POST'])
def getChatBotMessage():
    requestData = request.json
    passageType = requestData.get('passageType')

    inputs = tokenizer(requestData.get('question'), passages[passageType], return_tensors="tf")
    outputs = model(**inputs)
    answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    return make_response(jsonify({'result':tokenizer.decode(predict_answer_tokens)}))

# to start the server in debug mode
if __name__=="__main__":
	app.run(debug=True)


