# CHAT-BOT

An chat bot built with `transformers (TFDistilBertForQuestionAnswering)`  and `TensorFlow` 

The bot's services are consumed with `REST API CALLS` in a `Flask app`


## Steps to start the Application (Backend)

```
# go into chat-bot folder
cd chat-bot
```

**Creating and Activating Virtual Environment**

```
pip install virtualenv

# or

pip install venv
```

**Setup Virtual Environment**

```
python -m venv env
```

**Activate Virtual Environment**

```
# activate env (windows)

.\env\scripts\activate

# activate env (Linux/Mac)

source env/bin/activate
```

**Installing Dependencies**

```
pip install -r requirements.txt
```

**Starting Application**

```
flask run --host=0.0.0.0
```

**Deactivating Virtual Environment**

```
deactivate env
```

Visit http://localhost:5000 or http://0.0.0.0:5000 or http://yourIp:5000


**Back End (Server Side) Flask - Dependencies**

+ absl-py==1.2.0
+ astunparse==1.6.3
+ cachelib==0.9.0
+ cachetools==5.2.0
+ certifi==2022.6.15
+ charset-normalizer==2.1.0
+ click==8.1.3
+ colorama==0.4.5
+ filelock==3.8.0
+ Flask==2.2.2
+ Flask-Caching==2.0.1
+ Flask-Cors==3.0.10
+ flatbuffers==1.12
+ gast==0.4.0
+ google-auth==2.10.0
+ google-auth-oauthlib==0.4.6
+ google-pasta==0.2.0
+ grpcio==1.47.0
+ h5py==3.7.0
+ huggingface-hub==0.8.1
+ idna==3.3
+ itsdangerous==2.1.2
+ Jinja2==3.1.2
+ keras==2.9.0
+ Keras-Preprocessing==1.1.2
+ libclang==14.0.6
+ Markdown==3.4.1
+ MarkupSafe==2.1.1
+ numpy==1.23.2
+ oauthlib==3.2.0
+ onnx==1.12.0
+ onnxconverter-common==1.12.1
+ opt-einsum==3.3.0
+ packaging==21.3
+ protobuf==3.19.4
+ pyasn1==0.4.8
+ pyasn1-modules==0.2.8
+ pyparsing==3.0.9
+ PyYAML==6.0
+ regex==2022.7.25
+ requests==2.28.1
+ requests-oauthlib==1.3.1
+ rsa==4.9
+ six==1.16.0
+ tensorboard==2.9.1
+ tensorboard-data-server==0.6.1
+ tensorboard-plugin-wit==1.8.1
+ tensorflow==2.9.1
+ tensorflow-cpu==2.9.1
+ tensorflow-estimator==2.9.0
+ tensorflow-hub==0.12.0
+ tensorflow-io-gcs-filesystem==0.26.0
+ tensorflow-text==2.9.0
+ termcolor==1.1.0
+ tf2onnx==1.12.0
+ tokenizers==0.12.1
+ tqdm==4.64.0
+ transformers==4.21.1
+ typing_extensions==4.3.0
+ urllib3==1.26.11
+ Werkzeug==2.2.2
+ wrapt==1.14.1


