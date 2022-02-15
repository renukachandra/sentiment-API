## Target
1. deploy your deep learning model as a REST API 
2. add a form to take the input from the user
3. return the predictions from the model

We will use FastAPI to create it as an API and deploy it for free on Heroku.

# Step 1: Installations

1. FastAPI + Uvicorn
$ pip install fastapi uvicorn

2. Tensorflow 2
$ pip install tensorflow==2.6.0

3. Heroku
$ brew tap heroku/brew && brew install heroku

checkpoint sentiment.h5 can be used to serve prediction directly without running the model

# Step 2: Creating our Deep Learning Model

Model is saved in model.py

# Step 3: Creating a REST API using FAST API

APIs are created in app.py

# Step 4: Check predictions

1. run app.py
This will run your app on localhost. On the http://127.0.0.1:8000/predict route, you can see the input form.

2. Input your text and enjoy the predictions

