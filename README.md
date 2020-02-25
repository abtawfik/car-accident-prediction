# Predicting Accidents in Cary, NC

Given weather, road, and light conditions can we predict the number of accidents that day?

This repo is a simple example illustrating how to train and test a model, and then create an endpoint to make predictions from the trained model.
**Note**: This is just for illustrative purposes and the model is *not* meant to provide any real guidance.

## What's in this repo
The task of each file is described below:
* `train_model.py` - Loads the crash data, performs preprocessing, and then trains several models. The "best" model is selected by comparing to a simple benchmark model (a model that always guesses the mean of the target) against the test data. Models and some metadata are then saved.
* `api.py` - An interface so users can query the trained model. When the user passes weather, road, and light conditions the number of accidents is predicted.
* `preproc.py` - dataclasses and functions used for extracting features and splitting the data
* `benchmark.py` - contains the simple benchmark model
* `Crash_Test_Data.ipynb` - A jupyter notebook that has the same contents as `train_model.py`
* `data/` - contains the crash data retrieved from the Cary data portal

## Installing
After cloning the repo, install all the requirements (preferably in a virtual environment)
```
pip install -r requirements.txt
```


## Running
First train the model by running
```
python train_model.py
```
This will output several files:
* `acceptable_model.sav` - the trained model
* `cat2code.pkl` - the translation from string categories to an integer (e.g. if weather is 'DRY' it maps it to 2)
* `code2cat.pkl` - the opposite mapping from integer to category

You can now launch the REST endpoint and pass values to the model
```
hug -f api.py
```

This will launch `hug` and expose port 8000 on the localhost. To predict the number accidents given road, light, and weather conditions you curl the endpoint or just type this in the browser
```
http://localhost:8000/accidents?weather=CLEAR&road=DRY&light=DAWN
```

To see a list of choices for each variable just got to:
```
http://localhost:8000
```
