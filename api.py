
###############
# Computation #
###############
import numpy as np

###############
# Api library #
###############
import hug

####################
# Standard library #
####################
import pickle

#--------------------------------
# Read in model and encodings
#--------------------------------
with open('acceptable_model.sav', 'rb') as f:
    model = pickle.load(f)
with open('cat2code.pkl', 'rb') as f:
    cat2code_map = pickle.load(f)

#--------------------------------
# Specify the possible choices
#--------------------------------
weather_types = list(cat2code_map.get('Weather').keys())
road_types    = list(cat2code_map.get('Road_Conditions').keys())
light_types   = list(cat2code_map.get('Light_Condition').keys())

                 
#--------------------------------
# Define the endpoint
#--------------------------------
@hug.get()
def accidents(weather : hug.types.one_of(weather_types),
              light   : hug.types.one_of(light_types),
              road    : hug.types.one_of(road_types)):
    '''Predict the number of accidents in a day given the 
    weather, road, and light conditions

    Feature order is:
    features = ['Road_Conditions', 'Light_Condition', 'Weather']

    '''
    inputs = np.array([cat2code_map['Road_Conditions'].get(road),
                       cat2code_map['Light_Condition'].get(light),
                       cat2code_map['Weather'].get(weather)]).reshape(-1,1).T
    return {'Expected # of Accidents':np.round(model.predict(inputs))[0] }
    
