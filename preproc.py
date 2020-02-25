

####################
# Standard library #
####################
from pathlib import Path
from dataclasses import dataclass

######################
# 3rd party packages #
######################
import pandas as pd
import numpy as np
from toolz.curried import curry,compose

#####################
# Internal Specific #
#####################


@dataclass
class Categorical:
    '''Data container for categorical data that contains the mappings from
    unique integer to categorical name

    Also contains the training and test split needed to train a model
    '''
    filename    : str
    features    : list
    split       : float = 0.7
    time_column : str = 'Crash_Date'
    target      : str = 'Count'
    def __post_init__(self):
        #-------------------------------------
        # Save the Raw dataframe
        #-------------------------------------
        self.df_raw = read_crash_data(self.filename, self.features, astype='category')
        df_time     = read_crash_data(self.filename, [self.time_column], astype='str')
        #--------------------------------------------------
        # Save the categorical encoding maps
        #--------------------------------------------------
        self.df_encoded = process_crash_report_data(self.df_raw,
                                                    df_time,
                                                    self.features,
                                                    self.cat2code_map)
        #--------------------------------------------------
        # Split the training and testing data
        #--------------------------------------------------
        df_train, df_test  = split_train_and_test(self.df_encoded)
        self.Xtrain, self.Xtest = df_train[self.features], df_test[self.features]
        self.Ytrain, self.Ytest = df_train[self.target], df_test[self.target]

    @property
    def code2cat_map(self):
        df = self.df_raw
        return {col : dict(list(enumerate(df[col].cat.categories))) for col in df}

    @property
    def cat2code_map(self):
        mapper = self.code2cat_map
        return {col : {v:k for k,v in code2cat.items()} for col,code2cat in mapper.items()}


#---------------------------------------------------------------------
#
# Functions used to create a functional data processing pipeline
#
#---------------------------------------------------------------------
@curry
def process_crash_report_data(df, df_time, features, cat2codes):
    '''Composes the entire crash reports data pipeline taking a raw dataframe
    '''
    return compose(get_most_common_per_day,
                   make_datetime_index(df_time=df_time),
                   encode_categorical_data)(df, cat2codes)


@curry
def make_datetime_index(df, df_time):
    return df.set_index( pd.to_datetime(df_time[df_time.columns[0]].str.slice(0,19),
                                        format='%Y-%m-%d %H:%M:%S') ).sort_index()

@curry
def encode_categorical_data(df, mapper):
    return pd.concat( [pd.Series(df[col].map(mapper[col]), name=col) for col in df],
                      axis=1 )

@curry
def read_crash_data(input_file, features, astype):
    return pd.read_csv(input_file,
                       delimiter=';',
                       usecols=features).astype(astype)


def calc_mode(x):
    return x.mode() if x.shape[0] > 2 else x

def get_number_of_accidents_per_day(df_resampled):
    return df_resampled.size().rename('Count').dropna()


@curry
def get_most_common_per_day(df, freq='1D'):
    # Create resampled group
    resampled = df.resample(freq)
    # Compute most common occurrence of categorical data and clean up
    most_common = resampled.agg(calc_mode)
    most_common = most_common.set_index(most_common.index.droplevel(1)).dropna()
    most_common = most_common[~most_common.index.duplicated(keep='first')]
    # Count the number of accidents per day and store as a new column
    df_accident_count = get_number_of_accidents_per_day(resampled)
    return pd.concat( [most_common, df_accident_count], axis=1 ).dropna()



def split_train_and_test(df, split=0.75):
    shuffled = df.sample(frac=1, replace=False)
    itrain   = int(df.shape[0] * split)
    return shuffled.iloc[:itrain], shuffled.iloc[itrain:]


def calc_mse(actual, pred):
    return np.mean( (pred - actual)**2 )





