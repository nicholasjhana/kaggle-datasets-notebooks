#!/usr/bin/env python
# coding: utf-8

# ## Model Pipeline Structure
# 
# #### Funcitons we need
# 1. Feature selector
#     a. function to make the basic feature we always use
#     b. function to add the new features we make to the main df/array
# 2. Model cross validator
#     a. Can take a different model and return cross validations
#     b. Can grid search on parmaters
# 3. Model ensemble predictions
#     a. Makes predictions on multiple folds and averages them per model

# In[2]:


# from comet_ml import Experiment
    
# # Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="gBHDDs5I5XQ8SwfIiLMK5fKMx",
#                         project_name="ashre", workspace="nicholasjhana")

import numpy as np
import pandas as pd
import gc
from time import gmtime, strftime


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold


import lightgbm as lgb
# import comet_ml in the top of your file

from collections import defaultdict


# In[8]:


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


# ### Loading Data Functions

# In[52]:


def load_train_data(path='/Users/ns/code/nicholasjhana/ashrae-energy-prediction/data/'):
    
    #load train datasets and inital size
    building_df = pd.read_csv(path + "building_metadata.csv")
    weather_train = pd.read_csv(path + "weather_train.csv")
    train = pd.read_csv(path + "train.csv")
    print('inital load {}'.format(train.shape))
    
    #add additional features
    weather_train = make_weather_features(weather_train)
    building_df = transform_floors_to_bins(building_df)
    
    #merge train and building id data and check size
    train = train.merge(building_df, 
                        left_on = "building_id", 
                        right_on = "building_id", 
                        how = "left")
    print("after building df load {}".format(train.shape))
    
    #merge train and weather_train data and check size
    train = train.merge(weather_train, 
                        left_on = ["site_id", "timestamp"], 
                        right_on = ["site_id", "timestamp"],
                        how = "left")
    print("after weather load {}".format(train.shape))
    
    #clean up
    del building_df
    del weather_train
    
    return train


# In[53]:


def load_test_data(path='/Users/ns/code/nicholasjhana/ashrae-energy-prediction/data/'):
    
    #load the test datasets
    building_df = pd.read_csv(path + "building_metadata.csv")
    weather_test = pd.read_csv(path + "weather_test.csv")
    test = pd.read_csv(path + "test.csv")
    
    #check the size
    print('inital load {}'.format(test.shape))
    
    #add additional features to the weather and building datasets
    weather_test = make_weather_features(weather_test)
    building_df = transform_floors_to_bins(building_df)
    
    #merge test and building id and check size
    test = test.merge(building_df, 
                      left_on = "building_id", 
                      right_on = "building_id", 
                      how = "left")
    print("after building df load {}".format(test.shape))
    
    #merge test and wether_test and check size
    test = test.merge(weather_test, 
                      left_on = ["site_id", "timestamp"], 
                      right_on = ["site_id", "timestamp"], 
                      how = "left")
    print("after weather load {}".format(test.shape))
    
    #clean up
    del building_df
    del weather_test
    
    return test


# In[29]:


def make_weather_features(weather_df):
    
    #filling wind speed with zero value for nans. 
    #Needs to be checked with the values around the nan values
    weather_df['wind_speed'] = weather_df['wind_speed'].fillna(0)
    
    #filling dew temperatures with mean values for nans
    #needs to be checked with values around the nan values
    dew_mean = weather_df['dew_temperature'].mean()
    weather_df['dew_temperature'] = weather_df['dew_temperature'].fillna(dew_mean)
    
    
    #wind speed and dew temp transform as log(X - (min(X)-1))
    #this avoids negative values passed into the log
    weather_df['wind_speed_log'] = np.log(weather_df['wind_speed'] + 1)
    weather_df['dew_temp_log'] = np.log(weather_df['dew_temperature'] - (weather_df['dew_temperature'].min()-1))
    
    return weather_df


# In[12]:


####
#single, ‘low-’, ‘mid-’ and ‘high-rise’ are used, defined as 1, 2 - 5, 6–10 and ≥ 11
####

def transform_floors_to_bins(building_df):
    
    bins = [0,2,6,11,np.inf]
    labels = ['single', 'low', 'mid', 'high-rise']
    building_df['floor_count'] = pd.cut(building_df['floor_count'], bins, labels=labels).astype('str')
    #print(building_df['floor_count'].value_counts())
    
    return building_df
    


# ### Making Features Functions

# In[42]:


def make_time_features(data):
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["hour"] = data["timestamp"].dt.hour.astype(np.uint8)
    data["year"] = data["timestamp"].dt.year.astype(np.uint16)
    data["day"] = data["timestamp"].dt.day.astype(np.uint8)
    data["weekend"] = data["timestamp"].dt.weekday.astype(np.uint8)
    data["month"] = data["timestamp"].dt.month.astype(np.uint8)
    del data['timestamp']
    del data['year']

    return data


# In[41]:


def make_add_features(data):
    data['year_built'] = data['year_built']-1900
    data['square_feet_log'] = np.log(data['square_feet'])
    
    return data


# In[33]:


# def select_categroical_cols():
#     return ["site_id", "building_id", "primary_use", 'floor_count', "hour", "day", "weekend", "month", "meter"]

# def select_numerical_cols():
#     return ["square_feet_log", "year_built", "air_temperature", "cloud_coverage", "dew_temp_log", 'wind_speed_log']

# def get_all_features():
#     return select_categroical_cols() + select_numerical_cols()


# ### Dataset Creation Pipeline

# In[34]:


def prep_train_data(categorical, numerical, label_cols):
    train_df = load_train_data()
    train_df = make_time_features(train_df)
    train_df = make_add_features(train_df)
    
    #label encode all categorical/binned columns - can be turned into support function
    #see the following for use: 
    #https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
    dict_LabelEncoder = defaultdict(LabelEncoder)
    train_df[label_cols] =  train_df[label_cols].apply(lambda x: dict_LabelEncoder[x.name].fit_transform(x))


    train_df, NA_list_train = reduce_mem_usage(train_df)
    
    #select the target Y col and remove from train set
    target = np.log1p(train_df["meter_reading"])

    del train_df['meter_reading']
    
    #returns a list of the columns to be used in the model
    features = categorical + numerical
    train_df = train_df[features]
    
    numerical_scaler = MinMaxScaler()
    train_df[numerical] = numerical_scaler.fit_transform(train_df[numerical])

    print('Features in dataset')
    print(train_df.info())

    train_dict = dict()
    train_dict['labels'] = dict_LabelEncoder
    train_dict['scaler'] = numerical_scaler

    
    return train_df, target, train_dict


###Not in use currently
# Must be adapted to take in the label encoder dictionary
def prep_test_data(categorical, numerical, train_dict):
    test_df = load_test_data()
    test_df = make_time_features(test_df)
    test_df = make_add_features(test_df)
    
    dict_LabelEncoder = train_dict['labels']
    numerical_scaler = train_dict['scaler']
    label_cols = list(dict_LabelEncoder.keys())

    
    test_df[label_cols] =  test_df[label_cols].apply(lambda x: dict_LabelEncoder[x.name].transform(x))
    
    test_df, NA_list_train = reduce_mem_usage(test_df)
    
    features = categorical + numerical
    test_df = test_df[features]
    

    test_df[numerical] = numerical_scaler.transform(test_df[numerical])


    print('Features in dataset')
    print(test_df.info())
    
    return test_df


# ## Run the preprocessing pipeline

# In[21]:




def train_lgbm(train_df, target, params, categorical, folds=2, seed=42):

    kf = KFold(n_splits = folds, shuffle = True, random_state = seed)
    models = []
    for train_index, val_index in kf.split(train_df):
        train_X = train_df.iloc[train_index]
        val_X = train_df.iloc[val_index]
        train_y = target.iloc[train_index]
        val_y = target.iloc[val_index]
        lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categorical)
        lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categorical)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=300,
                        valid_sets=(lgb_train, lgb_eval),
                        early_stopping_rounds=100,
                        verbose_eval = 100)
        models.append(gbm)
        
    return models
    
def clean_up():
        del train_df, train_X, val_X, lgb_train, lgb_eval, train_y, val_y, target
        gc.collect()


# In[61]:


def run_training_model(params, categorical, numerical, label_cols = ['primary_use', 'floor_count']):
    
    train_df, target, train_dict = prep_train_data(categorical, numerical, label_cols)


    models = train_lgbm(train_df, target, params, categorical, folds=2, seed=42)
    
    outputs = dict()
    outputs['models'] = models
    outputs['params'] = params
    outputs['categories'] = categorical
    outputs['numerical'] = numerical
    outputs['labels'] = train_dict['labels']
    
    
    return outputs





def gen_test_prediction(models, path = './submissions/lgbm/', file_name='submission_v1_'):
    #run the model and generate predictions
    i=0
    res=[]
    step_size = 50000
    for j in range(int(np.ceil(test_df.shape[0]/50000))):
        res.append(np.expm1(sum([model.predict(test_df.iloc[i:i+step_size]) for model in models])/folds))
        i+=step_size
    
    #combine results
    res = np.concatenate(res)
    
    #set file name for submission
    submission_name = file_name + strftime("%Y%m%d-%H%m", gmtime()) + ".csv"
    
    submission = pd.read_csv('/Users/ns/code/nicholasjhana/ashrae-energy-prediction/data/sample_submission.csv')
    submission.shape, test.shape
    submission['meter_reading'] = res
    submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0
    submission.to_csv(path + submission_name, index=False)
    
    return submission
