import numpy as np
import pandas as pd
import os

#change to working directory
os.chdir("/Users/jacksonwalters/Documents/GitHub/enefit-kaggle/predict-energy-behavior-of-prosumers/")

#helper function to convert datetime strings to integers representing a time year-month-day hour-min-sec
from datetime import datetime
def datestr_to_int(datetime_str,date_format):
    if not pd.isna(datetime_str):
        return datetime.strptime(datetime_str, date_format).timestamp()
    else:
        return float('nan')

#function to load all data
def merged_df():

    #load the training data, dropping NaN's
    train = pd.read_csv("train.csv").dropna()
    train['datetime'] = train['datetime'].apply(lambda x: datestr_to_int(x,'%Y-%m-%d %H:%M:%S'))

    #load gas_prices
    gas_prices = pd.read_csv("gas_prices.csv")
    #convert date strings to ints
    gas_prices['forecast_date'] = gas_prices['forecast_date'].apply(lambda x: datestr_to_int(x,'%Y-%m-%d'))
    gas_prices = gas_prices.drop(columns=['origin_date'])
    gas_prices['data_block_id'] -= 1

    #load electricity_prices
    electricity_prices = pd.read_csv("electricity_prices.csv")
    #convert date strings to ints
    electricity_prices['forecast_date'] = electricity_prices['forecast_date'].apply(lambda x: datestr_to_int(x,'%Y-%m-%d %H:%M:%S'))
    electricity_prices = electricity_prices.drop(columns=['origin_date'])
    electricity_prices['data_block_id'] -= 1

    #load forecast_weather
    forecast_weather = pd.read_csv("forecast_weather.csv")
    #convert strings to ints
    forecast_weather = forecast_weather.drop(columns=['origin_datetime'])
    forecast_weather['forecast_datetime'] = forecast_weather['forecast_datetime'].apply(lambda x: datestr_to_int(x,'%Y-%m-%d %H:%M:%S%z'))
    forecast_weather = forecast_weather.rename(columns={'forecast_datetime':'forecast_date'})
    forecast_weather['data_block_id'] -= 1

    #merge gas prices and train.csv data
    #column names differ, so use left_on and right_on
    df = pd.merge(train, gas_prices, left_on=['data_block_id','datetime'], right_on=['data_block_id','forecast_date'], how='left')

    #merge train and gas_prices via left join on data_block_id
    #this leaves all rows of train, but matches
    df = df.merge(electricity_prices, on=['data_block_id','forecast_date'], how='left')

    #merge forecast_weather on forecast date
    df = df.merge(forecast_weather, on=['data_block_id','forecast_date'],how='left')

    #drop NaN rows
    df = df.dropna()

    return df