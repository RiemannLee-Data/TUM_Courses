
import os, sys
from customscripts import configuration
from customscripts import utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import tensorflow as tf
from tensorflow import keras


class DataWrangling():
    def __init__(self, weeknumber, main_file, unneeded_columns = None):
        self.main_data = utils.get_dataset_df(weeknumber, main_file)
        self.gdp_data = utils.get_dataset_df(4, "gdp")
        self.population_data = utils.get_dataset_df(4, "population")
        self.population_data['Country Code'] = np.where(self.population_data['Country Code']=='DEU', 'GER', self.population_data['Country Code'])
        self.gdp_data['Country Code'] = np.where(self.gdp_data['Country Code']=='DEU', 'GER', self.gdp_data['Country Code'])
    #def __init__(self, weeknumber, main_file, unneeded_columns):
    #    self.week_data = utils.get_dataset_df(weeknumber, main_file)
    #    
    #    self.week_data = self.week_data.drop(columns=unneeded_columns)
        
    #    
    #    self.week_data['Reporter ISO']=self.week_data['Reporter']
    #    self.week_data = utils.abbreviate_countries(self.week_data, 'Reporter ISO')
        #self.week3_data = utils.get_dataset_df(3, "all_categories")
        #self.week3_data['Product_Type'] = 'Food'
    #    self.week_data = self.week_data.rename(columns={'Period Desc.':"Time", 'Category':'Product_Type'})
    #    #self.week3_data = self.week3_data[~(self.week3_data['Trade Flow'] != 'Import')]
    #    self.gdp_data = utils.get_dataset_df(4, "GDP_per_capita")
    #    self.population_data = utils.get_dataset_df(4, "population")
    #    self.main_data = self.week_data.copy()
    def wrangle_step1(self):
        countries = self.main_data["Reporter"].unique().tolist()
        gdp_data = self.gdp_data[~(self.gdp_data.Year < 2009)]
        gdp_data = self.gdp_data[~(self.gdp_data.Year == 2020)]
        gdp_data = gdp_data[gdp_data["Country Code"].isin(countries)]
        population_data = self.population_data[self.population_data["Country Code"].isin(countries)]
        population_data = population_data[population_data.columns[~(population_data.columns < "2009")]]
        population_data = population_data.drop(columns={"2020", "Unnamed: 65"})
        #gdp_data = gdp_data.drop(columns={"2020"})
        main_data = self.main_data
        main_data = main_data[~main_data['Time'].isnull()]
        #main_data['Partner ISO'].fillna(main_data.Partner, inplace=True)
        years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

        gdpColumnName = 'Gdp_per_capita'
        populationColumnName = "Population"
        main_data['Time'] = pd.to_datetime(main_data['Time'], format="%Y-%m-%d")
        main_data = self.merge_annual_data_columnwise(main_data.copy(), gdp_data, years, gdpColumnName)
        main_data = self.merge_annual_data_rowwise(main_data.copy(), population_data, years, populationColumnName)
        self.main_data = main_data
        return main_data

    def wrangle_step2(self, columns_to_drop=None, needed_flow=None, product_to_remove=None):
        gdpColumnName = 'Gdp_per_capita'
        populationColumnName = "Population"
        data_all_cats = self.main_data.copy()
        data_all_cats=data_all_cats[~(data_all_cats.Time<"2009")]
        
        #print(f'Unique Times {data_all_cats["Time"].unique()}')
        data_all_cats.sort_values(by=['Time', 'Reporter'], inplace=True)
        norm_trade_col = 'Trade_val_per_capita'
        #norm_gdp_col = 'log (GDP) p/c'
        data_all_cats[norm_trade_col] = data_all_cats['Trade Value (US$)']/data_all_cats['Population']
        data_all_cats[gdpColumnName] = data_all_cats[gdpColumnName] /data_all_cats['Population']
        #print(f'Gdp per capita before log\n{data_all_cats[gdpColumnName].isnull().sum()}')
        #data_all_cats[gdpColumnName] = np.log(data_all_cats[gdpColumnName])
        #data_all_cats[norm_gdp_col] = np.log(data_all_cats[gdpColumnName])
        #data_all_cats = data_all_cats[data_all_cats["Trade Flow"]==needed_flow]
        data_all_cats = data_all_cats[~(data_all_cats['Category Code']==product_to_remove)]
        data_all_cats = data_all_cats.drop(columns=columns_to_drop)
        data_all_cats = data_all_cats.set_index(["Time"])
        data_all_cats.index = data_all_cats.index.to_period('M')
        return data_all_cats



    def merge_annual_data_columnwise(self, df, data, years, new_col_name):
        df[new_col_name] = 0.0
        #df['Year'] = df['Year'].astype('str') 
        for index, row in data.iterrows():
            val = int(row['Year'])
            country = str(row['Country Code'])
            df[new_col_name] = np.where((df['Time'].dt.year ==val) & (df['Reporter'] == country), row['Value'], df[new_col_name])
        return df

    def merge_annual_data_rowwise(self, df, data, years, new_col_name):
        df[new_col_name] = 0.0
        #df['Year'] = df['Year'].astype('str') 
        for index,row in data.iterrows():
            country = str(row["Country Code"])
            for i, val in enumerate(years):
                val = int(val)
                df[new_col_name] = np.where((df['Time'].dt.year ==val) & (df['Reporter'] == country), row[str(val)], df[new_col_name])
        return df










