import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pycountry

def get_dataset_dir():
    return f"{os.path.dirname(os.path.dirname(__file__))}/dataset"

def get_raw_dataset_path():
    return f"{get_dataset_dir()}/comtrade.csv"

def get_raw_dataset_df():
    return pd.read_csv(get_raw_dataset_path())

def get_race_dataset_path():
    return f"{get_dataset_dir()}/race_ALL.csv"

def get_raw_food_dataset_path():
    return f"{get_dataset_dir()}/food_data.csv"

def get_raw_food_dataset_df():
    return pd.read_csv(get_raw_food_dataset_path())

def get_clean_comtrade():
    raw_data = get_raw_dataset_df()
    data_no_junk = raw_data.dropna(axis=1, how="all")
    data_no_zeros = data_no_junk.loc[:, (data_no_junk != 0).any(axis=0)]
    data_drop_netweight = data_no_zeros.drop(columns=["Netweight (kg)"])
    data_drop_period = data_drop_netweight.drop(columns=["Period"])
    data_drop_period_year = data_drop_period.replace({"Period Desc.": r"\s\d{4}$"}, {"Period Desc.": ""}, regex=True)
    data_rename_period_desc = data_drop_period_year.rename(columns={"Period Desc.": "Month"})
    data_no_duplicate_columns = data_rename_period_desc.drop(columns=["Classification", "Trade Flow Code", "Reporter Code", "Commodity Code", "Commodity", "Partner"])
    return data_no_duplicate_columns

def get_clean_food_dataset():
    data = get_raw_food_dataset_df()
    data = data.drop(columns = ['Classification', 'Partner Code', 'Reporter Code', 'Unnamed: 0'])
    data = abbreviate_countries(data, 'Reporter')
    data = abbreviate_countries(data, 'Partner')
    data = data.rename(columns={"Year": "Date"})
    return data

def get_race_dataset_df():
    raw_data = pd.read_csv(get_race_dataset_path())
    raw_data["Period"] = pd.to_datetime(raw_data["Period"], format="%Y%m")
    columns_to_drop = [
        "Unnamed: 0",
        "Trade Flow Code",
        "Reporter Code",
        "Partner",
        "Year",
        "Period Desc."
    ]
    drop_columns = raw_data.drop(columns=columns_to_drop)
    data_no_junk = drop_columns.dropna(axis=1, how="all")
    data_no_zeros = data_no_junk.loc[:, (data_no_junk != 0).any(axis=0)]
    data_period_index = data_no_zeros.set_index("Period")
    return data_period_index

def get_race_dataset_2010123_df():
    return pd.read_csv(f"{get_dataset_dir()}/2010_123.csv")

## / End Week 1, 2

## Week3

def get_dataset_df(weekNumber, fileName):
    return pd.read_csv(f"{get_dataset_dir()}/week{weekNumber}/{fileName}.csv")

def merge_income_index_column(df, years):
    df['Income_Index'] = 0.0
    for i, val in enumerate(years):
        col = 'ii' + str(val)
        if(i == 0):
            df['Income_Index'] = np.where((df['Year'] <=val), df[col], df['Income_Index'])
        elif (i == (len(years) - 1)):
            prev = years[i-1]
            df['Income_Index'] = np.where((df['Year'] > prev), df[col], df['Income_Index'])
        else:
            prev = years[i-1]
            df['Income_Index'] = np.where((df['Year'] > prev) & (df['Year'] <= val), df[col], df['Income_Index'])
    return df

## End Week3

## General utilities:

# Thanks https://stackoverflow.com/a/45846841
def human_format(num, pos=None):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def get_2020_months():
    return ["January", "February", "March", "April", "May", "June", "July", "August", "September"]

def abbreviate_countries(df, col_name):
    #populate country abbreviation dictionary for replacement
    countries = {}
    for country in pycountry.countries:
        countries[country.name] = country.alpha_3
    countries['China, Hong Kong SAR'] = 'HKG'
    countries['China (Hong Kong SAR)'] = 'HKG'
    countries['Hong Kong, China (SAR)'] = 'HKG'
    countries['China, Macao SAR'] = 'MAC'
    countries['Macao, China (SAR)'] = 'MAC'
    countries['Rep. of Korea'] = 'KOR'
    countries['Republic of Korea'] = 'KOR'
    countries['Korea (Republic of)'] = 'KOR'
    countries['Korea (Rep. of)'] = 'KOR'
    # NB: this is North Korea
    countries['Democratic People\'s Republic of Korea'] = 'PRK'
    countries['Korea, Dem. P.R. of'] = 'PRK'
    countries['Congo (Democratic Republic of the)'] = 'COD'
    countries['Congo, Democratic Republic'] = 'COD'
    countries['Democratic Republic of the Congo'] = 'COD'
    countries['DR Congo'] = 'COD'
    countries['Vietnam'] = 'VNM'
    countries['Bolivia (Plurinational State of)'] = 'BOL'
    countries['United Republic of Tanzania'] = 'TZA'
    countries['Iran (Islamic Republic of)'] = 'IRN'
    countries['Iran'] = 'IRN'
    countries['State of Palestine'] = 'PSE'
    countries['Republic of Moldova'] = 'MDA'
    countries['Venezuela (Bolivarian Republic of)'] = 'VEN'
    countries['Bolivia'] = 'BOL'
    countries["Cote d'Ivoire"] = 'CIV'
    countries['Czech Republic'] = 'CZE'
    countries['Guinea Bissau'] = 'GNB'
    countries['Lao PDR'] = 'LAO'
    countries['Macedonia (TFYR)'] = 'MKD'
    countries['Micronesia (Federated States of)'] = 'FSM'
    countries['Moldova'] = 'MDA'
    countries['North Korea'] = 'PRK'
    countries['Occupied Palestinian Territory'] = '000'
    countries['South Korea'] = 'KOR'
    countries['Swaziland'] = 'SWZ'
    countries['Taiwan'] = 'TWN'
    countries['Tanzania'] = 'TZA'
    countries['United States of America'] = 'USA'
    countries['Venezuela'] = 'VEN'
    countries['Ireland, Republic of'] = 'IRL'
    countries['Bosnia & Herzegovina'] = 'BIH'
    countries['Eswatini (Swaziland)'] = 'SWZ'
    countries['Brunei'] = 'BRN'
    countries['Trinidad & Tobago'] = 'TTO'
    countries['Yemen, Republic of'] = 'YEM'
    countries['Russia'] = 'RUS'
    countries['Ivory Coast'] = 'CIV'
    countries['Syria'] = 'SYR'
    countries['Laos'] = 'LAO'
    countries['East Timor'] = 'TLS'

    #specify concatenate countries dictionary to needed columns
    con_map = {col_name: countries}
    df.replace(con_map, inplace=True)
    return df
