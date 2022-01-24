# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:36:34 2020

@author: riema
"""
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from customscripts import utils
from customscripts import configuration


formatter = ticker.FuncFormatter(utils.human_format)

blue_patch = mpatches.Patch(color='blue', label='Base Metals')
orange_patch = mpatches.Patch(color='orange', label='Chemicals and Plastics')
green_patch = mpatches.Patch(color='green', label='Guns')
red_patch = mpatches.Patch(color='red', label='Minerals')
industrial_handle = [blue_patch, orange_patch, green_patch, red_patch]

high_populated = ['CHN', 'IND', 'USA', 'BRA']
medium_populated = ['PAK', 'NGA', 'JPN', 'MEX']
low_populated = ['VNM', 'GER', 'EGY', 'TUR', 'THA', 'KOR', 'HKG']
populated_level = [high_populated, medium_populated, low_populated]

high_developed = ['GER', 'JPN', 'KOR', 'USA'] # with HKG excluded
medium_developed = ['BRA', 'MEX', 'THA', 'TUR'] # with CHN excluded
low_developed = ['EGY', 'NGA', 'PAK', 'VNM'] # with IND excluded
developed_level = [high_developed, medium_developed, low_developed]

# Pycharts Pie Import
from pyecharts.globals import CurrentConfig, OnlineHostType   # avoid no graph
from pyecharts import options as opts   # configuration
from pyecharts.charts import Pie
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType   # theme

def annual_cate_development(data):
    """
    plot the annual trade value of reporters for each category, with import trade value above the x-axis, export trade value below the x-axis
    countries are grouped by development levels
    """

    cat_p = ['High', 'Medium', 'Low']
    yticklabels = [['', '1T', '800B', '600B', '400B', '200B', '0','200B', '400B', '600B', '' ], 
                   ['', '100B', '50B', '0','50B', '100B', '150B', '' ], 
                   ['', '50B', '0','50B', '100B', '150B', '200B',  '250B','' ] ]
    for order, level in enumerate(cat_p):
        input_data = data.loc[data['Development Level'] == level]

        fig, ax = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
        ax_unpack = ax.ravel()
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.3, top=0.85)
        fig.suptitle("Annual trade value per capita of different categories for reporters in " + cat_p[order] +" development level", fontsize=15);

        reporter_names = input_data['Reporter ISO'].unique()

        reporter_category_value_data = input_data.groupby(['Year', 'Reporter ISO','Trade Flow', 'Category']) \
            .agg(Trade_Value_per_Capita=('Trade Value per capita', "sum")) \
            .reset_index() \
            .sort_values(by=['Year', 'Category'], ascending=True)
        
        # reverse the export trade value in the opposite direction
        # reporter_category_value_data['Trade_Value_per_Capita'] = reporter_category_value_data['Trade_Value_per_Capita'] \
        #     .mask(reporter_category_value_data['Trade Flow'] == 'Import', -reporter_category_value_data['Trade_Value_per_Capita'])

        for count, reporter in enumerate(reporter_names):

            # report_import_data = reporter_category_value_data.loc[reporter_category_value_data['Reporter ISO'] == reporter] \
            #     .loc[reporter_category_value_data['Trade Flow'] == 'Import']
            # sns.barplot(x=report_import_data['Year'], y=report_import_data['Trade_Value_per_Capita'], hue='Category', data=report_import_data, ax=ax_unpack[count], alpha=0.5)
            
            # report_export_data = reporter_category_value_data.loc[reporter_category_value_data['Reporter ISO'] == reporter] \
            #     .loc[reporter_category_value_data['Trade Flow'] == 'Export']
            # sns.barplot(x=report_export_data['Year'], y=report_export_data['Trade_Value_per_Capita'], hue='Category', data=report_export_data, ax=ax_unpack[count])

            report_net_value = reporter_category_value_data.loc[reporter_category_value_data['Reporter ISO'] == reporter] \
                .groupby(['Year', 'Category']) \
                .agg(Net_Value=("Trade_Value_per_Capita", "sum")) \
                .reset_index() \
                .sort_values(by=['Year', 'Category'], ascending=True)
            sns.lineplot(x=report_net_value['Year'], y=report_net_value['Net_Value'], hue='Category', data=report_net_value, ax=ax_unpack[count])

            ax_unpack[count].set_title('Trade value of  ' + reporter, fontsize=15)
            ax_unpack[count].grid(color='k', linestyle='--', alpha=0.5);
            ax_unpack[count].yaxis.set_label_text("Importr / Export trade value (US$)", fontsize=12)
            ax_unpack[count].xaxis.set_tick_params(rotation=60)
            ax_unpack[count].yaxis.set_major_formatter(formatter)
            # ax_unpack[count].legend(handles=industrial_handle)

        ax_unpack[0].set_yticklabels(yticklabels[order])

# Here we try to apply the method for stock data to the trade value data. Since the trade value every year is given, we need to calibrate 
# the parameters $\lambda \equiv \{A, B, p\} $ using the Baum-Welch algorithm to find a local maximizer $\lambda ^*$ of the probability function $P(O|\lambda)$

# We assume that the distribution corresponding with each hidden state is a Gaussian distribution; therefore, the number of parameters 
# $k = N^2 + 2N + 1$, where N is numbers of states used in the HMM.

# Choosing a number of hidden states for the HMM is a critical task. In the section, we use four common criteria: the AIC, the BIC, the HQC, 
# and the CAIC to evaluate the performances of HMM with different numbers of states. These criteria are suitable for HMM because, in the model
#  training algorithm, the Baum–Welch Algorithm, the EM method was used to maximize the log-likelihood of the model. We limit numbers of states
#   from two to six to keep the model simple and feasible to stock prediction.

def value_per_capita_development(data, development_args):
    """
    plot the annual trade value per capita of reporters for each category, with import trade value above the x-axis, export trade value below the x-axis
    countries are grouped by development levels
    """
    cat_p = ['high', 'medium', 'low']
    for order, level in enumerate(developed_level):
        input_data = data.loc[data['Reporter'].isin(level)]

        fig, ax = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
        ax_unpack = ax.ravel()
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.3, top=0.85)
        fig.suptitle("Annual trade value per capita of different categories for reporters in " + cat_p[order] +" development level", fontsize=15);

        reporter_names = input_data['Reporter'].unique().tolist()

        reporter_category_value_data = input_data.groupby(['Time', 'Reporter', 'Category Code']) \
            .agg(Trade_Value_per_Capita=('Trade Value per capita', "sum")) \
            .reset_index() \
            .sort_values(by=['Time', 'Category Code'], ascending=True)
        
        for count, reporter in enumerate(reporter_names):

            plot_data = reporter_category_value_data.loc[reporter_category_value_data['Reporter'] == reporter]
            sns.lineplot(x=plot_data['Time'], y=plot_data['Trade_Value_per_Capita'], hue='Category Code', data=plot_data, ax=ax_unpack[count])

            ax_unpack[count].set_title('Trade value of  ' + reporter, fontsize=15)
            ax_unpack[count].grid(color='k', linestyle='--', alpha=0.5);
            ax_unpack[count].yaxis.set_label_text("Importr trade value (US$)", fontsize=12)
            ax_unpack[count].xaxis.set_tick_params(rotation=90)
            ax_unpack[count].set_yscale('log')