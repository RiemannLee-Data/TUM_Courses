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


def annual_cate_development(data, development_args):
    """
    plot the annual trade value of reporters for each category, with export trade value above the x-axis, import trade value below the x-axis
    net trade value draw by the lines, and the countries are grouped by development levels
    """    
    
    input_data = data.loc[data['Development Level'] == development_args]

    fig, ax = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.3, top=0.85)
    fig.suptitle("Annual trade value per capita of different categories for reporters in " + development_args + " development level", fontsize=15);

    reporter_names = input_data['Reporter ISO'].unique()

    reporter_category_value_data = input_data.groupby(['Year', 'Reporter ISO','Trade Flow', 'Category']) \
        .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
        .reset_index() \
        .sort_values(by=['Year', 'Category'], ascending=True)
        
    # reverse the import trade value in the opposite direction
    reporter_category_value_data['Value_per_Capita'] = reporter_category_value_data['Value_per_Capita'] \
        .mask(reporter_category_value_data['Trade Flow'] == 'Import', -reporter_category_value_data['Value_per_Capita'])

    for count, reporter in enumerate(reporter_names):

        report_import_data = reporter_category_value_data.loc[reporter_category_value_data['Reporter ISO'] == reporter] \
            .loc[reporter_category_value_data['Trade Flow'] == 'Import']
        IM = sns.barplot(x=report_import_data['Year'], y=report_import_data['Value_per_Capita'], hue='Category', data=report_import_data, ax=ax_unpack[count], alpha=0.5)
        

        report_export_data = reporter_category_value_data.loc[reporter_category_value_data['Reporter ISO'] == reporter] \
            .loc[reporter_category_value_data['Trade Flow'] == 'Export']
        EX = sns.barplot(x=report_export_data['Year'], y=report_export_data['Value_per_Capita'], hue='Category', data=report_export_data, ax=ax_unpack[count])
        

        report_net_value = reporter_category_value_data.loc[reporter_category_value_data['Reporter ISO'] == reporter] \
            .groupby(['Year', 'Category']) \
            .agg(Net_Value=("Value_per_Capita", "sum")) \
            .reset_index() \
            .sort_values(by=['Year', 'Category'], ascending=True)
        NET = sns.lineplot(x=report_net_value['Year'], y=report_net_value['Net_Value'], hue='Category', data=report_net_value, ax=ax_unpack[count])
        NET.legend_.remove()
        
        # remove the repeated labels to only one unique legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_unpack[count].legend(by_label.values(), by_label.keys())

        ax_unpack[count].set_title('Trade value of  ' + reporter, fontsize=15)
        ax_unpack[count].grid(color='k', linestyle='--', alpha=0.5);
        ax_unpack[count].yaxis.set_label_text("Import / Export trade value per capita (US$)", fontsize=12)
        ax_unpack[count].xaxis.set_tick_params(rotation=60)




def reporter_category_pyramid(data, reporter_iso):

    """
    plot the annual trade value of specific reporter for each category in pyramid graph, with import trade value locate in the left part, export in the right part
    """
    input_data = data.loc[data['Reporter ISO'] == reporter_iso] \

    fig, ax = plt.subplots(1, 4, figsize=(25, 5))
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.3, top=0.85)
    fig.suptitle("Annual trade value of different categories for " + reporter_iso, fontsize=15);

    cate_names = input_data['Category'].unique()

    reporter_category_value_data = input_data.groupby(['Year','Trade Flow', 'Category']) \
            .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
            .reset_index() \
            .sort_values(by=['Year', 'Category'], ascending=False)
        
    # reverse the import trade value in the opposite direction
    reporter_category_value_data['Value_per_Capita'] = reporter_category_value_data['Value_per_Capita'] \
        .mask(reporter_category_value_data['Trade Flow'] == 'Import', -reporter_category_value_data['Value_per_Capita'])

    for count, industrial_category in enumerate(cate_names):

        category_import_data = reporter_category_value_data.loc[reporter_category_value_data['Category'] == industrial_category] \
            .loc[reporter_category_value_data['Trade Flow'] == 'Import']
        sns.barplot(y=category_import_data['Year'], x=category_import_data['Value_per_Capita'], data=category_import_data, ax=ax_unpack[count], alpha=0.5)
            
        category_export_data = reporter_category_value_data.loc[reporter_category_value_data['Category'] == industrial_category] \
            .loc[reporter_category_value_data['Trade Flow'] == 'Export']
        sns.barplot(y=category_export_data['Year'], x=category_export_data['Value_per_Capita'], data=category_export_data, ax=ax_unpack[count])

        ax_unpack[count].set_title('Trade value of  ' + industrial_category, fontsize=15)
        # ax_unpack[count].grid(color='k', linestyle='--', alpha=0.5);
        ax_unpack[count].xaxis.set_label_text("Import / Export Trade Value per capita", fontsize=12)
        # ax_unpack[count].yaxis.set_tick_params(rotation=60)
        ax_unpack[count].xaxis.set_major_formatter(formatter)




############################################################################################################################################
def plot_pie_diagram(data, reporter, year):
    """
    plot pie diagram usng pyecharm, don't think it a good way since we can only see the percentage at the first glance
    """

    reporter_data = data.loc[data['Reporter ISO'] == reporter] \
        .groupby(['Year', 'Trade Flow', 'Category']) \
        .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
        .reset_index() \
        .sort_values(by=['Year', 'Category'], ascending=True)

    reporter_pie = Pie() \
    .set_global_opts(
        xaxis_opts = opts.AxisOpts(is_show = False), yaxis_opts = opts.AxisOpts(is_show = False), 
        legend_opts = opts.LegendOpts(is_show = True), title_opts = opts.TitleOpts(title = "Trade Change of " + reporter), 
    ) \
    .set_series_opts(
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"),
        label_opts=opts.LabelOpts(is_show=True)
    )
    center_loc=[["25%", "50%"], ["75%", "50%"]]

    for count, y in enumerate(year):
        data_y = reporter_data.loc[reporter_data['Year'] == y]
        data_y_import = data_y.loc[data_y['Trade Flow'] == 'Import']
        import_data_y = [list(z) for z in zip(data_y_import['Category'].unique().tolist(), data_y_import['Value_per_Capita'].unique().tolist())]
        reporter_pie.add(series_name="Import Trade Value", center=center_loc[count], data_pair=import_data_y, radius=["25%", "40%"], label_opts=opts.LabelOpts(position='inner') )

        data_y_export = data_y.loc[data_y['Trade Flow'] == 'Export']
        export_data_y = [list(z) for z in zip(data_y_export['Category'].unique().tolist(), data_y_export['Value_per_Capita'].unique().tolist())]
        reporter_pie.add(series_name="Export Trade Value", center=center_loc[count], radius=["40%", "60%"], data_pair=export_data_y )

    return reporter_pie
############################################################################################################################################



def reporter_single_category_value(data, development_args, category_args):
    """
    plot the annual trade value of reporters for certain category, with import trade value above the x-axis, export trade value below the x-axis
    countries are grouped by development levels
    """
    for _, level in enumerate(development_args):
        input_data = data.loc[data['Development Level'] == level]

        fig, ax = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
        ax_unpack = ax.ravel()
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.3, top=0.8)
        fig.suptitle("Annual trade value of " + category_args + " for reporters in " + level + " development level", fontsize=15);

        reporter_names = input_data['Reporter ISO'].unique()

        reporter_category_value_data = input_data.groupby(['Year', 'Reporter ISO','Trade Flow', 'Category']) \
            .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
            .reset_index() \
            .sort_values(by=['Year', 'Category'], ascending=True)

        for count, reporter in enumerate(reporter_names):

            report_data = reporter_category_value_data.loc[reporter_category_value_data['Reporter ISO'] == reporter] \
                .loc[reporter_category_value_data['Category'] == category_args]
            sns.barplot(x=report_data['Year'], y=report_data['Value_per_Capita'], hue='Trade Flow', data=report_data, ax=ax_unpack[count])
            
            ax_unpack[count].set_title('Guns Trade value of  ' + reporter, fontsize=15)
            ax_unpack[count].grid(color='k', linestyle='--', alpha=0.5);

            ax_unpack[count].yaxis.set_label_text("Trade Value per capita", fontsize=12)
            ax_unpack[count].xaxis.set_tick_params(rotation=60)
            ax_unpack[count].yaxis.set_major_formatter(formatter)




def development_category_value(data, development_args):
    """
    plot the annual trade value for all countries in single development level, with 2 subplots showing the export and import trade value for each categoryS.
    """
    import_data = data.loc[data['Development Level'] == development_args]

    developemnt_category_value_data = import_data.groupby(['Year', 'Trade Flow', 'Category']) \
            .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
            .reset_index()
    
    flow_direction = developemnt_category_value_data['Trade Flow'].unique().tolist()

    fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.3, top=0.85)
    fig.suptitle("Annual trade value per capita of different categories for " + development_args +" development level", fontsize=15);

    for count, flow in enumerate(flow_direction):

        input_data = developemnt_category_value_data.loc[developemnt_category_value_data['Trade Flow'] == flow]
        sns.lineplot(x=input_data['Year'], y=input_data['Value_per_Capita'], hue='Category', data=input_data, ax=ax_unpack[count], markers=True, style='Category')
        
        ax_unpack[count].grid(color='k', linestyle='--', alpha=0.5);
        ax_unpack[count].set_title(flow + " Trade Value per capita", fontsize=12)
        ax_unpack[count].xaxis.set_tick_params(rotation=60)
        ax_unpack[count].yaxis.set_major_formatter(formatter)
    



def import_value_pattern(data_1, data_2, development_args):
    """
    Compare the import trade value of food and industrial product in different development level, then we can analyze the trade pattern.
    """

    import_data = [ data_1.loc[data_1['Development Level'] == development_args], data_2.loc[data_2['Development Level'] == development_args] ]
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.3, top=0.85)
    fig.suptitle("Import trade value for food and industrial products", fontsize=15);

    cat = ['food', 'indutrial products']

    for count, data_input in enumerate(import_data):
        group_data = data_input.groupby(['Year', 'Trade Flow', 'Category']) \
            .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
            .reset_index()
        input_data = group_data.loc[group_data['Trade Flow'] == 'Import'] 
        sns.lineplot(x=input_data['Year'], y=input_data['Value_per_Capita'], hue='Category', data=input_data, ax=ax[count], markers=True, style='Category')
        ax[count].grid(color='k', linestyle='--', alpha=0.5);
        ax[count].set_title("Import trade value of " + cat[count], fontsize=12)
        ax[count].xaxis.set_tick_params(rotation=60)
        ax[count].yaxis.set_major_formatter(formatter)
        ax[count].legend(framealpha=0.3)



def populated_value_pattern(data_1, data_2, populated_args):
    """
    Compare the import trade value of food and industrial product in different population level, then we can again analyze the trade pattern.
    The difference with the above function is that we need to define the population level here by ourself.
    """

    cat_p = ['high', 'medium', 'low']
    cat_c = ['food', 'industrial products']
    for num, level in enumerate(populated_args):
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax_unpack = ax.ravel()
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3, top=0.85)
        fig.suptitle("Import trade value for food and industrial products of " + cat_p[num] + " populated countries", fontsize=15);
        
        data = [ data_1.loc[data_1['Reporter ISO'].isin(level)], data_2.loc[data_2['Reporter ISO'].isin(level)] ]

        for count, data_input in enumerate(data):
            group_data = data_input.groupby(['Year', 'Trade Flow', 'Category']) \
                .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
                .reset_index()
            input_data = group_data.loc[group_data['Trade Flow'] == 'Import'] 
            sns.lineplot(x=input_data['Year'], y=input_data['Value_per_Capita'], hue='Category', data=input_data, ax=ax[count], markers=True, style='Category')
            ax[count].grid(color='k', linestyle='--', alpha=0.5);
            ax[count].set_title("Import trade value of " + cat_c[count], fontsize=12)
            ax[count].xaxis.set_tick_params(rotation=60)
            ax[count].yaxis.set_major_formatter(formatter)




def populated_value_category(data, populated_args, category_args):
    """
    Compare the trade value of a single category in different population levels.
    """
    cat_p = ['high', 'medium', 'low']
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3, top=0.85)
    fig.suptitle("Trade Value of " + category_args + " in different populated level");
    

    for count, country in enumerate(populated_args):
        sum_value = data.loc[data['Reporter ISO'].isin(country)] \
            .groupby(['Year', 'Trade Flow', 'Category']) \
            .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
            .reset_index()
        input_data = sum_value.loc[sum_value['Category'] == category_args]
        sns.lineplot(x=input_data['Year'], y=input_data['Value_per_Capita'], hue='Trade Flow', data=input_data, ax=ax[count], markers=True, style='Trade Flow')
        ax[count].grid(color='k', linestyle='--', alpha=0.5);
        ax[count].set_title("Trade value for guns of " + cat_p[count] + " populated countries")
        ax[count].xaxis.set_tick_params(rotation=60)
        ax[count].yaxis.set_major_formatter(formatter)



def reporter_single_category_population(data, population_args, category_args):
    level = ['high', 'medium', 'low']

    for num, pop in enumerate(population_args):
        input_data = data.loc[data['Reporter ISO'].isin(pop)]

        fig, ax = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
        ax_unpack = ax.ravel()
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.3, top=0.8)
        fig.suptitle("Annual trade value of " + category_args + " for reporters in " + level[num] + " population level", fontsize=15);

        reporter_names = input_data['Reporter ISO'].unique()

        reporter_category_value_data = input_data.groupby(['Year', 'Reporter ISO','Trade Flow', 'Category']) \
            .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
            .reset_index()

        for count, reporter in enumerate(reporter_names):

            report_data = reporter_category_value_data.loc[reporter_category_value_data['Reporter ISO'] == reporter] \
                .loc[reporter_category_value_data['Category'] == category_args]
            sns.barplot(x=report_data['Year'], y=report_data['Value_per_Capita'], hue='Trade Flow', data=report_data, ax=ax_unpack[count])
            
            ax_unpack[count].set_title('Trade value of  ' + reporter, fontsize=15)
            ax_unpack[count].grid(color='k', linestyle='--', alpha=0.5);

            ax_unpack[count].yaxis.set_label_text("Trade Value per capita", fontsize=12)
            ax_unpack[count].xaxis.set_tick_params(rotation=60)
            ax_unpack[count].yaxis.set_major_formatter(formatter)



def top_partner_category_value(data):
    """
    Choose the top 6 partners according to their total trade value over 20 years, and then show these partners' annual import and export trade value.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.4, top=0.8)

    fig.suptitle("Annual trade value of different industrial product of 3 top partners", fontsize=15);

    partner_category_value_data = data.groupby(['Year', 'Partner ISO', 'Category', 'Trade Flow']) \
        .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
        .reset_index()

    top_partners = partner_category_value_data.groupby(['Partner ISO']) \
        .agg(Total_Value=('Value_per_Capita', 'sum')) \
        .sort_values(by='Total_Value', ascending=False)[1:4] \
        .index \
        .tolist()

    partner_category_value_data['Value_per_Capita'] = partner_category_value_data['Value_per_Capita'] \
            .mask(partner_category_value_data['Trade Flow'] == 'Import', -partner_category_value_data['Value_per_Capita'])


    for count, partner in enumerate(top_partners):

        partner_import_data = partner_category_value_data.loc[partner_category_value_data['Partner ISO'] == partner] \
            .loc[partner_category_value_data['Trade Flow'] == 'Import'] \
            .sort_values(by=['Year', 'Category'], ascending=True)
        sns.barplot(x=partner_import_data['Year'], y=partner_import_data['Value_per_Capita'], hue='Category', data=partner_import_data, ax=ax_unpack[count], alpha=0.5)

        partner_export_data = partner_category_value_data.loc[partner_category_value_data['Partner ISO'] == partner] \
            .loc[partner_category_value_data['Trade Flow'] == 'Export'] \
            .sort_values(by=['Year', 'Category'], ascending=True)
        sns.barplot(x=partner_export_data['Year'], y=partner_export_data['Value_per_Capita'], hue='Category', data=partner_export_data, ax=ax_unpack[count])

        ax_unpack[count].grid(color='k', linestyle='--', alpha=0.5);
        ax_unpack[count].set_title('Trade value of different industrial product of ' + partner, fontsize=12)
        
        # remove the repeated labels to only one unique legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_unpack[count].legend(by_label.values(), by_label.keys())

        ax_unpack[count].grid(color='k', linestyle='--');
        ax_unpack[count].xaxis.set_tick_params(rotation=60)
        ax_unpack[count].yaxis.set_label_text("Import / Export Value (US$)", fontsize=12)



def bottom_partner_category_value(data):
    """
    Choose the top 3 bottom according to their total trade value over 20 years, and then show these partners' annual import and export trade value.
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 5), sharey=True, sharex=True)
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2, top=0.9)

    fig.suptitle("Annual trade value of different industrial product of 3 bottom partners", fontsize=15);
    
    partner_category_value_data = data.groupby(['Year', 'Partner ISO', 'Category', 'Trade Flow']) \
        .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
        .reset_index()

    bottom_partners = partner_category_value_data.groupby(['Partner ISO']) \
        .agg(Total_Value=('Value_per_Capita', 'sum')) \
        .sort_values(by='Total_Value', ascending=True)[0:3] \
        .index \
        .tolist()

    partner_category_value_data['Value_per_Capita'] = partner_category_value_data['Value_per_Capita'] \
            .mask(partner_category_value_data['Trade Flow'] == 'Export', -partner_category_value_data['Value_per_Capita'])

    for count, partner in enumerate(bottom_partners):

        partner_import_data = partner_category_value_data.loc[partner_category_value_data['Partner ISO'] == partner] \
            .loc[partner_category_value_data['Trade Flow'] == 'Import'] \
            .sort_values(by=['Year', 'Category'], ascending=True)
        sns.barplot(x=partner_import_data['Year'], y=partner_import_data['Value_per_Capita'], hue='Category', data=partner_import_data, ax=ax_unpack[count])

        partner_export_data = partner_category_value_data.loc[partner_category_value_data['Partner ISO'] == partner] \
            .loc[partner_category_value_data['Trade Flow'] == 'Export'] \
            .sort_values(by=['Year', 'Category'], ascending=True)
        sns.barplot(x=partner_export_data['Year'], y=partner_export_data['Value_per_Capita'], hue='Category', data=partner_export_data, ax=ax_unpack[count])

        ax_unpack[count].grid(color='k', linestyle='--', alpha=0.5);
        ax_unpack[count].set_title('Trade value of different industrial product of ' + partner, fontsize=15)
        # ax_unpack[count].legend(handles=industrial_handle);
        ax_unpack[count].grid(color='k', linestyle='--');
        ax_unpack[count].xaxis.set_tick_params(rotation=60)
        ax_unpack[count].yaxis.set_major_formatter(formatter)
        ax_unpack[count].yaxis.set_label_text("Export / Import Value (US$)", fontsize=12)
        ax_unpack[count].set_yticklabels(['', '2M', '1.5M', '1M','500K', '0', '500K', '1M',  '' ])




def reporter_transaction_num(data):
    """
    plot number of transaction of each kind of food categories for each reporter
    """
    fig, ax = plt.subplots(5, 3, figsize=(20, 16), sharey=True)
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.4, top=0.95)
    fig.suptitle("Number of transactions of 4 kind industrial product for each reporter", fontsize=12);

    reporter_transaction_count = data[{'Year', 'Reporter ISO', 'Category', 'Trade Flow'}] \
        .value_counts() \
        .reset_index() \
        .sort_values(by=['Year', 'Category'], ascending=True)
    reporter_transaction_count = reporter_transaction_count.rename(columns={0: "Count"})
    reporter_names = data['Reporter ISO'].unique()

    reporter_transaction_count['Count'] = reporter_transaction_count['Count'] \
        .mask(reporter_transaction_count['Trade Flow'] == 'Export', -reporter_transaction_count['Count'])

    for count, reporter in enumerate(reporter_names):
        reporter_import_data = reporter_transaction_count.loc[reporter_transaction_count['Reporter ISO'] == reporter] \
            .loc[reporter_transaction_count['Trade Flow'] == 'Import']
        sns.barplot(x=reporter_import_data['Year'], y=reporter_import_data['Count'], hue='Category', data=reporter_import_data, ax=ax_unpack[count])

        reporter_export_data = reporter_transaction_count.loc[reporter_transaction_count['Reporter ISO'] == reporter] \
            .loc[reporter_transaction_count['Trade Flow'] == 'Export']
        sns.barplot(x=reporter_export_data['Year'], y=reporter_export_data['Count'], hue='Category', data=reporter_export_data, ax=ax_unpack[count],alpha=0.7)

        ax_unpack[count].yaxis.set_label_text("Number of transaction", fontsize=12)
        ax_unpack[count].grid(color='k', linestyle='--');
        ax_unpack[count].set_title('Number of transactions for industrial product of  ' + reporter, fontsize=12)
        ax_unpack[count].legend(handles=[blue_patch, orange_patch, green_patch, red_patch])
        ax_unpack[count].xaxis.set_tick_params(rotation=45)
        ax_unpack[count].set_yticklabels(['', '2000', '1000', '0','1000', '' ])




def partner_top_transaction_num(data):
    """
    plot the number of transaction of each kind of food categories for chosen top partners
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.35, top=0.9)
    fig.suptitle("Number of transactions of different industrial products for chosen 3 top partners", fontsize=15);

    # define some variables, then give it to the function
    partner_transaction_count = data[{'Year', 'Partner ISO', 'Category', 'Trade Flow'}] \
        .value_counts() \
        .reset_index() \
        .sort_values(by=['Year', 'Category'], ascending=True)
    partner_transaction_count = partner_transaction_count.rename(columns={0: "Count"})

    # pick the top 3 partner in the number of total transactions over 20 years
    partner_top_transaction = partner_transaction_count.groupby(['Partner ISO']).agg(Total_Count=("Count", "sum")) \
        .reset_index() \
        .sort_values(by='Total_Count', ascending=False)[1:4]['Partner ISO'] \
        .unique() \
        .tolist()

    partner_transaction_count['Count'] = partner_transaction_count['Count'] \
        .mask(partner_transaction_count['Trade Flow'] == 'Export', -partner_transaction_count['Count'])
        
    for count, partner in enumerate(partner_top_transaction):
        partner_import_data = partner_transaction_count.loc[partner_transaction_count['Partner ISO'] == partner] \
            .loc[partner_transaction_count['Trade Flow'] == 'Import']
        sns.barplot(x=partner_import_data['Year'], y=partner_import_data['Count'], hue='Category', data=partner_import_data, ax=ax_unpack[count])

        partner_export_data = partner_transaction_count.loc[partner_transaction_count['Partner ISO'] == partner] \
            .loc[partner_transaction_count['Trade Flow'] == 'Export']
        sns.barplot(x=partner_export_data['Year'], y=partner_export_data['Count'], hue='Category', data=partner_export_data, ax=ax_unpack[count], alpha=0.7)
        
        ax_unpack[count].yaxis.set_label_text("Export / Import transaction number", fontsize=15)
        ax_unpack[count].grid(color='k', linestyle='--');
        ax_unpack[count].set_title('Number of transactions for different industrial product of  ' + partner, fontsize=12)
        ax_unpack[count].xaxis.set_tick_params(rotation=60)
        
        ax_unpack[count].legend(handles=industrial_handle)
        ax_unpack[count].set_yticklabels(['', '150', '100', '50', '0', '50', '100', '' ])



def partner_bottom_transaction_num(data):
    """
    plot number of transaction of each kind of food categories for chosen bottom partners
    """

    fig, ax = plt.subplots(1, 3, figsize=(20, 5), sharey=True, sharex=True)
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.35, top=0.9)
    fig.suptitle("Number of transactions of different industrial products for chosen 6 bottom partners", fontsize=15);

    # define some variables, then give it to the function
    partner_transaction_count = data[{'Year', 'Partner ISO', 'Category', 'Trade Flow'}] \
        .value_counts() \
        .reset_index() \
        .sort_values(by=['Year', 'Category'], ascending=True)
    partner_transaction_count = partner_transaction_count.rename(columns={0: "Count"})

    # pick the bottom 6 partner in the number of total transactions over 20 years
    partner_bottom_transaction = partner_transaction_count.groupby(['Partner ISO']).agg(Total_Count=("Count", "sum")) \
        .reset_index() \
        .sort_values(by='Total_Count', ascending=True)[0:4]['Partner ISO'] \
        .unique() \
        .tolist()
    
    partner_transaction_count['Count'] = partner_transaction_count['Count'] \
        .mask(partner_transaction_count['Trade Flow'] == 'Export', -partner_transaction_count['Count'])

    for count, partner in enumerate(partner_bottom_transaction):

        partner_import_data = partner_transaction_count.loc[partner_transaction_count['Partner ISO'] == partner] \
            .loc[partner_transaction_count['Trade Flow'] == 'Import']
        sns.barplot(x=partner_import_data['Year'], y=partner_import_data['Count'], hue='Category', data=partner_import_data, ax=ax_unpack[count])

        partner_export_data = partner_transaction_count.loc[partner_transaction_count['Partner ISO'] == partner] \
            .loc[partner_transaction_count['Trade Flow'] == 'Export']
        sns.barplot(x=partner_export_data['Year'], y=partner_export_data['Count'], hue='Category', data=partner_export_data, ax=ax_unpack[count], alpha=0.7)

        ax_unpack[count].yaxis.set_label_text("Export / Import transaction number", fontsize=12)
        ax_unpack[count].grid(color='k', linestyle='--');
        ax_unpack[count].set_title('Number of transactions for different industrial products of  ' + partner, fontsize=15)
        ax_unpack[count].legend(handles=industrial_handle)
        ax_unpack[count].xaxis.set_tick_params(rotation=60)
        ax_unpack[count].set_yticklabels(['', '10', '5', '0', '5', '10', '' ])




def reporter_top_partner_trade_value(data, developement_args):
    """
    plot trade value of each reporter with its top partners
    since we share the y-axis, it's better to group the data by their development level
    """
    input_data = data.loc[data['Development Level'] == developement_args]

    fig, ax = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.25, top=0.8)
    fig.suptitle("Annual industrial product's trade value of top 5 partners for reporters in " + developement_args + " development level", fontsize=15);

    # choose top 5 partners by total trade values over 20 years
    reporter_partner_year_value = input_data.groupby(['Year', 'Reporter ISO', 'Partner ISO', 'Trade Flow']) \
        .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
        .reset_index()

    reporter_partner_order = reporter_partner_year_value \
        .groupby(['Reporter ISO', 'Partner ISO']) \
        .sum() \
        .reset_index() \
        .sort_values(by='Value_per_Capita', ascending=False)

    reporter_names = input_data['Reporter ISO'].unique()
    reporter_partner_year_value['Value_per_Capita'] = reporter_partner_year_value['Value_per_Capita'] \
        .mask(reporter_partner_year_value['Trade Flow']=='Export', -reporter_partner_year_value['Value_per_Capita'])


    for count, reporter in enumerate(reporter_names):
        top_partner = reporter_partner_order.loc[reporter_partner_order['Reporter ISO'] == reporter][1:6]['Partner ISO'].unique().tolist()
        
        reporter_top_partner = reporter_partner_year_value \
            .loc[reporter_partner_year_value['Reporter ISO'] == reporter] \
            .loc[reporter_partner_year_value['Partner ISO'].isin(top_partner)]

        import_data = reporter_top_partner.loc[reporter_top_partner['Trade Flow']== 'Import']
        sns.barplot(x=import_data['Year'], y=import_data['Value_per_Capita'], hue='Partner ISO', data=import_data, ax=ax_unpack[count])

        export_data = reporter_top_partner.loc[reporter_top_partner['Trade Flow']== 'Export']
        sns.barplot(x=export_data['Year'], y=export_data['Value_per_Capita'], hue='Partner ISO', data=export_data, ax=ax_unpack[count], alpha=0.8)

        ax_unpack[count].yaxis.set_major_formatter(formatter)
        ax_unpack[count].yaxis.set_label_text("Export / Import Trade Value per capita", fontsize=15)
        ax_unpack[count].xaxis.set_tick_params(rotation=60)
        ax_unpack[count].set_title('Top 5 partners of  ' + reporter, fontsize=15)
        ax_unpack[count].grid(color='k', linestyle='--');
        
    


def reporter_bottom_partner_trade_value(data, developement_args):

    """
    plot the trade value for the bottom partners of each reporter
    seems not promising......
    """

    input_data = data.loc[data['Development Level'] == developement_args]

    fig, ax = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.25, top=0.8)
    fig.suptitle("Annual industrial product's trade value of top 5 partners for reporters in " + developement_args + " development level", fontsize=15);

    # choose top 5 partners by total trade values over 20 years
    reporter_partner_year_value = input_data.groupby(['Year', 'Reporter ISO', 'Partner ISO', 'Trade Flow']) \
        .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
        .reset_index()

    reporter_partner_order = reporter_partner_year_value \
        .groupby(['Reporter ISO', 'Partner ISO']) \
        .sum() \
        .reset_index() \
        .sort_values(by='Value_per_Capita', ascending=True)

    reporter_names = input_data['Reporter ISO'].unique()
    reporter_partner_year_value['Value_per_Capita'] = reporter_partner_year_value['Value_per_Capita'] \
        .mask(reporter_partner_year_value['Trade Flow']=='Export', -reporter_partner_year_value['Value_per_Capita'])


    for count, reporter in enumerate(reporter_names):
        top_partner = reporter_partner_order.loc[reporter_partner_order['Reporter ISO'] == reporter][1:6]['Partner ISO'].unique().tolist()
        
        reporter_top_partner = reporter_partner_year_value \
            .loc[reporter_partner_year_value['Reporter ISO'] == reporter] \
            .loc[reporter_partner_year_value['Partner ISO'].isin(top_partner)]

        import_data = reporter_top_partner.loc[reporter_top_partner['Trade Flow']== 'Import']
        sns.barplot(x=import_data['Year'], y=import_data['Value_per_Capita'], hue='Partner ISO', data=import_data, ax=ax_unpack[count])

        export_data = reporter_top_partner.loc[reporter_top_partner['Trade Flow']== 'Export']
        sns.barplot(x=export_data['Year'], y=export_data['Value_per_Capita'], hue='Partner ISO', data=export_data, ax=ax_unpack[count], alpha=0.8)

        ax_unpack[count].yaxis.set_major_formatter(formatter)
        ax_unpack[count].yaxis.set_label_text("Export / Import Trade Value per capita", fontsize=15)
        ax_unpack[count].xaxis.set_tick_params(rotation=60)
        ax_unpack[count].set_title('Top 5 partners of  ' + reporter, fontsize=15)
        ax_unpack[count].grid(color='k', linestyle='--');



def num_transaction_of_reporter(input_data, order_direction):
    
    # number of transaction of each reporter with its top or bottom 10 partners

    fig, ax = plt.subplots(5, 3, figsize=(20, 15), sharey=True)
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.35, top=0.85)
    
    transaction_count = input_data[{'Reporter ISO', 'Partner ISO'}].value_counts(ascending=order_direction) \
        .reset_index() \
        .rename(columns={0: 'Num'})

    sns.set(style="darkgrid")
    reporter_names = input_data['Reporter ISO'].unique()

    if order_direction:
        title_name = 'Bottom'
    else:
        title_name = "Top"
        
    for count, reporter in enumerate(reporter_names):
        data_to_plot = transaction_count.loc[transaction_count['Reporter ISO'] == reporter][0:10]
        sns.barplot(x=data_to_plot['Partner ISO'], y=data_to_plot['Num'], data=data_to_plot, ax=ax_unpack[count])
        ax_unpack[count].yaxis.set_label_text("Number of transaction", fontsize=12)
        ax_unpack[count].set_title(title_name + '  10 partners of  ' + reporter, fontsize=12)
    
    fig.suptitle("Number of transactions over 10 years with " + title_name + " 10 partners for each reporter", fontsize=15);


def reporter_partner_detail(input_data, reporter, partner):

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax_unpack = ax.ravel()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.35, top=0.85)

    fig.suptitle("Trade details between " + reporter + " and " + partner, fontsize=15);

    reporter_partner_data = input_data.loc[input_data['Reporter ISO'] == reporter] \
        .loc[input_data['Partner ISO'] == partner]
    
    # trade value of each kind of category over the years
    data_1 = reporter_partner_data.groupby(['Year', 'Category', 'Trade Flow']) \
        .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
        .reset_index() \
        .sort_values(by=['Year', 'Category'], ascending=True)

    data_1['Value_per_Capita'] = data_1['Value_per_Capita'] \
        .mask(data_1['Trade Flow'] == 'Export', -data_1['Value_per_Capita'])

    import_data = data_1.loc[data_1['Trade Flow'] == 'Import']
    sns.barplot(x=import_data['Year'], y=import_data['Value_per_Capita'], data=import_data, hue='Category', ax=ax_unpack[0])

    export_data = data_1.loc[data_1['Trade Flow'] == 'Export']
    sns.barplot(x=export_data['Year'], y=export_data['Value_per_Capita'], data=export_data, hue='Category', ax=ax_unpack[0], alpha=0.5)

    ax_unpack[0].yaxis.set_major_formatter(formatter)
    ax_unpack[0].yaxis.set_label_text("Trade Value per capita", fontsize=12)
    ax_unpack[0].set_title('Trade value of each categories over 20 years', fontsize=13)
    ax_unpack[0].xaxis.set_tick_params(rotation=60)

    # trade proportion of each kind of category
    data_2 = reporter_partner_data.groupby(['Year', 'Category']) \
        .agg(Value_per_Capita=("Trade Value per capita", "sum")) \
        .groupby(level=0) \
        .apply(lambda x: 100 * x / float(x.sum())) \
        .unstack() \
        .fillna(0) \

    data_2.plot(kind="bar", stacked=True, figsize=(20, 5), ax=ax_unpack[1], rot=0)
    # ax_unpack[1].get_legend().remove()
    ax_unpack[1].yaxis.set_label_text("Category ratio in trade value", fontsize=12)
    ax_unpack[1].set_title('Trade value ratio of each categories over 20 years', fontsize=13)
    ax_unpack[1].xaxis.set_tick_params(rotation=60)

    # here not plot the trade value of a year in total, but with original lineplot with shaded area
    # Emmm... the aliasing effect is very obvious, try with another method
    sns.lineplot(x=reporter_partner_data['Year'], y=reporter_partner_data['Trade Value per capita'], data=reporter_partner_data, hue='Category', ax=ax_unpack[2])
    ax_unpack[2].yaxis.set_major_formatter(formatter)
    ax_unpack[2].yaxis.set_label_text("Trade Value per capita", fontsize=12)
    ax_unpack[2].set_title('Trade value of each categories over 20 years', fontsize=13)
    ax_unpack[2].xaxis.set_tick_params(rotation=60)


def print_report_partner_info(data):
    reporter_names = data['Reporter ISO'].unique()

    order_in_value = data.groupby(['Reporter ISO', 'Partner ISO', 'Trade Flow']) \
        .sum() \
        .reset_index() \
        .sort_values(by='Trade Value per capita', ascending=False)

    order_in_num = data[{'Reporter ISO', 'Partner ISO', 'Trade Flow'}].value_counts(ascending=False) \
        .reset_index()
        
    direction = ['Import', 'Export']
    
    print('#########################################################################################################')
    for count, direc in enumerate(direction):
        print("Let\'s focus on the " + direc + " trade value:")
        value_data = order_in_value.loc[order_in_value['Trade Flow'] == direc]
        num_data = order_in_num.loc[order_in_num['Trade Flow'] ==  direc]
        for count, reporter in enumerate(reporter_names):
            print("The first " + direc + f" partner of {reporter} is {value_data.loc[value_data['Reporter ISO'] == reporter][1:2]['Partner ISO'].values} in trade value, and {num_data.loc[num_data['Reporter ISO'] == reporter][1:2]['Partner ISO'].values} in transaction number over 20 years")
        print('#########################################################################################################')
