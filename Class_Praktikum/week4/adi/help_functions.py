import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from customscripts import utils
from customscripts import configuration

formatter = ticker.FuncFormatter(utils.human_format)

high_gdp = ["USA", "DEU", "JPN", "HKG", "KOR"]
medium_gdp = ["TUR", "MEX", "BRA", "CHN", "THA", ]
low_gdp = ["EGY", "NGA", "VNM", "IND", "PAK"]
gdp_level = [high_gdp, medium_gdp, low_gdp]
category_label = ["high", "medium", "low"]


def adjust_human_axis(ax, which="y"):
    if which == "y":
        ticks = ax.get_yticks()
    else:
        ticks = ax.get_xticks()

    labels = [utils.human_format(abs(tick)) for tick in ticks]
    if which == "y":
        ax.set_yticklabels(labels)
    else:
        ax.set_xticklabels(labels)


def get_category_name(category_code, codes):
    return codes[codes["Category Code"]
                 == str(category_code)].iloc[0]["Category"]

def adjust_year_axis(g, dates, which="x"):
    def adjust_for_ax(ax, which):
        if which == "x":
            ax.set_xticklabels(labels=dates, rotation=45, ha="right")
        else:
            ax.set_yticklabels(labels=dates, rotation=45, ha="right")

    if (type(g) is sns.axisgrid.FacetGrid):
        for ax in g.axes.flatten():
            adjust_for_ax(ax, which)
    else:
        # When we don't have a FacetGrid, we suppose we only have one plot,
        # so just one ax to work with
        adjust_for_ax(g, which)


def plot_average_gdp(data_gdp):
    ordered = data_gdp.groupby("Country")["GDP"].aggregate(
        np.median).reset_index().sort_values("GDP", ascending=False)["Country"]
    ax = sns.barplot(data=data_gdp, x="GDP", y="Country",
                     color="salmon", ci=None,
                     order=ordered)
    ax.set_title("GDP Average of countries", fontsize=15)
    adjust_human_axis(ax, "x")


def plot_trade_balance(data, codes, title, col="Category Code"):
    if col:
        g = sns.catplot(
            kind="bar",
            data=data,
            x="Year",
            y="Trade Value (US$)",
            hue="Trade Flow",
            col=col,
            sharex=False,
            sharey=False,
            ci=None  # No error bars
        )
    else:
        g = sns.barplot(
            data=data,
            x="Year",
            y="Trade Value (US$)",
            hue="Trade Flow",
            ci=None  # No error bars
        )

    # If you add a suptitle without adjusting the axis, the seaborn facet titles
    # overlap it.
    plt.subplots_adjust(top=0.9)
    if col:
        g.fig.suptitle(title, fontsize=15)

    x_dates = data["Year"].dt.strftime("%Y").sort_values().unique()

    adjust_year_axis(g, x_dates)

    patch = mpatches.Patch(color='purple', label='Net Export')
    def draw_sum_line(ax, title):
        children = [child for child in ax.get_children() if type(child)
                    is mpatches.Rectangle]
        children_data = [child.get_height()
                         for child in children if child.get_height() != 1.0]
        children_imports = [child for child in children_data if child < 0]
        children_exports = [child for child in children_data if child >= 0]
        children_len = len(children_imports)
        children_sum = [children_exports[i] + children_imports[i]
                        for i in range(children_len)]

        sns.lineplot(data=children_sum, color="purple", ax=ax)
        ax.axhline(0, color="lightgray")
        ax.set_title(title)
        # where some data has already been plotted to ax
        handles, _ = ax.get_legend_handles_labels()

        # handles is a list, so append manual patch
        handles.append(patch) 

        # plot the legend
        ax.legend(handles=handles)

    if col:
        for (category_code), ax in g.axes_dict.items():
            draw_sum_line(ax, get_category_name(category_code, codes))
            adjust_human_axis(ax)

        g._legend.remove()
        g.fig.tight_layout()
    else:
        draw_sum_line(g, title)
        adjust_human_axis(g)


def negative_imports_data(df):
    data_all = df.copy()
    data_all_imports = -1 * \
        data_all["Trade Value (US$)"][data_all["Trade Flow"] == "Import"]
    data_all["Trade Value (US$)"][data_all["Trade Flow"]
                                  == "Import"] = data_all_imports
    return data_all


def plot_trade_balance_country_level(df, codes, level):
    data_all = negative_imports_data(df)
    data_level = data_all[data_all["GDP Level"] == level]
    plot_trade_balance(data_level, codes,
                       "Annual trade value for " + level + " GDP countries")


def plot_trade_balance_country(df, codes, country):
    data_all = negative_imports_data(df)
    data_all_country = data_all[data_all["Reporter ISO"] == country]
    plot_trade_balance(data_all_country, codes,
                       "Annual trade value for " + country)


def plot_trade_balance_all_countries(df, codes):
    data_all = negative_imports_data(df)

    plot_trade_balance(
        data_all, codes, "Annual overall trade value for each category")


def plot_trade_balance_all_categories(df, codes):
    data_all = negative_imports_data(df)

    plot_trade_balance(data_all, codes, "Annual overall trade value", None)


def plot_gdp_countries_corr(data_gdp_wide, title):
    # Correlation between different countries GDP
    data_gdp_corr = data_gdp_wide.corr(method="pearson")
    data_gdp_corr_mask = np.triu(np.ones_like(data_gdp_corr, dtype=bool))

    # Generate a custom diverging colormap
    data_gdp_corr_cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(data_gdp_corr, mask=data_gdp_corr_mask, cmap=data_gdp_corr_cmap,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    g.set_title(title)
    plt.show()


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_correlations(df, n=5, ascending=True, abs=True):
    if abs:
        au_corr = df.corr().abs().unstack()
    else:
        au_corr = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(
        ascending=ascending)
    return au_corr[0:n]


def plot_corr_countries_level(data, codes, level):
    data_misc_grouped = data.groupby(["Year", "Reporter ISO", "Category Code", "GDP Level"])["Trade Value (US$)"].sum().reset_index()

    data_misc_grouped_level = data_misc_grouped[data_misc_grouped["GDP Level"] == level]
    data_misc_wide = data_misc_grouped_level.pivot_table(index="Year", columns="Category Code", values="Trade Value (US$)")

    data_gdp_corr = data_misc_wide.corr(method="pearson")
    data_gdp_corr_mask = np.triu(np.ones_like(data_gdp_corr, dtype=bool))

    # Generate a custom diverging colormap
    data_gdp_corr_cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(data_gdp_corr, mask=data_gdp_corr_mask, cmap=data_gdp_corr_cmap,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    ticks = g.get_xticklabels()

    labels = [tick.get_text() + " - " + get_category_name(tick.get_text(), codes) for tick in ticks];
    g.set_xticklabels(labels, rotation=90);
    g.set_yticklabels(labels, rotation=0);
    g.set_title("Correlation amongst category of " + level + " GDP countries")
    plt.show();

def plot_corr_product_category(data, codes, title = "Correlation amongst categories"):
    data_misc_grouped = data.groupby(["Year", "Reporter ISO", "Category Code", "GDP Level"])["Trade Value (US$)"].sum().reset_index()

    data_misc_wide = data_misc_grouped.pivot_table(index="Year", columns="Category Code", values="Trade Value (US$)")

    data_gdp_corr = data_misc_wide.corr(method="pearson")
    data_gdp_corr_mask = np.triu(np.ones_like(data_gdp_corr, dtype=bool))

    # Generate a custom diverging colormap
    data_gdp_corr_cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(data_gdp_corr, mask=data_gdp_corr_mask, cmap=data_gdp_corr_cmap,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    ticks = g.get_xticklabels()

    labels = [tick.get_text() + " - " + get_category_name(tick.get_text(), codes) for tick in ticks];
    g.set_xticklabels(labels, rotation=90);
    g.set_yticklabels(labels, rotation=0);
    g.set_title(title)
    plt.show();
