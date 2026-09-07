import polars as pl
from pathlib import Path
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as mpatches
import numpy as np
from text_process import *


def plot_map_ses(df, two_three_classes, ses_column, output_path):
    '''
    Given the df with socioeconomic class of each user and geography column, plot the grid map with colors according to the class
    df: df of users SES
    two_three_classes: 2 | 3 according to if we have 2 or 3 classes
    ses_column: name of the column containing the SES class value
    output_path: output path of the figure
    '''

    # lower, middle, high classes
    if two_three_classes == 2:

        # define colors
        color_map = {
            "lower": "blue",
            "high": "red"
        }

        legend_elements = [
        Patch(facecolor="blue", label="Low SES"),
        Patch(facecolor="red", label="High SES")
    ]
        
        df = df[df[ses_column] != 'medium']
        
    elif two_three_classes == 3:

        # define colors
        color_map = {
            "lower": "blue",
            "medium": "green",
            "high": "red"
        }

        legend_elements = [
        Patch(facecolor="blue", label="Low SES"),
        Patch(facecolor="green", label="Medium SES"),
        Patch(facecolor="red", label="High SES")
    ]
        
        df = df[df[ses_column] != 'medium']

    # map colors
    df["color"] = df[ses_column].map(color_map)

    print(df)

    # plot total
    fig, ax = plt.subplots(figsize=(10,10))

    df.plot(
        color=df["color"],
        edgecolor="black",
        linewidth=0.2,
        ax=ax
    )

    

    ax.legend(handles=legend_elements, title="SES")

    ax.axis('off')
    fig.savefig(output_path)


if __name__ == "__main__":

    # -------------------------------------------------------------------------------------------------------------
    # PLOT CELLS WITH SOCIOECONOMIC STATUS ANS USERS ON THE MAP COLOURED ACCORDING TO THEIR SOCIOECONOMIC STATUS
    # -------------------------------------------------------------------------------------------------------------

    # df with geographical info
    geo_users_path = '../geo_ses_files/FR/user_home_locations_and_attributes_200m.parquet'
    # df with users ses
    ses_users_path_ok = '../geo_ses_files/FR/users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE.parquet'
    users_path = '../geo_ses_files/FR/user_geo_income_200m_threshold_0.7_PARIS_LARGE_LARGE.parquet'

    # read files
    geo_users_df = gpd.read_parquet(geo_users_path)
    ses_users_ok_df = pd.read_parquet(ses_users_path_ok)[['user_id', 'SES_users']]
    users_df = pd.read_parquet(users_path)

    # merge df with ses users and selected users through threshold
    ses_users_df = pd.merge(ses_users_ok_df, users_df, on='user_id')

    print(len(ses_users_df))
    print(len(users_df))

    # set geometry column
    geometry_col = 'carreau_geometry' # 'home_geometry ! carreau_geometry
    # set number of classes
    num_classes = 3 # 2 | 3

    # output file
    file_name = '200m_paris_LARGE_LARGE_threshold_0.7_' + str(num_classes) + '_classes_' + geometry_col + '.png'
    map_output_path = '../geo_ses_files/FR/maps 200m/paris LARGE LARGE scaled/' + file_name

    # rename column for merging
    geo_users_df = geo_users_df.rename(columns={'userId': 'user_id'})

    # merge geo and ses df
    merged_df = geo_users_df.merge(ses_users_df, on='user_id')
    merged_df = merged_df.set_geometry(geometry_col)

    # create map
    plot_map_ses(merged_df, num_classes, 'SES_users', map_output_path)



    # ----------------------------------------------------------------------------------------------------
    # PLOT USERS AS POINTS in THEIR HOMW LOCATIONS 
    # ----------------------------------------------------------------------------------------------------


    # paths

    # geometry
    geo_users_path = '../geo_ses_files/FR/user_home_locations_and_attributes_200m.parquet'
    # users to plot
    ses_users_path_ok = '../geo_ses_files/FR/users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE.parquet'
    users_path = '../predicted users to plot/test_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv'
     # '../users_train_dev_test_tot.csv' -> tot users train + dev + test
     # #'../geo_ses_files/FR/user_geo_income_200m_threshold_0.7_PARIS_LARGE_LARGE.parquet' -> tot users ile de france
     # #'../predicted users to plot/A PRIORI all_features_with_length_vocab_negation_pluralization_mean_2_2_classes.csv' -> predicted a priori (best scores)
     # #'../predicted users to plot/test_test_paris_LARGE_LARGE_tot_errors_with_medium.csv' -> true values test set 2 and 3 classes
     # #'../predicted users to plot/test_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv' -> true values test set 2_2 classes    

    # output path
    maps_path = '../classification_data/FR/maps/users_ILE_DE_FRANCE_TRUE_2_2_classes_COL_COMPASS.png' # '../classification_data/FR/maps/users_ILE_DE_FRANCE_PRED_3_classes_COL_COMPASS.png' '../classification_data/FR/maps/users_ILE_DE_FRANCE_TRUE_2_2_classes_COL_COMPASS.png' '../classification_data/FR/maps/users_ILE_DE_FRANCE_TOT_TRAIN_DEV_TEST_COL.png'   #'../classification_data/FR/maps/users_ILE_DE_FRANCE_TOT_TOT_COL.png'

    # num of classes
    class_num = "2"

    # prediction or true
    true_pred = "true" # true | pred | "tot_train_dev_test" | tot


    # colors
    color_low = "#E8853A"   # arancione
    color_medium = "#2E9E76"   # verde
    color_high = "#7355A0"   # viola


    # create map of users sample

    # geometry
    geo_df = gpd.read_parquet(geo_users_path)
    geo_df = geo_df.rename(columns={'userId': 'user_id'})


    # plot test true and predicted
    # users to plot
    if true_pred == 'tot':
        users_df = pd.read_parquet(users_path)[['user_id']]
    elif true_pred == 'tot_train_dev_test':
        users_df = pd.read_csv(users_path).drop_duplicates()
    else:
        users_df = pd.read_csv(users_path).drop_duplicates(subset=['user_id'])


    # rename "label" into "SES_users" for the predicted ones (we can leave it also with the others, it has no efffect)
    users_df = users_df.rename(columns={'label': 'SES_users'})
    # converto into string for merging
    users_df['user_id'] = users_df['user_id'].astype(str)

    if true_pred == 'tot' or true_pred == 'tot_train_dev_test':
        #TODO only with real data plot (all the users, no test/pred files)
        ses_users_ok_df = pd.read_parquet(ses_users_path_ok)[['SES_users', 'user_id']]
        # merge df with ses users and selected users through threshold
        users_df = pd.merge(ses_users_ok_df, users_df, on='user_id')

    # create df of Paris
    tot_df = geo_df.merge(users_df, on='user_id')

    print(tot_df)

    #TODO only when we use test df with medium classs but we want to plot only the other two classes
    #

    # Paris lower, middle, high classes

    if class_num == '2' and true_pred == "true":

        tot_df = tot_df[tot_df['SES_users'] != 'medium']

        #TODO only when using pred df, which has 0 and 1 and 2
        if true_pred == "pred":
            tot_df['SES_users'] = tot_df['SES_users'].map({0: 'lower', 2: 'high'})

        color_map = {
                    "lower": color_low,
                    "high": color_high
                }

    elif class_num == '2' and true_pred == "pred":

        tot_df = tot_df[tot_df['SES_users'] != 'medium']

        #TODO only when using pred df, which has 0 and 1 and 2
        if true_pred == "pred":
            tot_df['SES_users'] = tot_df['SES_users'].map({0: 'lower', 1: 'high'})

        color_map = {
                    "lower": color_low,
                    "high": color_high
                }

    elif class_num == '2_2':
    
        #TODO only when using pred df, which has 0 and 1 and 2
        if true_pred == "pred":
            tot_df['SES_users'] = tot_df['SES_users'].map({0: 'lower', 1: 'high'})

        color_map = {
                    "lower": color_low,
                    "high": color_high
                }
            
    # define colors
    elif class_num == '3':

        #TODO only when using pred df, which has 0 and 1 and 2
        if true_pred == "pred":
            tot_df['SES_users'] = tot_df['SES_users'].map({0: 'lower', 1: 'medium', 2: 'high'})

        color_map = {
            "lower": color_low,
            "medium": color_medium,
            "high": color_high
        }
    
    # sort by SES class
    tot_df = tot_df.sort_values(by='SES_users', ascending=False)

    # map colors
    tot_df["color"] = tot_df['SES_users'].map(color_map)

    # convert colors to RGBA with your desired alpha
    rgba_colors = [mcolors.to_rgba(c, alpha=0.3) for c in tot_df["color"]]


    # add map in background

    # outer IdF communes
    communes = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/communes-version-simplifiee.geojson")

    # keep only Île-de-France
    idf = communes[communes["code"].str.startswith(("75","77","78","91","92","93","94","95"))].copy()

    # department code = first 2 digits of commune code
    idf["dep"] = idf["code"].str[:2]

    # dissolve communes into departments -> only department boundaries remain
    departments = idf.dissolve(by="dep")

    departments = departments.to_crs(tot_df.crs)

    # plot: outer communes + Paris arrondissements as grey background, points on top
    fig, ax = plt.subplots(figsize=(12,12))
    departments.plot(ax=ax, color="none", edgecolor="grey", linewidth=1.0, zorder=1)
    tot_df.plot(color=rgba_colors, markersize=7, alpha=0.5, ax=ax, zorder=2)
    ax.set_axis_off()


    # add scalebar

    # dx = length of one CRS unit in metres. If CRS is metric (e.g. Lambert-93), dx=1.
    scalebar = ScaleBar(
        dx=1, units="m", location="lower left",
        length_fraction=0.25, scale_loc="bottom",
        font_properties={"size": 15},
        box_alpha=0.6,
        bbox_to_anchor=(0.01, 0.05),
        bbox_transform=ax.transAxes
    )
    ax.add_artist(scalebar)


    # add compass

    def add_compass(ax, x=0.08, y=0.15, size=0.05):
        """Draw a north compass at axes-fraction (x, y)."""
        # circle
        circle = mpatches.Circle((x, y), size, transform=ax.transAxes,
                                facecolor="white", edgecolor="black",
                                linewidth=1.5, zorder=10)
        ax.add_patch(circle)

        # north-pointing triangle (filled top half)
        ax.annotate("", xy=(x, y + size*0.9), xytext=(x, y - size*0.5),
                    xycoords=ax.transAxes,
                    arrowprops=dict(facecolor="black", edgecolor="black",
                                    width=2, headwidth=10, headlength=12),
                    zorder=11)

        # "N" label above
        ax.text(x, y + size*1.3, "N", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, fontweight="bold",
                zorder=11)

    add_compass(ax, x=0.09, y=0.15, size=0.04)

    if class_num == "2" or class_num == "2_2":
        legend_elements = [
                    Patch(facecolor=color_low, label="Low SES"),
                    Patch(facecolor=color_high, label="High SES")
                ]
    else:
        legend_elements = [
            Patch(facecolor=color_low, label="Low SES"),
            Patch(facecolor=color_medium, label="Medium SES"),
            Patch(facecolor=color_high, label="High SES")
        ]

    ax.legend(handles=legend_elements, title="SES", fontsize=20, title_fontsize=22)
    ax.axis('off')

    plt.tight_layout()

    fig.savefig(maps_path)