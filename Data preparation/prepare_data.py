import polars as pl
from pathlib import Path
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from text_process import *


'''This script performs the main steps to clean the tweets, detect language, assign SES to users and create the final datasets (train, dev, test) with the users and their tweets.'''


# set directory
cwd = Path.cwd()
parent_dir = cwd.parent




def cleaning_step_1(geo_users_tweets_dir, geo_users_path, tot_tweets_path):
    '''
    Given the directory with geolocated users' tweets, the geolocated users file, concatenate all tweets of geolocated users 
    in time period 2014-2018 and keep only ones with at least 3 geolocated tweets, save in the tot_tweets_path
    '''

    # list files of geolocated users
    tweets_files = os.listdir(geo_users_tweets_dir)

    # sort by date
    sorted_tweets_month_files_filtered = sorted(tweets_files)

    # remove not to use files (total and 2019 since partial data only available there)
    to_remove = {'2019-01.parquet', '2019-02.parquet', '2019-03.parquet', 'tot_geo_users_tweets.parquet'}
    tweets_month_files = [x for x in sorted_tweets_month_files_filtered if x not in to_remove]

    # read files and put into a list to concatenate
    df_list = []
    for i in range(len(tweets_month_files)):

        #df = pl.scan_parquet(geo_users_tweets_dir / tweets_month_files[i])
        df = pl.read_parquet(geo_users_tweets_dir / tweets_month_files[i])
        df_list += [df]

    # create unified file
    tot_tweets = pl.concat(df_list)

    #n_rows = tot_tweets.select(pl.len()).collect()
    #print(n_rows)

    # extract tweets of users with at least 3 geolocated tweets

    # read geo_df with geo info and geolocated users
    geo_users_df_fr = gpd.read_parquet(geo_users_path)

    # from geolocated users df, extract users with at least 3 tweets
    geo_users_df_fr_sub = geo_users_df_fr[['userId', 'home_tweet_count', 'idINSPIRE', 'income_col']]
    pol_geo_users_df_fr_sub = pl.from_pandas(geo_users_df_fr_sub)
    pol_geo_users_df_fr_sub_3 = pol_geo_users_df_fr_sub.filter(pl.col('home_tweet_count') >= 3)
    # rename userId column to perform join
    pol_geo_users_df_fr_sub_3 = pol_geo_users_df_fr_sub_3.rename({'userId': 'user_id'})

    # create subset
    tot_tweets_greater_3 = tot_tweets.join(pol_geo_users_df_fr_sub_3, on='user_id')

    # save unified tweets
    tot_tweets_greater_3.write_parquet(tot_tweets_path)


def remove_spammers(df_path, output_tweets_path, output_suspicious_users_path):
    '''
    Given a df of tweets with the corresponding users, check if some users post more than three tweets within a the same two seconds window, 
    if so, remove them and their tweets from the df, return the cleaned df and the list of removed users
    '''

    # read df
    df = pl.scan_parquet(df_path)

    # ensure datetime
    df = df.with_columns(
        pl.col("created_at").cast(pl.Datetime)
    )

    #df = df.filter(pl.col('user_id') == '72516079')

    # sort
    df = df.sort(["user_id", "created_at"])

    #df = df.head(1000000)

    # create shifted column (t + 2)
    df = df.with_columns(
        pl.col("created_at")
        .shift(-2)
        .over("user_id")
        .alias("t_plus2")
    )

    # flag users with >= 3 tweets in 2 seconds
    df = df.with_columns(
        (
            (pl.col("t_plus2") - pl.col("created_at"))
            .dt.total_seconds() <= 2
        ).alias("flag")
    )

    df = df.collect()

    # get users to remove
    users_to_remove = (
        df.filter(pl.col("flag"))
        .select("user_id")
        .unique()
        .to_series()
        .to_list()
    )

    # filter them out
    df_clean = df.filter(
        ~pl.col("user_id").is_in(users_to_remove)
    ).drop(["t_plus2", "flag"])
    
    df_spammers = df.filter(pl.col("user_id").is_in(users_to_remove))
    df_spammers = df_spammers.to_pandas()
    
    print(users_to_remove)
    print(df_spammers)
    print(df_clean)

    # save cleaned df and list of removed users
    df_clean.write_parquet(output_tweets_path)
    #users_to_remove_df.write_csv(output_suspicious_users_path)
    pd.DataFrame({"user_id": users_to_remove}).to_csv(output_suspicious_users_path, index=False)
    df_spammers.to_csv("spammers_tweets_tot.csv", index=False)

    return df_clean, users_to_remove


def filter_geographical_area(geo_users_path, tweets_users_geo_path, boundary_file, GEO_GEOMETRY_COLUMN, filtered_users_path, filtered_tweets_path):

    # read geo users df and df of users income and ses
    geo_users_df = gpd.read_parquet(geo_users_path)
    #users_geo_income_ses_df = pd.read_parquet(users_geo_income_ses_path)

    # read tweets df
    tweets_df = pl.scan_parquet(tweets_users_geo_path)

    # rename column for merging
    geo_users_df = geo_users_df.rename(columns={'userId': 'user_id'})

    # merge df
    #merged_df = geo_users_df.merge(users_geo_income_ses_df, on='user_id')

    # extract Paris cells -> iris grid
    boundary = gpd.read_parquet(boundary_file)
    #paris_boundary = paris_boundary.set_geometry('iris_geometry')

    #todooooooooooooooooooooooooooooooooooooooooooooooooooooo
    # 1. Set correct geometry
    geo_users_df = geo_users_df.set_geometry(GEO_GEOMETRY_COLUMN)
    #merged_no_middle_df = merged_no_middle_df.set_geometry(GEO_GEOMETRY_COLUMN)

    # 2. Ensure it's a proper GeoDataFrame
    geo_users_df = gpd.GeoDataFrame(geo_users_df, geometry=GEO_GEOMETRY_COLUMN)
    #merged_no_middle_df = gpd.GeoDataFrame(merged_no_middle_df, geometry=GEO_GEOMETRY_COLUMN)

    # 3. Set CRS (since yours is EPSG:3035)
    geo_users_df.set_crs("EPSG:3035", inplace=True, allow_override=True)
    #merged_no_middle_df.set_crs("EPSG:3035", inplace=True, allow_override=True)
    boundary.set_crs("EPSG:3035", inplace=True, allow_override=True)

    # 4. Spatial join
    #geo_cells = gpd.sjoin(geo_users_df, boundary, predicate="within")

    # extract only Paris cells
    # tot classes
    geo_cells = gpd.sjoin(geo_users_df, boundary, predicate="intersects")
    geo_cells_users = geo_cells[['user_id', 'idINSPIRE', 'ind', 'ind_snv_mean', 'code_insee', 'code_iris', 'P21_POP', 'COMB_MED21', 'COMB_MED21_source']]
    print(geo_cells)
    print(geo_cells.columns)
    # no middle class
    #paris_cells_no_middle = gpd.sjoin(merged_no_middle_df, paris_boundary, predicate="intersects")

    # save all users in filtered area
    geo_cells_users.to_parquet(filtered_users_path)

    print(geo_cells_users)

    print('saved')



def cleaning_step_2(input_path, output_path):
    '''
    Given the file of tweets obtained from cleaning_step_1, perform double text cleaning:
    1) light cleaning for classifier -> remove mentions and URLs -> count number of words
    2) full cleaning for language detection -> remove mentions, URLs, hashtags, punctuation -> count number of words
    Save the final df with both types of cleaned text
    '''
    
    # Lazy pipeline to clean text

    (
        pl.scan_parquet(input_path)
        
        # keep only needed columns for efficiency
        .select(["id", "text", "user_id", "created_at", "lang"])
        
        # Add both cleaning columns
        .with_columns([
            # 1) Light cleaning: URLs + mentions
            pl.col("text")
            .str.replace_all(r"https?://\S+|www\.\S+", "")
            .str.replace_all(r"@\w+", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("clean_light"),
            
            # 2) Full cleaning: French-friendly
            pl.col("text")
            .str.replace_all(r"https?://\S+|www\.\S+", "")
            .str.replace_all(r"@\w+", "")
            .str.replace_all(r"#\w+", "")                    # remove hashtags
            .str.replace_all(r"[^\p{L}\p{N}\s'-]", "")      # remove punctuation, keep accents, apostrophes, hyphens
            .str.replace_all(r"\s+", " ")                   # collapse multiple spaces
            .str.strip_chars()
            .alias("clean_full")
        ])
        
        # Add word counts for both cleaned columns
        .with_columns([
            pl.col("clean_light").str.count_matches(r"\b[\p{L}\p{N}'-]+\b").alias("num_words_light"),
            pl.col("clean_full").str.count_matches(r"\b[\p{L}\p{N}'-]+\b").alias("num_words_full")
        ])
        
        # Optional: test subset first
        #.head(1000)
        
        # Collect and write to a new Parquet file (streaming, low RAM)
        .sink_parquet(output_path)
    )


def cleaning_step_3(input_path, output_path, min_num_words=5, lang_threshold=0.7):
    '''
    Given the file with cleaned text from cleaning_step_2, perform language detection by comparing language detected 
    by Twitter and by the language detector, before that we filter only cleaned tweets with number of words >= 5
    Save the result
    '''

    lazy_df = (pl.scan_parquet(input_path)
            .select(["id", "user_id", "created_at", "lang", "clean_light", "clean_full", "num_words_full"])
            .filter(pl.col("num_words_full") >= min_num_words)  # keep only tweets with >=5 words
            #.head(1000)
                )

    # Add a placeholder column of correct type
    lazy_df = lazy_df.with_columns(
        pl.lit("").cast(pl.String).alias("final_lang")  # same type Polars expects
    )

    # Then use map_batches to fill it
    def fill_final_lang(batch):
        batch = batch.with_columns(
            pl.Series([
                majority_language_twitter(row["clean_light"], row["lang"], lang_threshold)
                for row in batch.iter_rows(named=True)
            ]).alias("final_lang")
        )
        return batch

    lazy_df = lazy_df.map_batches(fill_final_lang)

    lazy_df.sink_parquet(output_path)

    print('Saved df !')



def assign_income(df_type, geo_users_path, tot_tweets_path, users_geo_income_tot_path, users_geo_income_threshold_path):
    '''
    Given the geolocated users file, the cleaned tweets from cleaning_step_3, join cleaned tweets with info about geo areas and population
    Use both total tweets without threshold on probability score in language detection and tweets with threshold on probability score in language detection (here default 0.7)
    Save both results
    geo_cell_col: idINSPIRE (grid 200 df), code_iris (iris df)
    income_col: ind_snv_mean (grid 200 df), COMB_MED21 (iris df)
    '''

    if df_type == '200m':
        geo_cell_col = 'idINSPIRE'
        income_col = 'ind_snv_mean'
        pop_col = 'ind'
    elif df_type == 'iris':
        geo_cell_col = 'code_iris'
        income_col = 'COMB_MED21'
        pop_col = 'P21_POP'

    # read df
    # geolocated users with their geographical area -> drop empty and zero geo cell code and income
    geo_users_df = pl.scan_parquet(geo_users_path).select(['user_id', geo_cell_col, income_col, pop_col]).filter(pl.col(geo_cell_col).is_not_null() & pl.col(income_col).is_not_null() & pl.col(income_col).is_not_nan() & (pl.col(income_col) > 0)) # pl.col('idINSPIRE').is_not_nan() &
    #print(geo_users_df.collect())
    # tweets of geolocated users
    tot_tweets_df = pl.scan_parquet(tot_tweets_path).select(['user_id', 'id', 'clean_light', 'created_at', 'num_words_full', 'final_lang'])#
    print(tot_tweets_df.columns)
    x = tot_tweets_df.collect()
    print(x.filter(pl.col('final_lang').is_null()))
    # rename columns for performing join
    #geo_users_df = geo_users_df.rename({'userId': 'user_id'})

    # whole dataset (even tweets with not detected lang by the lang detector)
    # group by user, count number of tweets per user, keep list of tweets per user (with light clean text for classifier)
    grouped_users_tot = (tot_tweets_df
                            .group_by('user_id')
                            .agg([
                                pl.len().alias("num_tweets_per_user"),
                                pl.col("id").alias("tweets_ids"),
                                #pl.col("text").alias("tweets"),
                                pl.col("clean_light").alias("clean_text_light_list")
                            ])
                            #.head(1000)
    )
    # drop nan from geo users df
    # drop nan/null geo cells & nan/null/lower_than_zero income cells
    #geo_users_cleaned_df = geo_users_df.filter(pl.col('idINSPIRE').is_not_null() & pl.col('idINSPIRE').is_not_nan() & pl.col('ind_snv_mean').is_not_null() & pl.col('ind_snv_mean').is_not_nan() & (pl.col('ind_snv_mean') > 0))

    # check how many unique users we have and with how many tweets
    # 1) above threshold for lang detection
    #tot_tweets_threshold = tot_tweets_df.filter(pl.col("final_lang").is_not_null())

    # 2) on the filtered dataset (only tweets with detected lang also by the lang detector)
    grouped_users_threshold = (tot_tweets_df.filter(pl.col("final_lang").is_not_null())
                        .group_by('user_id')
                        .agg([
                            pl.len().alias("num_tweets_per_user"),
                            pl.col("id").alias("tweets_ids"),
                            #pl.col("text").alias("tweets"),
                            pl.col("clean_light").alias("clean_text_light_list")
                        ])
                        #.head(1000)
    )


    # join geo and tweets df

    # total
    joined_tot_df = grouped_users_tot.join(geo_users_df, on='user_id')
    # with threshold on language confidence
    joined_threshold_df = grouped_users_threshold.join(geo_users_df, on='user_id')

    # save outputs
    joined_tot_df.sink_parquet(users_geo_income_tot_path)
    joined_threshold_df.sink_parquet(users_geo_income_threshold_path)



def gini_from_lorenz(x, y):
    """
    x: cumulative users/population/tweets fraction (sorted, from 0 to 1)
    y: cumulative income fraction (sorted, from 0 to 1)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # ensure starts at 0 and ends at 1
    if x[0] > 0:
        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)

    if x[-1] < 1:
        x = np.append(x, 1)
        y = np.append(y, 1)

    area = np.trapz(y, x)
    gini = 1 - 2 * area
    return gini


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




def assign_ses(medium_class, users_geo_income_path, df_type, ses_users_output_path, ses_income_output_path, output_dir_plots):
    '''
    Group users by geographical cells, compute cumulative sum over users and over income, divide users into SES 
    according to equal fraction of users in each class (method we use in our work) and according to equal fraction
    of income, save the obtained dfs, plot (only for the SES according to users) the cumulative fraction of 
    users VS income, tweets VS income, divisions into classes
    Parameters:
    - df_type = '200m' | 'iris'
    - geo_cell_col = 'code_iris' (iris df) | 'idINSPIRE' (grid 200m df)
    - income_col = 'ind_snv_mean' (grid 200m df) | 'COMB_MED21' (iris df)
    - pop_col = 'ind' (grid 200m df) | 'P21_POP'
    '''

    if df_type == '200m':
        geo_cell_col = 'idINSPIRE'
        income_col = 'ind_snv_mean'
        pop_col = 'ind'
    elif df_type == 'iris':
        geo_cell_col = 'code_iris'
        income_col = 'COMB_MED21'
        pop_col = 'P21_POP'

    # read df with useful columns
    users_geo_income_df = pl.scan_parquet(users_geo_income_path).select(['user_id', 'num_tweets_per_user', geo_cell_col, income_col, pop_col])

    print(users_geo_income_df.collect())

    # group by geographical cell and count population, number of users, and number of tweets within each of them
    grouped_geo = (users_geo_income_df.group_by(geo_cell_col)
                .agg([
                    pl.col('user_id').count().alias('num_users'),
                    pl.col('user_id').alias('list_users_ids'),
                    pl.col('num_tweets_per_user').sum().alias('num_tweets'),
                    pl.col(income_col).first().alias(income_col),
                    pl.col(pop_col).first().alias(pop_col)
                    ])
                )
    
    # plot distribution of users and poplation per geographical cell

    grouped_geo_plot = grouped_geo.collect()

    # users
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(grouped_geo_plot['num_users'], bins=100, edgecolor='black')
    ax.set_yscale('log')
    ax.set_xlabel('Population per geographical cell')
    ax.set_ylabel('Frequency')
    #fig.savefig(output_dir_plots / 'users_distr_PARIS_LARGE_LARGE.png')

    # population
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(grouped_geo_plot[pop_col], bins=100, edgecolor='black')
    ax.set_yscale('log')
    ax.set_xlabel('Users per geographical cell')
    ax.set_ylabel('Frequency')
    #fig.savefig(output_dir_plots / 'pop_distr_PARIS_LARGE_LARGE.png')

    grouped_geo_sorted = grouped_geo.sort(income_col, descending=False)

    print(grouped_geo_sorted.collect())

    # compute cumulative sum over different values
    # users
    grouped_geo_sorted = grouped_geo_sorted.with_columns(
        (pl.col('num_users').cum_sum() / pl.col('num_users').sum())
        .alias('cum_fraction_users')
    )
    # compute cumulative sum over different values
    # users
    grouped_geo_sorted = grouped_geo_sorted.with_columns(
        (pl.col(pop_col).cum_sum() / pl.col(pop_col).sum())
        .alias('cum_fraction_population')
    )
    #grouped_geo_users = grouped_geo_users.sort(pl.col('ind_snv_mean').reverse())
    # tweets
    grouped_geo_sorted = grouped_geo_sorted.with_columns(
        (pl.col('num_tweets').cum_sum() / pl.col('num_tweets').sum())
        .alias('cum_fraction_tweets')
    )
    # income
    grouped_geo_sorted = grouped_geo_sorted.with_columns(
        (pl.col(income_col).cum_sum() / pl.col(income_col).sum())
        .alias('cum_fraction_income')
    )

    grouped_geo_sorted = grouped_geo_sorted.collect()


    # divide into different SES

    # according to equal fraction of users

    users_col = 'cum_fraction_users'

    # divide into 2 or 3 classes according to parameter medium_class

    if medium_class == True:

        grouped_geo_users = grouped_geo_sorted.with_columns(
            pl.when(pl.col(users_col) <= 0.33333).then(pl.lit('lower'))
            .when((pl.col(users_col) > 0.33333) & (pl.col(users_col) <= 0.66666)).then(pl.lit('medium'))
            .otherwise(pl.lit('high'))
            .alias('SES_users')
        )

        # according to equal fraction of income

        income_col = 'cum_fraction_income' # cum_fraction_users

        grouped_geo_income = grouped_geo_sorted.with_columns(
            pl.when(pl.col(income_col) <= 0.33333).then(pl.lit('lower'))
            .when((pl.col(income_col) > 0.33333) & (pl.col(income_col) <= 0.66666)).then(pl.lit('medium'))
            .otherwise(pl.lit('high'))
            .alias('SES_income')
        )
    
    else:

        grouped_geo_users = grouped_geo_sorted.with_columns(
            pl.when(pl.col(users_col) <= 0.5).then(pl.lit('lower'))
            .otherwise(pl.lit('high'))
            .alias('SES_users')
        )

        # according to equal fraction of income

        income_col = 'cum_fraction_income' # cum_fraction_users

        grouped_geo_income = grouped_geo_sorted.with_columns(
            pl.when(pl.col(income_col) <= 0.5).then(pl.lit('lower'))
            .otherwise(pl.lit('high'))
            .alias('SES_income')
        )


    print(grouped_geo_users.sort('num_users', descending=True))
    print(grouped_geo_income.sort('num_users', descending=True))

    # create df of users with their corresponding SES (lower/high)
    grouped_geo_users_filtered = (grouped_geo_users
                                .select(['list_users_ids', 'num_users', 'SES_users'])
                                .explode('list_users_ids')
                                .rename({'list_users_ids': 'user_id'}))
    grouped_geo_income_filtered = (grouped_geo_income
                                .select(['list_users_ids', 'num_users', 'SES_income'])
                                .explode('list_users_ids')
                                .rename({'list_users_ids': 'user_id'}))

    # add SES to each user in initial df with all the info users_geo_income_df
    users_geo_income_df = users_geo_income_df.collect()
    # SES users
    users_geo_users_ses_df = grouped_geo_users_filtered.join(users_geo_income_df, on='user_id')
    # SES income
    users_geo_income_ses_df = grouped_geo_income_filtered.join(users_geo_income_df, on='user_id')

    # save df of users ses
    users_geo_users_ses_df.write_parquet(ses_users_output_path)#output_users_geo_income_ses_path
    users_geo_income_ses_df.write_parquet(ses_income_output_path)#output_users_geo_income_ses_path


    # PLOTS -> we only use the users division into ses, which is the one we use for our work

    # 1) plot fraction of users VS cumulative distribution of income

    x = grouped_geo_sorted['cum_fraction_users'].to_numpy()
    y = grouped_geo_sorted['cum_fraction_income'].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(x, y, s=1)

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]

    ax.plot(lims, lims, '--', color='gray')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel('Fraction of users', size=15)#users
    ax.set_ylabel('Normalized cumulative distribution function of income', size=15)

    #plt.savefig(output_dir_plots / 'frac_users_VS_income_PARIS_LARGE_LARGE.png')


    # 2) plot fraction of users and population and tweets VS cumulative distribution of income

    x1 = grouped_geo_sorted['cum_fraction_users']
    x2 = grouped_geo_users['cum_fraction_population']
    x3 = grouped_geo_sorted['cum_fraction_tweets']
    y = grouped_geo_sorted['cum_fraction_income']

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(x1, y, s=1, label='Users')
    ax.scatter(x2, y, s=1, color='green', label='Population')
    ax.scatter(x3, y, s=1, color='red', label='Tweets')

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]

    ax.plot(lims, lims, '--', color='gray')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel('Fraction of users / population / tweets', size=15)#users
    ax.set_ylabel('Normalized cumulative distribution function of income', size=15)

    ax.legend(fontsize=12, markerscale=5)

    #plt.savefig(output_dir_plots / 'frac_users_pop_tweets_VS_income_PARIS_LARGE_LARGE.png')

    # compute gini index
    # users VS income
    gini_users = gini_from_lorenz(x1, y)

    # population VS income
    gini_pop = gini_from_lorenz(x2, y)

    # tweets VS income
    gini_tweets = gini_from_lorenz(x3, y)

    # save gini
    gini_df = pd.DataFrame({'category': ['num users', 'population', 'num tweets'], 'gini': [gini_users, gini_pop, gini_tweets]})
    #gini_df.to_csv(output_dir_plots / 'gini_lorenz.csv', index=False)
    print('Gini index users-income:', gini_users)
    print('Gini population-income:', gini_pop)
    print('Gini tweets-income:', gini_tweets)


    # 3) plot fraction of users VS cumulative distribution of income -> divide into classes according to same population

    x = grouped_geo_sorted['cum_fraction_users'] #cum_fraction_users
    y = grouped_geo_sorted['cum_fraction_income']

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(x, y, s=1)

    ax.set_xlabel('Fraction of users', size=15)#users
    ax.set_ylabel('Normalized cumulative distribution function of income', size=15)

    # vertical lines
    if medium_class == True:

        x1 = 0.33333
        x2 = 0.66666

        idx_vertical_1 = np.argmin(np.abs(x - x1))
        idx_vertical_2 = np.argmin(np.abs(x - x2))
        print(idx_vertical_1)
        print(idx_vertical_2)

        ax.axvline(x=x1, color='r')
        ax.axvline(x=x2, color='r')

        ax.axhline(y=y[int(idx_vertical_1)], color='g', linestyle='--')
        ax.axhline(y=y[int(idx_vertical_2)], color='g', linestyle='--')

    else:

        x1 = 0.5

        idx_vertical_1 = np.argmin(np.abs(x - x1))
        print(idx_vertical_1)

        ax.axvline(x=x1, color='r')

        ax.axhline(y=y[int(idx_vertical_1)], color='g', linestyle='--')


    plt.show()

    plt.savefig(output_dir_plots / 'frac_users_VS_income_division_users_PARIS_LARGE_LARGE_2_classes.png')
    

    # 4) plot fraction of users VS cumulative distribution of income -> divide into classes according to same income

    x = grouped_geo_sorted['cum_fraction_users']#cum_fraction_users
    y = grouped_geo_sorted['cum_fraction_income']

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(x, y, s=1)

    ax.set_xlabel('Fraction of users', size=15)
    ax.set_ylabel('Normalized cumulative distribution function of income', size=15)

    # horizontal lines
    if medium_class == True:

        y1 = 0.33333
        y2 = 0.66666

        idx_horizontal_1 = np.where(np.diff(np.sign(y - y1)))[0][0]
        idx_horizontal_2 = np.where(np.diff(np.sign(y - y2)))[0][0]

        ax.axhline(y=y1, color='r')
        ax.axhline(y=y2, color='r')

        ax.axvline(x=x[int(idx_horizontal_1)], color='g', linestyle='--')
        ax.axvline(x=x[int(idx_horizontal_2)], color='g', linestyle='--')

    else:

        y1 = 0.5

        idx_horizontal_1 = np.where(np.diff(np.sign(y - y1)))[0][0]

        ax.axhline(y=y1, color='r')

        ax.axvline(x=x[int(idx_horizontal_1)], color='g', linestyle='--')

    plt.show()

    plt.savefig(output_dir_plots / 'frac_users_VS_income_division_income_PARIS_LARGE_LARGE_2_classes.png')
    


def create_users_sample(users_geo_income_ses_path, suspicious_users_path, test_users_path, num_small_tiles, num_users_big_tile, num_users_per_class, no_middle, sampled_users_path):
    

    # num_small_tiles = 10

    # read df
    #tweets_users_geo_df = pl.scan_parquet(tweets_users_geo_path)
    users_geo_income_ses_df = pl.scan_parquet(users_geo_income_ses_path)

    # suspicious users df
    suspicious_users_df = pl.read_csv(suspicious_users_path)
    # remove users with too many tweets
    suspicious_users = list(suspicious_users_df['user_id'])
    suspicious_users = [str(x) for x in suspicious_users]
    users_geo_income_ses_df = users_geo_income_ses_df.filter(~pl.col('user_id').is_in(suspicious_users))

    # check if train or test sample, we first do test and then train
    # so, if train, remove test users from the used df
    # set test_users_path = None if we run test sample, otherwise open the test users sample file
    if test_users_path != None:
        test_users_df = pl.read_parquet(test_users_path)
        test_users = list(test_users_df['user_id'])
        test_users = [str(x) for x in test_users]
        users_geo_income_ses_df = users_geo_income_ses_df.filter(~pl.col('user_id').is_in(test_users))


    # joined df
    #tweets_users_geo_income_ses_df = tweets_users_geo_df.join(users_geo_income_ses_df, on='user_id')

    # PIPELINE

    # remove middle class users
    if no_middle == True:
        df = users_geo_income_ses_df.filter(pl.col('SES_users') != 'medium')
    else:
        df = users_geo_income_ses_df
    '''df_lower = df.filter(pl.col('SES_users') == 'lower')
    df_high = df.filter(pl.col('SES_users') == 'high')'''

    # --------------------------------------------------
    # 1. extract coordinates from INSPIRE grid code
    # --------------------------------------------------

    df = df.with_columns([
        pl.col("idINSPIRE").str.extract(r"N(\d+)", 1).cast(pl.Int64).alias("northing"),
        pl.col("idINSPIRE").str.extract(r"E(\d+)", 1).cast(pl.Int64).alias("easting")
    ])

    # --------------------------------------------------
    # 2. create big tiles (20 km)
    # --------------------------------------------------

    df = df.with_columns([
        (pl.col("northing") // 20000).alias("tile_y"),
        (pl.col("easting") // 20000).alias("tile_x")
    ])

    # unique id for big tiles
    df = df.with_columns(
        pl.concat_str(["tile_x","tile_y"], separator="_").alias("big_tile")
    )

    # --------------------------------------------------
    # 3. create large spatial strata (~100 km)
    # ensures coverage across France
    # --------------------------------------------------

    df = df.with_columns([
        (pl.col("northing") // 100000).alias("strata_y"),
        (pl.col("easting") // 100000).alias("strata_x")
    ])

    df = df.with_columns(
        pl.concat_str(["strata_x","strata_y"], separator="_").alias("stratum")
    )

    # --------------------------------------------------
    # 4. sample big tiles inside each stratum
    # --------------------------------------------------

    sampled_big_tiles = (
        df.select(["big_tile", "stratum", "SES_users"])
        .unique()
        .group_by(["SES_users", "stratum"])
        .agg(
            pl.col("big_tile").sample(
                # sample 80 % of big tiles within each stratum, check that there is at least one big tile, id there is 
                # only 1 pick tath one
                n=pl.max_horizontal([(pl.count("big_tile") * 0.8).cast(pl.Int32), pl.lit(1)])
            )
        ).explode("big_tile")
    )

    # extract df with lower class users and high class users
    df_small_lower = sampled_big_tiles.filter(pl.col('SES_users') == 'lower')
    df_small_high = sampled_big_tiles.filter(pl.col('SES_users') == 'high')
    if no_middle == False:
        df_small_middle = sampled_big_tiles.filter(pl.col('SES_users') == 'medium')

    # keep only users (from df) in those big tiles
    df_tiles = df.join(sampled_big_tiles, on=["stratum", "big_tile", "SES_users"], how="inner")

    # --------------------------------------------------
    # 5. sample n small tiles (200m) per big tile
    # --------------------------------------------------

    sampled_small_tiles = (
        df_tiles.group_by(["big_tile", "SES_users"])
        .agg(
            pl.col("idINSPIRE").unique().sample(
                # sample n small tiles (if there are at least n), otherwise all the small tiles in the big tile
                n=pl.min_horizontal([pl.col("idINSPIRE").n_unique(), pl.lit(num_small_tiles)])
            )
        )
        .explode("idINSPIRE")
    )

    # keep users filtered before (from df_tiles) inside those small tiles
    df_small = df_tiles.join(
        sampled_small_tiles,
        on=["big_tile", "idINSPIRE", "SES_users"],
        how="inner"
    )
    

    # extract df with lower class users and high class users
    df_small_lower = df_small.filter(pl.col('SES_users') == 'lower')
    df_small_high = df_small.filter(pl.col('SES_users') == 'high')
    if no_middle == False:
        df_small_middle = df_small.filter(pl.col('SES_users') == 'lower')

    # --------------------------------------------------
    # 6. sample one or more users per big tile
    # --------------------------------------------------

    # lower class
    df_users_lower = (
        df_small_lower
        .with_columns(
            # shuffle rows of each big tile
            pl.int_range(0, pl.len()).shuffle().over("big_tile").alias("rnd")
        )
        .sort("rnd")
        .group_by("big_tile")
        # extract n = num_users_big_tile users per big tile among the small tiles selected before belonging to that big tile
        .head(num_users_big_tile)
    )

    # high class
    df_users_high = (
        df_small_high
        .with_columns(
            # shuffle rows of each big tile
            pl.int_range(0, pl.len()).shuffle().over("big_tile").alias("rnd")
        )
        .sort("rnd")
        .group_by("big_tile")
        # extract n = num_users_big_tile users per big tile among the small tiles selected before belonging to that big tile
        .head(num_users_big_tile)
    )

    if no_middle == False:
        # middle class
        df_users_middle = (
            df_small_middle
            .with_columns(
                # shuffle rows of each big tile
                pl.int_range(0, pl.len()).shuffle().over("big_tile").alias("rnd")
            )
            .sort("rnd")
            .group_by("big_tile")
            # extract n = num_users_big_tile users per big tile among the small tiles selected before belonging to that big tile
            .head(num_users_big_tile)
        )


    # extract 500 users from lower class and 500 users from high class
    df_users_lower = df_users_lower.collect()

    users_sample_lower_df = df_users_lower.sample(n=num_users_per_class)
    users_sample_high_df = df_users_high.sample(n=num_users_per_class)

    if no_middle == False:
        users_sample_middle_df = df_users_middle.sample(n=num_users_per_class)
        tot_sampled_users_df = pl.concat([users_sample_lower_df, users_sample_middle_df, users_sample_high_df])
    else:
        tot_sampled_users_df = pl.concat([users_sample_lower_df, users_sample_high_df])        

    tot_sampled_users_df.write_parquet(sampled_users_path)


def create_users_sample_paris(users_geo_income_ses_path, suspicious_users_path, test_users_path, num_tweets_per_user, num_small_tiles, num_users_big_tile, num_users_per_class, no_middle, sampled_users_path):
    '''
    Sample users randomly in the chosen area (Ile-De-France) and uniformely in space and creating balanced classes (lower, medium, high) according to users SES class'''


    # num_small_tiles = 10

    # read df
    #tweets_users_geo_df = pl.scan_parquet(tweets_users_geo_path)
    # users df
    users_geo_income_ses_df = pl.scan_parquet(users_geo_income_ses_path)
    # select users with at least n tweets
    users_geo_income_ses_df = users_geo_income_ses_df.filter(pl.col('num_tweets_per_user') >= num_tweets_per_user)
    # suspicious users df
    suspicious_users_df = pl.read_csv(suspicious_users_path)
    # remove users with too many tweets
    suspicious_users = list(suspicious_users_df['user_id'])
    suspicious_users = [str(x) for x in suspicious_users]
    users_geo_income_ses_df = users_geo_income_ses_df.filter(~pl.col('user_id').is_in(suspicious_users))

    print('total users')
    print(users_geo_income_ses_df.collect())
    
    # check if train or test sample, we first do test and then train
    # so, if train, remove test users from the used df
    # set test_users_path = None if we run test sample, otherwise open the test users sample file
    if test_users_path != None:
        test_users_df = pl.read_parquet(test_users_path)
        test_users = list(test_users_df['user_id'])
        test_users = [str(x) for x in test_users]
        users_geo_income_ses_df = users_geo_income_ses_df.filter(~pl.col('user_id').is_in(test_users))

    # joined df
    #tweets_users_geo_income_ses_df = tweets_users_geo_df.join(users_geo_income_ses_df, on='user_id')

    # PIPELINE

    # remove middle class users
    if no_middle == True:
        df = users_geo_income_ses_df.filter(pl.col('SES_users') != 'medium')
    else:
        df = users_geo_income_ses_df
    '''df_lower = df.filter(pl.col('SES_users') == 'lower')
    df_high = df.filter(pl.col('SES_users') == 'high')'''

    # --------------------------------------------------
    # 1. extract coordinates from INSPIRE grid code
    # --------------------------------------------------

    df = df.with_columns([
        pl.col("idINSPIRE").str.extract(r"N(\d+)", 1).cast(pl.Int64).alias("northing"),
        pl.col("idINSPIRE").str.extract(r"E(\d+)", 1).cast(pl.Int64).alias("easting")
    ])

    # --------------------------------------------------
    # 2. create big tiles (20 km)
    # --------------------------------------------------

    df = df.with_columns([
        (pl.col("northing") // 3000).alias("tile_y"),# paris + ring -> 2000 
        (pl.col("easting") // 3000).alias("tile_x")
    ])

    # unique id for big tiles
    df = df.with_columns(
        pl.concat_str(["tile_x","tile_y"], separator="_").alias("big_tile")
    )

    # --------------------------------------------------
    # 3. create large spatial strata (~100 km)
    # ensures coverage across France
    # --------------------------------------------------

    '''df = df.with_columns([
        (pl.col("northing") // 100000).alias("strata_y"),
        (pl.col("easting") // 100000).alias("strata_x")
    ])

    df = df.with_columns(
        pl.concat_str(["strata_x","strata_y"], separator="_").alias("stratum")
    )'''

    # --------------------------------------------------
    # 4. sample big tiles inside each stratum
    # --------------------------------------------------
    #print(df.collect())
    '''sampled_big_tiles = (
        df.select(["stratum","big_tile"])
        .unique()
        .with_columns(pl.int_range(0, pl.len()).shuffle().over("stratum").alias("rnd"))
        .filter(pl.col("rnd") < 20)
        .select(["stratum","big_tile"])
    )'''
    
    sampled_big_tiles = (
        df.select(["big_tile", "SES_users"])
        .unique()
        .group_by(["SES_users"])
        .agg(
            pl.col("big_tile").sample(
                # sample 80 % of big tiles within each stratum, check that there is at least one big tile, id there is 
                # only 1 pick tath one
                n=pl.max_horizontal([(pl.count("big_tile") * 0.8).cast(pl.Int32), pl.lit(1)])
            )
        ).explode("big_tile")
    )

    # extract df with lower class users and high class users
    df_small_lower = sampled_big_tiles.filter(pl.col('SES_users') == 'lower')
    df_small_high = sampled_big_tiles.filter(pl.col('SES_users') == 'high')
    if no_middle == False:
        df_small_middle = sampled_big_tiles.filter(pl.col('SES_users') == 'medium')

    # keep only users (from df) in those big tiles
    df_tiles = df.join(sampled_big_tiles, on=["big_tile", "SES_users"], how="inner")

    # --------------------------------------------------
    # 5. sample n small tiles (200m) per big tile
    # --------------------------------------------------

    sampled_small_tiles = (
        df_tiles.group_by(["big_tile", "SES_users"])
        .agg(
            pl.col("idINSPIRE").unique().sample(
                # sample n small tiles (if there are at least n), otherwise all the small tiles in the big tile
                n=pl.min_horizontal([pl.col("idINSPIRE").n_unique(), pl.lit(num_small_tiles)])
            )
        )
        .explode("idINSPIRE")
    )

    # keep users filtered before (from df_tiles) inside those small tiles
    df_small = df_tiles.join(
        sampled_small_tiles,
        on=["big_tile", "idINSPIRE", "SES_users"],
        how="inner"
    )

    print(df_small.collect())

    # extract df with lower class users and high class users
    df_small_lower = df_small.filter(pl.col('SES_users') == 'lower')
    df_small_high = df_small.filter(pl.col('SES_users') == 'high')
    if no_middle == False:
        df_small_middle = df_small.filter(pl.col('SES_users') == 'medium')

    # --------------------------------------------------
    # 6. sample one or more users per big tile
    # --------------------------------------------------

    # lower class
    df_users_lower = (
        df_small_lower
        .with_columns(
            # shuffle rows of each big tile
            pl.int_range(0, pl.len()).shuffle().over("big_tile").alias("rnd")
        )
        .sort("rnd")
        .group_by("big_tile")
        # extract n = num_users_big_tile users per big tile among the small tiles selected before belonging to that big tile
        .head(num_users_big_tile)
    )

    # high class
    df_users_high = (
        df_small_high
        .with_columns(
            # shuffle rows of each big tile
            pl.int_range(0, pl.len()).shuffle().over("big_tile").alias("rnd")
        )
        .sort("rnd")
        .group_by("big_tile")
        # extract n = num_users_big_tile users per big tile among the small tiles selected before belonging to that big tile
        .head(num_users_big_tile)
    )

    # middle class
    if no_middle == False:
        df_users_middle = (
            df_small_middle
            .with_columns(
                # shuffle rows of each big tile
                pl.int_range(0, pl.len()).shuffle().over("big_tile").alias("rnd")
            )
            .sort("rnd")
            .group_by("big_tile")
            # extract n = num_users_big_tile users per big tile among the small tiles selected before belonging to that big tile
            .head(num_users_big_tile)
        )

    #print(df_users_lower.select(['idINSPIRE']).collect())
    #print(df_users_high.select('idINSPIRE').collect())

    # extract 500 users from lower class and 500 users from high class
    df_users_lower = df_users_lower.collect()
    df_users_high = df_users_high.collect()
    df_users_middle = df_users_middle.collect()
    print('users lower', df_users_lower)
    print('users high', df_users_high)
    print('users middle', df_users_middle)
    users_sample_lower_df = df_users_lower.sample(n=num_users_per_class)
    users_sample_high_df = df_users_high.sample(n=num_users_per_class)
    if no_middle == False:
        #df_users_middle = df_users_middle.collect()
        users_sample_middle_df = df_users_middle.sample(n=num_users_per_class)

    if no_middle == True:
        tot_sampled_users_df = pl.concat([users_sample_lower_df, users_sample_high_df])
    else:
        tot_sampled_users_df = pl.concat([users_sample_lower_df, users_sample_middle_df, users_sample_high_df])

    print('ECCOLO', tot_sampled_users_df)
        

    tot_sampled_users_df.write_parquet(sampled_users_path)


def sample_tweets_per_user(user_path, tweets_path, num_tweets_per_user):
    '''Sample n tweets per user'''

    user_df = pl.scan_parquet(user_path)
    tweets_df = pl.scan_parquet(tweets_path)

    merged_df = user_df.join(tweets_df, on='user_id', how='inner')

    sampled_df = (
        merged_df
        .group_by('user_id')
        .head(num_tweets_per_user)
    )

    sampled_df = sampled_df.collect()

    #sampled_df.write_csv(output_path)
    
    return sampled_df


def sample_tweets_per_user_tot(user_path, tweets_path):
    '''Get all tweets of the users'''

    user_df = pl.scan_parquet(user_path)
    tweets_df = pl.scan_parquet(tweets_path)

    merged_df = user_df.join(tweets_df, on='user_id', how='inner')

    merged_df = merged_df.collect()

    #sampled_df.write_csv(output_path)
    
    return merged_df


if __name__ == "__main__":

    # set directory
    cwd = Path.cwd()
    parent_dir = cwd.parent


    # paths
    tot_tweets_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018_no_spammers.parquet' #'tweets_geo_users_2014_2018.parquet'

    # variables
    df_type = '200m' # '200m' | 'iris'
    #geo_cell_col = 'idINSPIRE' # 'code_iris' (iris df) | 'idINSPIRE' (grid 200m df)
    #income_col = 'ind_snv_mean' # 'ind_snv_mean' (grid 200m df) | 'COMB_MED21' (iris df)

    # ----------------------------
    # 1) CLEANING: STEP 1
    # ----------------------------

    # concatenate all tweets of geolocated users in time period 2014-2018 and keep only ones with at least 3 geolocated tweets
    # we can use tweets_geolocated_users_200m and user_home_locations_and_attributes_200m.parquet because the geolocated users 
    # are the same in the iris and 200m grid, they are only assigned to different celles in the two systams

    '''# paths
    # folder with tweets divided by month
    geo_users_tweets_dir = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_200m'
    # geolocated users with geographical area and population
    geo_users_path = parent_dir / 'geo_ses_files' / 'FR' / 'user_home_locations_and_attributes_200m.parquet'
    # outputs
    # output path for unified tweets from 2014 until 2019 for geolocated users with at least 3 geolocated tweets
    tot_tweets_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018.parquet'

    # cleaning step 1
    cleaning_step_1(geo_users_tweets_dir, geo_users_path, tot_tweets_path)

    # filter suspicious users out
    # paths
    tot_tweets_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018.parquet'
    output_tweets_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018_no_spammers.parquet'
    output_suspicious_users_path = parent_dir / 'geo_ses_files' / 'FR' / 'suspicious users' / 'suspicious_users_2_sec.csv'
    # extract reliable users
    df_clean, users_to_remove = remove_spammers(tot_tweets_path, output_tweets_path, output_suspicious_users_path)'''

    # if needed, reduce geographical area

    '''AREA = 'PARIS' # FR | PARIS
    CREATE_PARIS_DF = True # True | False
    GEO_GEOMETRY_COLUMN = 'carreau_geometry'
    no_middle = False

    # path geolocated users
    geo_users_path = parent_dir / 'geo_ses_files' / 'FR' / 'user_home_locations_and_attributes_200m.parquet' # user_home_locations_and_attributes_200m.parquet
    # path of df of all tweets of geolocated users after all the cleaning
    tweets_users_geo_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018_no_spammers.parquet' #'tweets_geo_users_2014_2018_cleaned_lang_0.7.parquet'
    # path of the df with users, number of tweets per user, geo cell, income, SES
    #users_geo_income_ses_path = parent_dir / 'geo_ses_files' / 'FR' / 'users_geo_income_200m_threshold_0.7_SES_USERS_NEW_no_spammers.parquet' #'users_geo_income_200m_threshold_0.7_SES_USERS_NEW.parquet'
    
    # boundaries file
    boundary_file = '../useful stuff/paris boundaries/paris_boundary_LARGE_LARGE.parquet' # possible files : paris_boundary_75, paris_boundary_LARGE, paris_boundary_LARGE_LARGE
    
    # filtered users path
    filtered_users_path = parent_dir / 'geo_ses_files' / 'FR' / 'user_home_locations_and_attributes_200m_PARIS_LARGE_LARGE.parquet'

    # filtered tweets path
    filtered_tweets_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018_no_spammers_PARIS_LARGE_LARGE.parquet'

    filter_geographical_area(geo_users_path, tweets_users_geo_path, boundary_file, GEO_GEOMETRY_COLUMN, filtered_users_path, filtered_tweets_path)'''



    # ----------------------------
    # 2) CLEANING: STEP 2
    # ----------------------------

    # 1) light cleaning for classifier -> remove mentions and URLs -> count number of words
    # 2) full cleaning for language detection -> remove mentions, URLs, hashtags, punctuation -> count number of words

    # paths
    input_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018_no_spammers.parquet' #'tweets_geo_users_2014_2018.parquet'
    output_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018_cleaned_no_spammers.parquet' #'tweets_geo_users_2014_2018_cleaned.parquet'

    # cleaning step 2
    cleaning_step_2(input_path, output_path)



    # ----------------------------
    # 3) CLEANING: STEP 3
    # ----------------------------

    # language detection by comparing language detected by Twitter and by the language detector, before that we filter only cleaned tweets with number of words >= 5

    # paths
    input_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018_cleaned_no_spammers.parquet' #'tweets_geo_users_2014_2018_cleaned.parquet'
    output_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018_cleaned_lang_0.7_no_spammers.parquet' #'tweets_geo_users_2014_2018_cleaned_lang_0.7.parquet'

    # cleaning step 3
    cleaning_step_3(input_path, output_path, min_num_words=5, lang_threshold=0.7)

    

    # ----------------------------
    # 4) ASSIGN INCOME
    # ----------------------------

    # join cleaned tweets of geolocated users with info about geographical areas and population

    # paths
    # geolocated users with their geographical area
    geo_users_path = parent_dir / 'geo_ses_files' / 'FR' / 'user_home_locations_and_attributes_200m_PARIS_LARGE_LARGE.parquet' #'user_home_locations_and_attributes_200m.parquet' #user_home_locations_and_attributes_200m.parquet
    # different cleaning steps tweets data
    #tweets_2014_2018_tot_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018.parquet'
    #tweets_2014_2018_cleaned_1_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018_cleaned.parquet'
    #tweets_2014_2018_cleaned_2_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018_cleaned_lang_0.7.parquet'

    # total cleaned tweets of geolocated users
    #tot_tweets_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'cleaning_steps_files' / 'tweets_geo_users_2014_2018_cleaned_no_spammers.parquet' #'tweets_geo_users_2014_2018_cleaned_lang_0.7.parquet'
    tot_tweets_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'cleaning_steps_files' / 'tweets_geo_users_2014_2018_cleaned_lang_0.7_no_spammers.parquet'
    # output
    # user, geo, income
    users_geo_income_tot_path = parent_dir / 'geo_ses_files' / 'FR' / 'user_geo_income_200m_PARIS_LARGE_LARGE.parquet' #users_geo_income_200m_tot.parquet
    users_geo_income_threshold_path = parent_dir / 'geo_ses_files' / 'FR' / 'user_geo_income_200m_threshold_0.7_PARIS_LARGE_LARGE.parquet' #'users_geo_income_200m_threshold_0.7_NEW.parquet' #users_geo_income_200m_threshold_0.7.parquet

    #users_geo_income_tot_path = '../geo_ses_files/FR/user_geo_income_200m_PARIS_LARGE_LARGE.parquet' #users_geo_income_200m_tot.parquet
    #users_geo_income_threshold_path = '../geo_ses_files/FR/user_geo_income_200m_threshold_0.7_PARIS_LARGE_LARGE.parquet'

    # variables
    #geo_cell_col = 'idINSPIRE' # 'code_iris' (iris df), 'idINSPIRE' (grid 200m df)
    #income_col = 'ind_snv_mean' # 'ind_snv_mean' (grid 200m df), 'COMB_MED21' (iris df)

    # assign income
    assign_income(df_type, geo_users_path, tot_tweets_path, users_geo_income_tot_path, users_geo_income_threshold_path)
    


    # ----------------------------
    # 5) ASSIGN SES
    # ----------------------------

    # paths
    df_type = '200m' # 'iris'
    # df of users with their tweets and geographical cell and income
    file_1 = 'users_geo_income_' + df_type + '_threshold_0.7_NEW.parquet'
    users_geo_income_path = parent_dir / 'geo_ses_files' / 'FR' / file_1 # users_geo_income_iris_threshold_0.7.parquet
    # output path for df with users, number of tweets per user, geo cell, income, SES
    #output_users_geo_income_ses_path = parent_dir / 'geo_ses_files' / 'FR' / 'users_geo_income_ses_200m_threshold_0.7.parquet'
    file_2 = 'users_geo_income_' + df_type + '_threshold_0.7_SES_USERS_NEW.parquet'
    ses_users_output_path = parent_dir / 'geo_ses_files' / 'FR' / file_2
    file_3 = 'users_geo_income_' + df_type + '_threshold_0.7_SES_INCOME_NEW.parquet'
    ses_income_output_path = parent_dir / 'geo_ses_files' / 'FR' / file_3


    MIDDLE_CLASS = False # False if we want to divide into 2 classes, True if into 3 classes
    
    file_4 = 'plots PARIS LARGE LARGE ' + df_type
    output_dir_plots = parent_dir / 'geo_ses_files' / 'FR' / file_4

    users_geo_income_path = parent_dir / 'geo_ses_files' / 'FR' / 'user_geo_income_200m_PARIS_LARGE_LARGE.parquet'

    ses_users_output_path = parent_dir / 'geo_ses_files' / 'FR' / 'users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE_2_classes.parquet'

    ses_income_output_path = parent_dir / 'geo_ses_files' / 'FR' / 'users_geo_income_200m_SES_INCOME_PARIS_LARGE_LARGE_2_classes.parquet'

    # assign ses
    assign_ses(MIDDLE_CLASS, users_geo_income_path, df_type, ses_users_output_path, ses_income_output_path, output_dir_plots)
    
    # extract users not in spammers
    users_df = pl.read_parquet(parent_dir / 'geo_ses_files' / 'FR' / 'users_geo_income_200m_threshold_0.7_SES_USERS_NEW.parquet')
    # convert user_id to string
    users_df = users_df.with_columns(pl.col('user_id').cast(pl.String))
    print(users_df)
    spammers_df = pl.read_csv(parent_dir / 'geo_ses_files' / 'FR' / 'suspicious users' / 'suspicious_users_2_sec.csv')
    spammers_df = spammers_df.with_columns(pl.col('user_id').cast(pl.String))
    spammers_list = list(spammers_df['user_id'])
    users_no_spammers_df = users_df.filter(~pl.col('user_id').is_in(spammers_list))
    print(users_no_spammers_df)
    users_no_spammers_df.write_parquet(parent_dir / 'geo_ses_files' / 'FR' / 'users_geo_income_200m_threshold_0.7_SES_USERS_NEW_no_spammers_2_classes.parquet')
    

    # PLOT CELLS WITH SOCIOECONOMIC STATUS ANS USERS ON THE MAP COLOURED ACCORDING TO THEIR SOCIOECONOMIC STATUS

    '''# df with geographical info
    geo_users_path = parent_dir / 'geo_ses_files' / 'FR' / 'user_home_locations_and_attributes_200m.parquet'
    # df with users ses
    ses_users_path_ok = parent_dir / 'geo_ses_files' / 'FR' / 'users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE.parquet'
    users_path = parent_dir / 'geo_ses_files' / 'FR' / 'user_geo_income_200m_threshold_0.7_PARIS_LARGE_LARGE.parquet'
    
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
    map_output_path = parent_dir / 'geo_ses_files' / 'FR' / 'maps 200m' / 'paris LARGE LARGE scaled'/ file_name

    # rename column for merging
    geo_users_df = geo_users_df.rename(columns={'userId': 'user_id'})

    # merge geo and ses df
    merged_df = geo_users_df.merge(ses_users_df, on='user_id')
    merged_df = merged_df.set_geometry(geometry_col)

    # create map
    plot_map_ses(merged_df, num_classes, 'SES_users', map_output_path)'''



    # ---------------------------------------------------------------
    # 5) CREATE SAMPLE OF USERS/TWEETS GEOLOCATED AND WITH THEIR SES
    # ---------------------------------------------------------------
    
    AREA = 'PARIS LARGE LARGE' # FR | PARIS |PARIS LARGE LARGE
    CREATE_PARIS_DF = False # True | False
    GEO_GEOMETRY_COLUMN = 'carreau_geometry'
    no_middle = False

    # read df of users income and ses
    TEST_TRAIN_SAMPLE = 'TRAIN' # TEST | TRAIN
    no_middle = False


    # path geolocated users
    geo_users_path = parent_dir / 'geo_ses_files' / 'FR' / 'user_home_locations_and_attributes_200m.parquet' # user_home_locations_and_attributes_200m.parquet
    # path of df of all tweets of geolocated users after all the cleaning
    tweets_users_geo_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'tweets_geo_users_2014_2018_cleaned_lang_0.7_no_spammers.parquet'
    # path of the df with users, number of tweets per user, geo cell, income, SES
    users_geo_income_ses_path = parent_dir / 'geo_ses_files' / 'FR' / 'users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE.parquet' #'users_geo_income_200m_threshold_0.7_SES_USERS_NEW.parquet'
    
    # paris boundaries file
    paris_boundary_file = '../useful stuff/paris boundaries/paris_boundary_LARGE_LARGE.parquet' # possible files : paris_boundary_75, paris_boundary_LARGE, paris_boundary_LARGE_LARGE
    
    # output path
    paris_users_path = parent_dir / 'geo_ses_files' / 'FR' / 'PARIS_LARGE_LARGE_users_200m_no_spammers.parquet' #'PARIS_users_200m.parquet'
    paris_filtered_users_path = parent_dir / 'geo_ses_files' / 'FR' / 'PARIS_LARGE_LARGE_users_200m_at_leat_5_tweets_lang_FR_no_spammers.parquet' #'PARIS_users_200m_at_leat_5_tweets_lang_FR.parquet'
    #sampled_users_path = parent_dir / 'classification_data' / 'FR' / 'TEST_users_PARIS_PROVA_no_spammers_with_middle.parquet'
    

    geo_users_200m = parent_dir / 'geo_ses_files' / 'FR' / 'user_home_locations_and_attributes_200m.parquet'
    geo_users_iris = parent_dir / 'geo_ses_files' / 'FR' / 'user_home_locations_and_attributes.parquet'
    

    #geo_users_200m_df = pl.scan_parquet(geo_users_200m)
    #geo_users_iris = pl.scan_parquet(geo_users_iris)

    #geo_users_200m_df = geo_users_200m_df.collect()
    #geo_users_iris = geo_users_iris.collect()
    #print(geo_users_iris.columns)

    if AREA == 'PARIS' and CREATE_PARIS_DF == True:

        # read geo users df and df of users income and ses
        geo_users_df = gpd.read_parquet(geo_users_path)
        users_geo_income_ses_df = pd.read_parquet(users_geo_income_ses_path)
        

        # rename column for merging
        geo_users_df = geo_users_df.rename(columns={'userId': 'user_id'})

        # merge df
        merged_df = geo_users_df.merge(users_geo_income_ses_df, on='user_id')

        # drop middle class
        if no_middle == True:
            merged_no_middle_df = merged_df[merged_df['SES_users'] != 'medium']

        # extract Paris cells -> iris grid
        paris_boundary = gpd.read_parquet(paris_boundary_file)
        #paris_boundary = paris_boundary.set_geometry('iris_geometry')

        #todooooooooooooooooooooooooooooooooooooooooooooooooooooo
        # 1. Set correct geometry
        merged_df = merged_df.set_geometry(GEO_GEOMETRY_COLUMN)
        #merged_no_middle_df = merged_no_middle_df.set_geometry(GEO_GEOMETRY_COLUMN)

        # 2. Ensure it's a proper GeoDataFrame
        merged_df = gpd.GeoDataFrame(merged_df, geometry=GEO_GEOMETRY_COLUMN)
        #merged_no_middle_df = gpd.GeoDataFrame(merged_no_middle_df, geometry=GEO_GEOMETRY_COLUMN)

        # 3. Set CRS (since yours is EPSG:3035)
        merged_df.set_crs("EPSG:3035", inplace=True, allow_override=True)
        #merged_no_middle_df.set_crs("EPSG:3035", inplace=True, allow_override=True)
        paris_boundary.set_crs("EPSG:3035", inplace=True, allow_override=True)

        # 4. Spatial join
        paris_cells = gpd.sjoin(merged_df, paris_boundary, predicate="within")

        # extract only Paris cells
        # tot classes
        paris_cells = gpd.sjoin(merged_df, paris_boundary, predicate="intersects")
        # no middle class
        #paris_cells_no_middle = gpd.sjoin(merged_no_middle_df, paris_boundary, predicate="intersects")

        #print(paris_cells_no_middle)
        #print(paris_cells.columns)

        paris_cells = paris_cells.rename(columns={'idINSPIRE_x': 'idINSPIRE', 'SES_users_x': 'SES_users'})
        paris_cells_filtered = paris_cells[['user_id', 'idINSPIRE', 'SES_users']]

        # save all users in paris
        paris_cells_filtered.to_parquet(paris_users_path)


        # filter users in Paris that have at least 5 tweets and with lang == fr

        # red and filter tweets df
        tweets_user_df = pl.scan_parquet(tweets_users_geo_path)
        tweets_user_df = tweets_user_df.filter(pl.col('final_lang') == 'fr')
        tweets_users_grouped_df = (tweets_user_df.group_by('user_id')
                                   .agg(
                                       pl.col('id').count().alias('num_tweets')
                                   )
                                   .sort('num_tweets')
                                   )
        tweets_users_grouped_df = tweets_users_grouped_df.filter((pl.col('num_tweets') >= 5) & (pl.col('num_tweets') <= 15000))

        # read Paris users df
        paris_users_df = pl.scan_parquet(paris_users_path)

        # merge Pars users and and tweets df
        paris_filtered_users_df = paris_users_df.join(tweets_users_grouped_df, on='user_id')

        # save filtered Paris users with lang == fr and at least 5 tweets
        paris_filtered_users_df = paris_filtered_users_df.collect()
        paris_filtered_users_df.write_parquet(paris_filtered_users_path)


    elif AREA == 'PARIS' and CREATE_PARIS_DF == False:

        TEST_TRAIN_SAMPLE = 'TRAIN' # TEST | TRAIN
        no_middle = False

        # paths
        users_geo_income_ses_path = parent_dir / 'geo_ses_files' / 'FR' / 'PARIS_users_200m.parquet'
        suspicious_users_path = parent_dir / 'geo_ses_files' / 'FR' / 'suspicious users' / 'suspicious_users_2_sec.csv'
        if TEST_TRAIN_SAMPLE == 'TRAIN':
            sampled_users_path = parent_dir / 'classification_data' / 'FR' / 'TRAIN_users_PARIS_PROVA_no_spammers_with_middle.parquet'
            maps_path = parent_dir / 'classification_data' / 'FR' / 'maps' / 'TRAIN_users_PARIS_no_spammers_with_middle.png'
            test_users_path = parent_dir / 'classification_data' / 'FR' / 'TEST_users_PARIS_PROVA_no_spammers_with_middle.parquet'
            num_users_per_class = 2100
        if TEST_TRAIN_SAMPLE == 'TEST':
            sampled_users_path = parent_dir / 'classification_data' / 'FR' / 'TEST_users_PARIS_PROVA_no_spammers_with_middle.parquet'
            maps_path = parent_dir / 'classification_data' / 'FR' / 'maps' / 'TEST_users_PARIS_no_spammers_with_middle.png'
            test_users_path = None
            num_users_per_class = 500

        users_geo_income_ses_df = pl.scan_parquet(users_geo_income_ses_path)

        print(users_geo_income_ses_df.collect())

        #original_train = pl.read_parquet(parent_dir / 'classification_data' / 'FR' / 'TRAIN_users_PARIS_PROVA.parquet')
        #original_test = pl.read_parquet(parent_dir / 'classification_data' / 'FR' / 'TEST_users_PARIS_PROVA.parquet')
        #no_spammers_train = pl.read_parquet(parent_dir / 'classification_data' / 'FR' / 'TRAIN_users_PARIS_PROVA_no_spammers.parquet')
        #no_spammers_test = pl.read_parquet(parent_dir / 'classification_data' / 'FR' / 'TEST_users_PARIS_PROVA_no_spammers.parquet')

        # suspicious users df
        #suspicious_users_df = pl.read_csv(suspicious_users_path)
        # remove users with too many tweets
        #suspicious_users = list(suspicious_users_df['user_id'])
        #suspicious_users = [str(x) for x in suspicious_users]
        #filtered_original_train = original_train.filter(~pl.col('user_id').is_in(suspicious_users))
        #filtered_original_test = original_test.filter(~pl.col('user_id').is_in(suspicious_users))
        #filtered_no_spammers_train = no_spammers_train.filter(~pl.col('user_id').is_in(suspicious_users))
        #filtered_no_spammers_test = no_spammers_test.filter(~pl.col('user_id').is_in(suspicious_users))
        
        # set variables
        num_small_tiles = 25 #15
        num_users_big_tile = 300 #100
        
        # read df of users
        df = pl.read_parquet(paris_users_path)

        # read tweets users
        tweets_df = pl.scan_parquet(tweets_users_geo_path)

        # create sample of users
        create_users_sample_paris(paris_filtered_users_path, suspicious_users_path, test_users_path, num_small_tiles, num_users_big_tile, num_users_per_class, no_middle, sampled_users_path)

        # create map of users sample
        users_df = pd.read_parquet(sampled_users_path)
        geo_df = gpd.read_parquet(geo_users_path)
        geo_df = geo_df.rename(columns={'userId': 'user_id'})

        # create df of Paris
        tot_df = geo_df.merge(users_df, on='user_id')
        print(tot_df)
        print(tot_df.columns)
        print(tot_df.geometry.name)

        # Paris lower, middle, high classes

        # define colors
        if no_middle == True:
            color_map = {
                "lower": "blue",
                "high": "red"
            }
        else:
            color_map = {
                "lower": "blue",
                "medium": "green",
                "high": "red"
            }

        # map colors
        tot_df["color"] = tot_df['SES_users'].map(color_map)

        # plot total
        fig, ax = plt.subplots(figsize=(10,10))

        tot_df.plot(
            color=tot_df["color"],
            #edgecolor="black",
            linewidth=0.2,
            ax=ax
        )

        legend_elements = [
            Patch(facecolor="blue", label="Low SES"),
            Patch(facecolor="red", label="High SES")
        ]

        ax.legend(handles=legend_elements, title="SES")
        ax.axis('off')

        fig.savefig(maps_path)
    
    else:

        # paths
        users_geo_income_ses_path = parent_dir / 'geo_ses_files' / 'FR' / 'users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE.parquet'
        suspicious_users_path = parent_dir / 'geo_ses_files' / 'FR' / 'suspicious users' / 'suspicious_users_2_sec.csv'
        if TEST_TRAIN_SAMPLE == 'TRAIN':
            sampled_users_path = parent_dir / 'classification_data' / 'FR' / 'TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_50_tweets_50.parquet'
            maps_path = parent_dir / 'classification_data' / 'FR' / 'maps' / 'TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_50_tweets_50.png'
            test_users_path = parent_dir / 'classification_data' / 'FR' / 'TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_50_tweets_50.parquet'
            num_users_per_class = 750
        if TEST_TRAIN_SAMPLE == 'TEST':
            sampled_users_path = parent_dir / 'classification_data' / 'FR' / 'TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_50_tweets_50.parquet'
            maps_path = parent_dir / 'classification_data' / 'FR' / 'maps' / 'TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_50_tweets_50.png'
            test_users_path = None
            num_users_per_class = 500

        #if TEST_TRAIN_SAMPLE == 'TRAIN':
        #    sampled_users_path = parent_dir / 'classification_data' / 'FR' / 'new_users.parquet'
        #    maps_path = parent_dir / 'classification_data' / 'FR' / 'maps' / 'extra.png'
        #    test_users_path = parent_dir / 'classification_data' / 'FR' / 'tot_df.parquet'
        #    num_users_per_class = 750
        #if TEST_TRAIN_SAMPLE == 'TEST':
        #    sampled_users_path = parent_dir / 'classification_data' / 'FR' / 'TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_50_tweets_50.parquet'
        #    maps_path = parent_dir / 'classification_data' / 'FR' / 'maps' / 'TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_50_tweets_50.png'
        #    test_users_path = None
        #    num_users_per_class = 500
        
        users_geo_income_ses_df = pl.scan_parquet(users_geo_income_ses_path)

        print(users_geo_income_ses_df.collect())

        # set variables
        num_small_tiles = 25 #25 #15
        num_users_big_tile = 100 #300 #100
        num_tweets_per_user = 50 #30
        
        # read df of users
        #df = pl.read_parquet(paris_users_path)

        # read tweets users
        tweets_df = pl.scan_parquet(tweets_users_geo_path)

        # create sample of users
        create_users_sample_paris(users_geo_income_ses_path, suspicious_users_path, test_users_path, num_tweets_per_user, num_small_tiles, num_users_big_tile, num_users_per_class, no_middle, sampled_users_path)

        # create map of users sample
        users_df = pd.read_parquet(sampled_users_path)
        geo_df = gpd.read_parquet(geo_users_path)
        geo_df = geo_df.rename(columns={'userId': 'user_id'})

        # create df of Paris
        tot_df = geo_df.merge(users_df, on='user_id')
        print(tot_df)
        print(tot_df.columns)
        print(tot_df.geometry.name)

        # Paris lower, middle, high classes

        # define colors
        if no_middle == True:
            color_map = {
                "lower": "blue",
                "high": "red"
            }
        else:
            color_map = {
                "lower": "blue",
                "medium": "green",
                "high": "red"
            }

        # map colors
        tot_df["color"] = tot_df['SES_users'].map(color_map)

        # plot total
        fig, ax = plt.subplots(figsize=(10,10))

        tot_df.plot(
            color=tot_df["color"],
            #edgecolor="black",
            linewidth=0.2,
            ax=ax
        )

        legend_elements = [
            Patch(facecolor="blue", label="Low SES"),
            Patch(facecolor="red", label="High SES")
        ]

        ax.legend(handles=legend_elements, title="SES")
        ax.axis('off')

        fig.savefig(maps_path)


    '''
    # ---------------------------------------------------------------
    # 5) CREATE SAMPLE OF USERS/TWEETS GEOLOCATED AND WITH THEIR SES
    # ---------------------------------------------------------------

    # samples users paths
    sample_users_train_path = parent_dir / 'classification_data' / 'FR' / 'TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets.parquet' # TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_5_tweets_50
    sample_users_test_path = parent_dir / 'classification_data' / 'FR' / 'TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets.parquet'

    # cleaned tweets path
    tweets_users_geo_path = parent_dir / 'twitter_home_location_FR' / 'tweets_geolocated_users_unified' / 'cleaning_steps_files' / 'tweets_geo_users_2014_2018_cleaned_lang_0.7_no_spammers.parquet'

    # output path
    sample_tweets_train_path = parent_dir / 'classification_data' / 'FR' / 'tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_tot.csv'
    sample_tweets_test_path = parent_dir / 'classification_data' / 'FR' / 'tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_tot.csv'

    # number of tweets per user
    num_tweets_per_user = 30

    # extract tweets per user
    #sample_tweets_train_df = sample_tweets_per_user(sample_users_train_path, tweets_users_geo_path, num_tweets_per_user)
    #sample_tweets_test_df = sample_tweets_per_user(sample_users_test_path, tweets_users_geo_path, num_tweets_per_user)
    
    # extract tot tweets per user
    sample_tweets_train_df = sample_tweets_per_user_tot(sample_users_train_path, tweets_users_geo_path)
    sample_tweets_test_df = sample_tweets_per_user_tot(sample_users_test_path, tweets_users_geo_path)

    # keep only useful columns
    sample_tweets_train_df = sample_tweets_train_df[['user_id', 'SES_users', 'num_tweets_per_user', 'id', 'clean_light']]
    sample_tweets_test_df = sample_tweets_test_df[['user_id', 'SES_users', 'num_tweets_per_user', 'id', 'clean_light']]

    print(sample_tweets_train_df)
    print(sample_tweets_test_df)

    # save sampled tweets
    sample_tweets_train_df.write_csv(sample_tweets_train_path)
    sample_tweets_test_df.write_csv(sample_tweets_test_path)
    '''

    # ---------------------------------------------------------------
    # 6) RE-MAP 3 CLASSES INTO 2 CLASSES (IF NEEDED)
    # ---------------------------------------------------------------

    '''user_ses_path = '../geo_ses_files/FR/users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE_2_classes.parquet'
    user_ses_df = pl.read_parquet(user_ses_path)
    # keep only useful columns
    user_ses_df = user_ses_df.select(['user_id', 'SES_users'])
    #user_ses_df = user_ses_df.rename({'SES_users': 'SES_users_NEW'})

    train_df = pl.read_csv('../classification_data/FR/tweets 30 per user/train_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION.csv')
    dev_df = pl.read_csv('../classification_data/FR/tweets 30 per user/test_dev_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION.csv')
    test_df = pl.read_csv('../classification_data/FR/tweets 30 per user/test_test_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION.csv')

    # drop old SES column
    train_df = train_df.drop('SES_users')
    dev_df = dev_df.drop('SES_users')
    test_df = test_df.drop('SES_users')

    # transform user_id into string
    train_df = train_df.with_columns(
        pl.col('user_id').cast(pl.String)
    )
    dev_df = dev_df.with_columns(
        pl.col('user_id').cast(pl.String)
    )
    test_df = test_df.with_columns(
        pl.col('user_id').cast(pl.String)
    )

    # join to get new SES
    train_df = train_df.join(user_ses_df, on='user_id')
    dev_df = dev_df.join(user_ses_df, on='user_id')
    test_df = test_df.join(user_ses_df, on='user_id')

    # check the size of the two classes
    print(len(train_df.filter(pl.col('SES_users') == 'lower')))
    print(len(dev_df.filter(pl.col('SES_users') == 'lower')))
    print(len(test_df.filter(pl.col('SES_users') == 'lower')))

    # save dfs
    train_df.write_csv('../classification_data/FR/tweets 30 per user/train_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION_2_classes.csv')
    dev_df.write_csv('../classification_data/FR/tweets 30 per user/test_dev_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION_2_classes.csv')
    test_df.write_csv('../classification_data/FR/tweets 30 per user/test_test_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION_2_classes.csv')'''