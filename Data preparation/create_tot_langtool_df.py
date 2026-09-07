import pandas as pd
import polars as pl
import os


'''This script has as input the files obtained from LanguageTool and creates the final dataset with all the users and the number of errors for 
each category, it also adds tweets text and SES ground truth for each user. The final dataset is saved in parquet format.'''


# paths

# directory and files with languagetool results
langtool_chunks_dir = '../classification_data/FR/new_users_tweets_langtool_done/' #'../classification_data/FR/TWEETS_TRAIN_30_chosen_correct/'  #'../lang_tool_chunks_paris_no_spammers/' '../classification_data/FR/TWEETS_TRAIN_30_chosen_NEW_3_classes/' # '../classification_data/FR/tweets_NEW_TEST_OK/' 
files = os.listdir(langtool_chunks_dir)

# total users tweets
users_tot_tweets_path = 'to_lang_tool.parquet'

# tweets
tweets_path = '../twitter_home_location_FR/tweets_geolocated_users_unified/cleaning_steps_files/tweets_geo_users_2014_2018_cleaned_lang_0.7_no_spammers.parquet'

# users df with SES ground truth -> 2-class and 3-class settings
users_3_classes_path = '../geo_ses_files/FR/users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE.parquet'
users_2_classes_path = '../geo_ses_files/FR/users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE_2_classes.parquet'


# read data
users_3_classes = pl.read_parquet(users_3_classes_path).select(['user_id', 'SES_users', 'ind_snv_mean'])
users_2_classes = pl.read_parquet(users_2_classes_path).select(['user_id', 'SES_users'])
users_2_classes = users_2_classes.rename({'SES_users': 'SES_users_2_classes'})

tot_df = pd.DataFrame()

users_tot_tweets_df = pl.read_parquet(users_tot_tweets_path)
users_tot_tweets_df = users_tot_tweets_df.select(['tweet_id', 'user_id'])
users_tot_tweets_df = users_tot_tweets_df.with_columns(
        pl.col('user_id').cast(pl.String)
    )


# set output path
OUTPUT_PATH = '../errors df/new_users_tweets_test.parquet' #'../classification_data/FR/chunks_paris_merged_enriched/chunks_paris_merged_enriched_until_492.parquet'../errors df/train_with_NEW_errors_30_tweets_chosen_NEW_3_classes'../classification_data/FR/chunks_paris_merged_enriched/chunks_paris_merged_enriched_until_444.parquet' #'../classification_data/FR/chunks_paris_LARGE_LARGE_merged_enriched/test_paris_LARGE_LARGE_with_medium.parquet' 


# iterate over considered files

i = 0
dfs = []

for file in files:

    print('File', str(i+1))

    df = pd.read_parquet(langtool_chunks_dir + file)

    df = df.reset_index()
    
    # Step 1: pivot with aggregation
    pivot_df = df.pivot_table(
        index=['tweet_id', 'user_id'],#, 'SES_users'
        columns='category',
        values='count',
        aggfunc='sum',
        fill_value=0
    )

    # Step 2: flatten index
    pivot_df = pivot_df.reset_index()

    # Optional: remove column name
    pivot_df.columns.name = None

    # extract all categories and add related columns and fill with 0 where it is nan
    all_categories = sorted(df['category'].unique())
    pivot_df = pivot_df.reindex(columns=['tweet_id', 'user_id'] + all_categories, fill_value=0) #, 'SES_users'

    # add text
    df_tweets = pl.scan_parquet(tweets_path)
    df_tweets = df_tweets.rename({'id': 'tweet_id'})
    pivot_df = pl.from_pandas(pivot_df).lazy()
    pivot_df = pivot_df.with_columns(
        pl.col('user_id').cast(pl.String)
    )
    
    pivot_total_df = pivot_df

    # set 0 instead of null where missing values for certain categories
    pivot_total_df = pivot_total_df.fill_null(0)

    # save dfs in a list
    dfs.append(pivot_df)

    i += 1

# fill nan with 0
tot_df = (pl.concat(dfs, how='diagonal').fill_null(0)).collect()


# extract unique tweets ids
tweets_langtool = set(list(tot_df['tweet_id']))
tweets_tot_users = set(list(users_tot_tweets_df['tweet_id']))


# add users with no mistakes
tweets_no_mistakes = users_tot_tweets_df.filter(pl.col('tweet_id').is_in(list(tweets_tot_users - tweets_langtool)))
tot_df = pl.concat([tot_df, tweets_no_mistakes], how='diagonal').fill_null(0)
tot_df = tot_df.sort('user_id')

# add information of user's belonging class in the 2-class and 3-class settings
tot_df = users_3_classes.join(tot_df, on='user_id', how='inner')
tot_df = users_2_classes.join(tot_df, on='user_id', how='inner')
tot_df = tot_df.with_columns(pl.col("tweet_id").cast(pl.Utf8))
tot_df = tot_df.lazy()

# join with tweets df
tot_df = (df_tweets.join(tot_df, on='tweet_id', how='inner'))
# drop non-useful columns for training
tot_df = tot_df.drop('created_at', 'clean_full', 'num_words_full', 'lang', 'user_id_right')
# rename text column to be consistent with following scripts
tot_df = tot_df.rename({'clean_light': 'text'})


tot_df = tot_df.collect()


# check for duplicates

dupl=(
    tot_df.group_by("user_id")
    .agg(
        pl.col("text").count().alias("total_tweets"),
        pl.col("text").n_unique().alias("unique_tweets")
    )
    .with_columns(
        (pl.col("total_tweets") - pl.col("unique_tweets")).alias("n_duplicates")
    )
    .filter(pl.col("n_duplicates") > 0)
    .sort("n_duplicates", descending=True)
)

print(dupl)

dupl1 = (
    tot_df.group_by("tweet_id")
    .agg(pl.col("user_id").n_unique().alias("n_users"))
    .filter(pl.col("n_users") > 1)
    .sort("n_users", descending=True)
)

print(dupl1)

# save final df
tot_df.write_parquet(OUTPUT_PATH)

print(tot_df.columns)