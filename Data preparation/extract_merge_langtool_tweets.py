import re
import polars as pl

#train_df = pl.scan_csv('../classification_data/FR/tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_5_tweets_50.csv')
#test_df = pl.scan_csv('../classification_data/FR/tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_5_tweets_50.csv')

#train_df = pl.scan_csv('../classification_data/FR/tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_text_chosen.csv')
#test_df = pl.scan_csv('../classification_data/FR/tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_5_tweets_50.csv')


'''This script helps optimize the process of sending tweets to LanguageTool, identifying tweets with already detected errors with the tool and the ones still need to be sent'''

# -----------------------------------------------------------------------
# PART 1 -> EXTRACT TWEETS WITH ALREADY LANGTOOL ERRORS AND ONES WITHOUT
# -----------------------------------------------------------------------

train_df = pl.scan_csv('../classification_data/FR/new_users_tweets.csv')#'../classification_data/FR/sampled 30 tweets clustering/selected_tweets_30_train.csv'
test_df = pl.scan_csv('../classification_data/FR/new_users_tweets.csv')#'../classification_data/FR/sampled 30 tweets clustering/selected_tweets_30_test.csv'


#train_df = train_df.rename({'tweet_id': 'id'})


# read files with already langtool errors
# paris big group tweets
errors_prev_df = pl.scan_parquet('../classification_data/FR/chunks_paris_merged_enriched/chunks_paris_merged_enriched_until_492.parquet')
# sampled tweets 1
train_errors_df_1 = pl.scan_parquet('../classification_data/FR/chunks_paris_LARGE_LARGE_merged_enriched/train_paris_LARGE_LARGE_with_medium.parquet')
test_errors_df_1 = pl.scan_parquet('../classification_data/FR/chunks_paris_LARGE_LARGE_merged_enriched/test_paris_LARGE_LARGE_with_medium.parquet')
# sampled tweets 2 (users with at least 30 tweets)
train_errors_df_2 = pl.scan_csv('../classification_data/FR/TRAIN_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv')
dev_errors_df_2 = pl.scan_csv('../classification_data/FR/TEST_DEV_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv')
test_errors_df_2 = pl.scan_csv('../classification_data/FR/TEST_TEST_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv')
test_errors_df_2 = pl.concat([dev_errors_df_2, test_errors_df_2])
train_merging_df = train_df.rename({'id': 'tweet_id'})
test_merging_df = train_df.rename({'id': 'tweet_id'})

'''if 'id ' in train_df.columns:
    print('ciao')
    train_merging_df = train_df.rename({'id': 'tweet_id'})
else:
    train_merging_df = train_df

if 'id ' in test_df.columns:
    test_merging_df = test_df.rename({'id': 'tweet_id'})
else:
    test_merging_df = test_df'''

# cast to string for merging

errors_prev_df = errors_prev_df.with_columns(
    pl.col('tweet_id').cast(pl.String)
)

train_merging_df = train_merging_df.with_columns(
    pl.col('tweet_id').cast(pl.String)
)
test_merging_df = test_merging_df.with_columns(
    pl.col('tweet_id').cast(pl.String)
)

train_errors_df_1 = train_errors_df_1.with_columns(
    pl.col('tweet_id').cast(pl.String)
)
test_errors_df_1 = test_errors_df_1.with_columns(
    pl.col('tweet_id').cast(pl.String)
)

train_errors_df_2 = train_errors_df_2.with_columns(
    pl.col('tweet_id').cast(pl.String)
)
test_errors_df_2 = test_errors_df_2.with_columns(
    pl.col('tweet_id').cast(pl.String)
)


#train_merging_df = train_df.drop(['num_tweets_per_user', 'TWEET_NUM_WORDS', 'VOCAB_SIZE', 'has_standard_neg', 'has_nonstandard_neg', 'standard_plural', 'non_standard_plural'])
#test_merging_df = test_df.drop(['num_tweets_per_user', 'TWEET_NUM_WORDS', 'VOCAB_SIZE', 'has_standard_neg', 'has_nonstandard_neg', 'standard_plural', 'non_standard_plural'])

# prepare df to merge
errors_prev_df = errors_prev_df.select(['tweet_id', 'AGREEMENT','CASING','CAT_ELISION','CAT_GRAMMAIRE','CAT_HOMONYMES_PARONYMES','CAT_MAJUSCULES', 'CAT_MARQUES_DE_COMMERCE',
                                        'CAT_PLEONASMES','CAT_REGIONALISMES','CAT_REGLES_DE_BASE','CAT_TOURS_CRITIQUES','CAT_TYPOGRAPHIE','MISC','MULTITOKEN_SPELLING',
                                        'PONCTUATION_POINT','PONCTUATION_VIRGULE','PUNCTUATION','REPETITIONS_STYLE','SEMANTICS','STYLE','TYPOGRAPHY','TYPOS', 'PONCTUATION_STYLE_OS'])
train_errors_df_1 = train_errors_df_1.drop(['user_id', 'SES_users'])
test_errors_df_1 = test_errors_df_1.drop(['user_id', 'SES_users'])
train_errors_df_2 = train_errors_df_2.drop(['user_id', 'SES_users'])
test_errors_df_2 = test_errors_df_2.drop(['user_id', 'SES_users'])
# concatenate them and remove duplicated tweets
tot_errors_df = pl.concat([errors_prev_df, train_errors_df_1, test_errors_df_1, train_errors_df_2, test_errors_df_2], how='diagonal_relaxed')#


#tot_errors_df = tot_errors_df.collect()
tot_errors_df = tot_errors_df.unique(subset=['tweet_id'], keep='first')

# drop columns that are not useful for merging
tot_errors_df = tot_errors_df.drop(['text', 'text_right', 'Unnamed: 0'])

merged_train_df = train_merging_df.join(tot_errors_df, on='tweet_id', how='inner')

merged_test_df = test_merging_df.join(tot_errors_df, on='tweet_id', how='inner')

df_missing_errors_train = train_merging_df.filter(
    ~pl.col('tweet_id').is_in(merged_train_df.select("tweet_id").collect()["tweet_id"])
)
df_missing_errors_test = test_merging_df.filter(
    ~pl.col('tweet_id').is_in(merged_test_df.select("tweet_id").collect()["tweet_id"])
)

print('ok train', len(list(merged_test_df.collect()['user_id'].unique())))
print('missing train', len(list(df_missing_errors_train.collect()['user_id'].unique())))

if 'clean_light' in df_missing_errors_train.columns:
    df_missing_errors_train = df_missing_errors_train.rename({'clean_light': 'text'})
if 'clean_light' in df_missing_errors_test.columns:
    df_missing_errors_test = df_missing_errors_test.rename({'clean_light': 'text'})

# save tweets with already langtool errors
merged_train_df.collect().write_parquet('../errors df/new_users_tweets_ok.parquet')
#merged_test_df.collect().write_parquet('../errors df/test_with_errors_30_tweets_chosen_correct.parquet')

# save tweets without langtool errors
df_missing_errors_train.collect().write_parquet('../errors df/new_users_tweets_langtool.parquet')
#df_missing_errors_test.collect().write_parquet('../errors df/test_for_langtool_30_tweets_chosen_correct.parquet')




# -----------------------------------------------------------------------------
# PART 2 -> MERGE TWEETS WITH ALREADY AND NEW LANGTOOL ERRORS , ADD ALSO TEXT
# -----------------------------------------------------------------------------

# input paths
# train
df_old_errors_train = pl.read_parquet('../errors df/train_with_errors_30_tweets_chosen_correct.parquet') # train_with_errors_30_tweets_chosen_NEW_3_classes
df_new_errors_train = pl.read_parquet('../errors df/train_30_tweets_chosen_correct.parquet') # train_with_NEW_errors_30_tweets_chosen_NEW_3_classes
df_to_langtool_train = pl.read_parquet('../errors df/train_for_langtool_30_tweets_chosen_correct.parquet') # train_for_langtool_30_tweets_chosen_NEW_3_classes
# test
df_old_errors_test = pl.read_parquet('../errors df/test_with_errors_30_tweets_chosen_correct.parquet')
df_new_errors_test = pl.read_parquet('../errors df/test_30_tweets_chosen_correct.parquet')
df_to_langtool_test = pl.read_parquet('../errors df/test_for_langtool_30_tweets_chosen_correct.parquet')
# tweets df to add text
tweets_path = '../twitter_home_location_FR/tweets_geolocated_users_unified/cleaning_steps_files/tweets_geo_users_2014_2018_cleaned_lang_0.7_no_spammers.parquet'

# output paths
output_parquet_train = '../classification_data/FR/TWEETS_30_chosen_correct_LANGTOOL/TWEETS_30_chosen_correct_LANGTOOLS_train.parquet'
output_csv_train = '../classification_data/FR/TWEETS_30_chosen_correct_LANGTOOL/TWEETS_30_chosen_correct_LANGTOOLS_train.csv'
output_parquet_test = '../classification_data/FR/TWEETS_30_chosen_correct_LANGTOOL/TWEETS_30_chosen_correct_LANGTOOLS_test.parquet'
output_csv_test = '../classification_data/FR/TWEETS_30_chosen_correct_LANGTOOL/TWEETS_30_chosen_correct_LANGTOOLS_test.csv'

# check which tweets have no errors (difference between df_to_langtool and df_new_errors_test)
# check which columns mismatch type between the two datasets
#for col in df_new_errors_test.columns:
#    print(
#        col,
#        df_old_errors_test[col].dtype,
#        df_new_errors_test[col].dtype
#    )


# TWEETS DF
tweets_df = pl.read_parquet(tweets_path, columns=['id', 'clean_light'])
tweets_df = tweets_df.rename({'id': 'tweet_id'})

# TRAIN

# extracts float columns from old df to convert into int
float_cols_train = [
    col for col, dtype in df_old_errors_train.schema.items()
    if dtype in (pl.Float32, pl.Float64)
]
df_old_errors_train = df_old_errors_train.with_columns(
    [pl.col(col).cast(pl.Int64) for col in float_cols_train]
)

# convert into string users ids
df_old_errors_train = df_old_errors_train.with_columns(pl.col('user_id').cast(pl.String))
df_to_langtool_train = df_to_langtool_train.with_columns(pl.col('user_id').cast(pl.String))

# drop useless columns
df_new_errors_train = df_new_errors_train.drop(['SES_users', 'SES_users_2_classes', 'final_lang', 'ind_snv_mean'])

# concatenate old and new df with errors
concat_train_df = pl.concat([df_old_errors_train.select(df_new_errors_train.columns), df_new_errors_train])

# concat also tweets with no errors
# from initial df to send to langtool, keep only rows that are not in tweets for which we have langtool errors (new)
df_to_langtool_train_filtered = df_to_langtool_train.join(
    df_new_errors_train.select('tweet_id'),
    on='tweet_id',
    how='anti'
)

# columns expected from df1
cols_train = concat_train_df.columns
# keep only df1 columns and add missing ones
df2_aligned_train = df_to_langtool_train_filtered.select([
    pl.col(c) if c in df_to_langtool_train_filtered.columns else pl.lit(None).alias(c)
    for c in cols_train
])

# concatenate
concat_train_df = pl.concat([concat_train_df, df2_aligned_train])

# replace null with 0
concat_train = concat_train_df.fill_null(0)

# add text
concat_train_df = concat_train_df.join(tweets_df, on='tweet_id')

# save df train
concat_train_df.write_parquet(output_parquet_train)
concat_train_df.write_csv(output_csv_train)



# TEST

# extracts float columns from old df to convert into int
float_cols_test = [
    col for col, dtype in df_old_errors_test.schema.items()
    if dtype in (pl.Float32, pl.Float64)
]
df_old_errors_test = df_old_errors_test.with_columns(
    [pl.col(col).cast(pl.Int64) for col in float_cols_test]
)

# convert into string users ids
df_old_errors_test = df_old_errors_test.with_columns(pl.col('user_id').cast(pl.String))
df_to_langtool_test = df_to_langtool_test.with_columns(pl.col('user_id').cast(pl.String))

# drop useless columns
df_new_errors_test = df_new_errors_test.drop(['SES_users', 'SES_users_2_classes', 'final_lang', 'ind_snv_mean'])

#merged_train_df = df_old_errors_train.join(df_new_errors_train)
# concat tweets with already and new langtool errors
concat_test_df = pl.concat([df_old_errors_test.select(df_new_errors_test.columns), df_new_errors_test])

# concat also tweets with no errors
# from initial df to send to langtool, keep only rows that are not in tweets for which we have langtool errors (new)
df_to_langtool_test_filtered = df_to_langtool_test.join(
    df_new_errors_test.select('tweet_id'),
    on='tweet_id',
    how='anti'
)

# columns expected from df1
cols_test = concat_test_df.columns
# keep only df1 columns and add missing ones
df2_aligned_test = df_to_langtool_test_filtered.select([
    pl.col(c) if c in df_to_langtool_test_filtered.columns else pl.lit(None).alias(c)
    for c in cols_test
])

# concatenate
concat_test_df = pl.concat([concat_test_df, df2_aligned_test])

# replace null with 0
concat_test_df = concat_test_df.fill_null(0)

# add text
concat_test_df = concat_test_df.join(tweets_df, on='tweet_id')

# save df test
concat_test_df.write_parquet(output_parquet_test)
concat_test_df.write_csv(output_csv_test)




# ADD INCOME and SES

'''# input paths
input_train = '../clasification_data/FR/new_users_tweets.csv' #'../classification_data/FR/TWEETS_30_chosen_correct_LANGTOOL/TWEETS_30_chosen_correct_LANGTOOLS_train.csv' #'../classification_data/FR/TWEETS_30_chosen_NEW_3_classes_LANGTOOL_with_other_features/tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN.csv'
input_test = '../clasification_data/FR/new_users_tweets.csv'
 #'../classification_data/FR/TWEETS_30_chosen_NEW_3_classes_LANGTOOL_with_other_features/tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN.csv'
users_info = '../geo_ses_files/FR/user_geo_income_200m_threshold_0.7_PARIS_LARGE_LARGE.parquet'
users_3_classes_path = '../geo_ses_files/FR/users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE.parquet'
users_2_classes_path = '../geo_ses_files/FR/users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE_2_classes.parquet'

# output paths
output_train_3_classes = '../classification_data/FR/new_users_tweets_.csv' #'../classification_data/FR/TWEETS_30_chosen_NEW_3_classes_LANGTOOL_with_other_features/tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_income.csv'
output_test_3_classes = '../classification_data/FR/TWEETS_30_chosen_correct_LANGTOOL/TWEETS_30_chosen_correct_LANGTOOLS_test_with_SES_income.csv' #'../classification_data/FR/TWEETS_30_chosen_NEW_3_classes_LANGTOOL_with_other_features/tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_income.csv'
output_train_2_classes = '../classification_data/FR/TWEETS_30_chosen_correct_LANGTOOL/TWEETS_30_chosen_correct_LANGTOOLS_train_with_SES_income_2_classes.csv' #'../classification_data/FR/TWEETS_30_chosen_NEW_3_classes_LANGTOOL_with_other_features/tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_income.csv'
output_test_2_classes = '../classification_data/FR/TWEETS_30_chosen_correct_LANGTOOL/TWEETS_30_chosen_correct_LANGTOOLS_test_with_SES_income_2_classes.csv' #'../classification_data/FR/TWEETS_30_chosen_NEW_3_classes_LANGTOOL_with_other_features/tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_income.csv'


# read files
input_train_df = pl.read_csv(input_train)
input_train_df = df = input_train_df.with_columns(
    pl.col("user_id").cast(pl.String)
)
input_test_df = pl.read_csv(input_test)
input_test_df = df = input_test_df.with_columns(
    pl.col("user_id").cast(pl.String)
)
#users_info_df = pl.read_parquet(users_info, columns=['user_id', 'ind_snv_mean'])
users_3_classes = pl.read_parquet(users_3_classes_path).select(['user_id', 'SES_users', 'ind_snv_mean'])
users_2_classes = pl.read_parquet(users_2_classes_path).select(['user_id', 'SES_users', 'ind_snv_mean'])
#users_2_classes = users_2_classes.rename({'SES_users': 'SES_users_2_classes'})

# merge files
merged_train_3 = input_train_df.join(users_3_classes, on='user_id')
merged_test_3 = input_test_df.join(users_3_classes, on='user_id')
merged_train_2 = input_train_df.join(users_2_classes, on='user_id')
merged_test_2 = input_test_df.join(users_2_classes, on='user_id')

# save results
merged_train_3.write_csv(output_train_3_classes)
merged_test_3.write_csv(output_test_3_classes)
merged_train_2.write_csv(output_train_2_classes)
merged_test_2.write_csv(output_test_2_classes)
'''

'''merged_train_df = merged_train_df.collect()
merged_test_df = merged_test_df.collect()
train_errors_df = train_errors_df.collect()
test_errors_df = test_errors_df.collect()'''



'''train_tot_errors_df = pl.concat([train_errors_df, merged_train_df], how='diagonal')
test_tot_errors_df = pl.concat([test_errors_df, merged_test_df], how='diagonal')

print(train_tot_errors_df)'''
'''print('lower')
print(train_tot_errors_df.collect().filter(pl.col('SES_users') == 'lower'))
print(test_tot_errors_df.collect().filter(pl.col('SES_users') == 'lower'))
print('medium')
print(train_tot_errors_df.collect().filter(pl.col('SES_users') == 'medium'))
print(test_tot_errors_df.collect().filter(pl.col('SES_users') == 'medium'))
print('high')
print(train_tot_errors_df.collect().filter(pl.col('SES_users') == 'high'))
print(test_tot_errors_df.collect().filter(pl.col('SES_users') == 'high'))'''



'''errors_missing_train_df = train_merging_df.join(
    train_tot_errors_df.select(["tweet_id"]),
    on="tweet_id",
    how="anti"
)

errors_missing_test_df = test_merging_df.join(
    test_tot_errors_df.select(["tweet_id"]),
    on="tweet_id",
    how="anti"
)

lang_tool_train_df = errors_missing_train_df.collect()
lang_tool_test_df = errors_missing_test_df.collect()

print(lang_tool_train_df)
print(lang_tool_test_df)

# save df for lang tool
lang_tool_train_df.write_parquet('../tweets_TRAIN_for_lang_tool_round_2.parquet')
lang_tool_test_df.write_parquet('../tweets_TEST_for_lang_tool_round_2.parquet')

'''