import polars as pl

def split_dev_test_2_classes_old(df):

    # unique users
    users = df.select("user_id").unique()

    # shuffle users
    users = users.sample(fraction=1.0, shuffle=True)

    # split users
    n = users.height // 2

    users_1 = users.head(n)
    users_2 = users.slice(n)

    # filter tweets
    df_1 = df.join(users_1, on="user_id", how="inner")
    df_2 = df.join(users_2, on="user_id", how="inner")

    return df_1, df_2

    
def split_dev_test_2_classes(df):

    # unique users
    users = df.select(["user_id", "SES_users"]).unique()
    users_low = users.filter(pl.col('SES_users') == 'lower')
    users_high = users.filter(pl.col('SES_users') == 'high')

    # shuffle users
    users_low = users_low.sample(fraction=1.0, shuffle=True)
    users_high = users_high.sample(fraction=1.0, shuffle=True)

    # split users
    n_low = users_low.height // 2
    n_high = users_high.height // 2

    users_1_low = users_low.head(n_low)
    users_2_low = users_low.slice(n_low)

    users_1_high = users_high.head(n_high)
    users_2_high = users_high.slice(n_high)

    # concatenate users and shuffle
    users_1 = pl.concat([users_1_low, users_1_high])
    users_2 = pl.concat([users_2_low, users_2_high])
    users_1 = users_1.sample(fraction=1.0, shuffle=True)
    users_2 = users_2.sample(fraction=1.0, shuffle=True)

    # filter tweets
    df_1 = df.join(users_1, on="user_id", how="inner")
    df_2 = df.join(users_2, on="user_id", how="inner")

    return df_1, df_2


def split_dev_test_3_classes(df):

    # unique users
    users = df.select(["user_id", "SES_users"]).unique()
    users_low = users.filter(pl.col('SES_users') == 'lower')
    users_medium = users.filter(pl.col('SES_users') == 'medium')
    users_high = users.filter(pl.col('SES_users') == 'high')

    # shuffle users
    users_low = users_low.sample(fraction=1.0, shuffle=True)
    users_medium = users_medium.sample(fraction=1.0, shuffle=True)
    users_high = users_high.sample(fraction=1.0, shuffle=True)

    # split users
    n_low = users_low.height // 2
    n_medium = users_medium.height // 2
    n_high = users_high.height // 2

    users_1_low = users_low.head(n_low)
    users_2_low = users_low.slice(n_low)

    users_1_medium = users_medium.head(n_medium)
    users_2_medium = users_medium.slice(n_medium)

    users_1_high = users_high.head(n_high)
    users_2_high = users_high.slice(n_high)

    # concatenate users and shuffle
    users_1 = pl.concat([users_1_low, users_1_medium, users_1_high])
    users_2 = pl.concat([users_2_low, users_2_medium, users_2_high])
    users_1 = users_1.sample(fraction=1.0, shuffle=True)
    users_2 = users_2.sample(fraction=1.0, shuffle=True)

    # filter tweets
    df_1 = df.join(users_1, on="user_id", how="inner")
    df_2 = df.join(users_2, on="user_id", how="inner")

    return df_1, df_2


if __name__ == "__main__":


    # EXTRACT TEST/DEV SETS
    
    # from parquet to csv -> train set
    '''INPUT_PATH_TRAIN = 'dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium.parquet'
    OUTPUT_TRAIN_DIR = 'dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium.csv'

    df_train = pl.read_parquet(INPUT_PATH_TRAIN)
    df_train = df_train.drop('text')
    df_train = df_train.rename({'text_right': 'text'})
    df_train.write_csv(OUTPUT_TRAIN_DIR)'''

    num_classes = 3
    
    if num_classes == 3:
        INPUT_PATH_TEST = 'dev_phase_normal/input with medium large/tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT.parquet'
        OUTPUT_DEV_DIR = 'dev_phase_normal/input with medium large/tweets_TEST_DEV_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT.csv'
        OUTPUT_TEST_DIR = 'dev_phase_normal/input with medium large/tweets_TEST_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT.csv'
    elif num_classes == 2:
        INPUT_PATH_TEST = 'dev_phase_normal/input with medium large/tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes.parquet'
        OUTPUT_DEV_DIR = 'dev_phase_normal/input with medium large/tweets_TEST_DEV_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes.csv'
        OUTPUT_TEST_DIR = 'dev_phase_normal/input with medium large/tweets_TEST_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes.csv'

    df = pl.read_parquet(INPUT_PATH_TEST)

    if num_classes == 3:
        dev_df, test_df = split_dev_test_3_classes(df)
    elif num_classes == 2:
        dev_df, test_df = split_dev_test_2_classes(df)

    #dev_df = dev_df.drop('text')
    #dev_df = dev_df.rename({'text_right': 'text'})

    #test_df = test_df.drop('text')
    #test_df = test_df.rename({'text_right': 'text'})

    print(len(dev_df))
    print(len(test_df))

    print(len(dev_df['user_id'].unique()))
    print(len(test_df['user_id'].unique()))

    print(len(test_df.filter(pl.col('SES_users') == 'lower')))
    print(len(test_df.filter(pl.col('SES_users') == 'medium')))
    print(len(test_df.filter(pl.col('SES_users') == 'high')))

    '''dev_df.write_csv(OUTPUT_DEV_DIR)
    test_df.write_csv(OUTPUT_TEST_DIR)'''

    # not useful
    '''df_train_0 = pl.read_csv('dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors.csv')
    df_dev_0   = pl.read_csv('dev_phase_normal/input with medium large/test_dev_paris_LARGE_LARGE_tot_errors.csv')
    df_test_0  = pl.read_csv('dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors.csv')
    tot_0 = pl.concat([df_train_0, df_dev_0, df_test_0], how='diagonal')

    df_train_1 = pl.read_csv('dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium.csv')
    df_dev_1 = pl.read_csv('dev_phase_normal/input with medium large/test_dev_paris_LARGE_LARGE_tot_errors_with_medium.csv')
    df_test_1 = pl.read_csv('dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium.csv')
    tot_1 = pl.concat([df_train_1, df_dev_1, df_test_1], how='diagonal')

    low_0 = tot_0.filter(pl.col("SES_users") == "lower")
    medium_0   = tot_0.filter(pl.col("SES_users") == "medium")
    high_0  = tot_0.filter(pl.col("SES_users") == "high")

    low_1 = tot_1.filter(pl.col("SES_users") == "lower")
    medium_1   = tot_1.filter(pl.col("SES_users") == "medium")
    high_1  = tot_1.filter(pl.col("SES_users") == "high")
    
    cols = ["AGREEMENT", "CASING", "CAT_ELISION", "CAT_GRAMMAIRE", "CAT_HOMONYMES_PARONYMES", "CAT_MAJUSCULES", "CAT_PLEONASMES", "CAT_REGLES_DE_BASE", "CAT_TOURS_CRITIQUES", "CAT_TYPOGRAPHIE", "MISC", "MULTITOKEN_SPELLING", "PONCTUATION_POINT", "PONCTUATION_VIRGULE", "PUNCTUATION", "REPETITIONS_STYLE", "SEMANTICS", "STYLE", "TYPOGRAPHY", "TYPOS", "CAT_MARQUES_DE_COMMERCE", "CAT_REGIONALISMES"]
    
    print((low_0.select(pl.sum_horizontal(cols) != 0)).sum().item())
    print((medium_0.select(pl.sum_horizontal(cols) != 0)).sum().item())
    print((high_0.select(pl.sum_horizontal(cols) != 0)).sum().item())

    print((low_1.select(pl.sum_horizontal(cols) != 0)).sum().item())
    print((medium_1.select(pl.sum_horizontal(cols) != 0)).sum().item())
    print((high_1.select(pl.sum_horizontal(cols) != 0)).sum().item())'''


    # ADD INCOME FOR REGRESSION

    '''TRAIN_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium.csv'
    DEV_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/test_dev_paris_LARGE_LARGE_tot_errors_with_medium.csv'
    TEST_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium.csv'

    # read df
    income_ses_df = pl.read_parquet('users_geo_income_200m_SES_USERS_PARIS_LARGE_LARGE.parquet')
    train_df = pl.read_csv(TRAIN_FINAL_SES_ERRORS_PATH)
    dev_df = pl.read_csv(DEV_FINAL_SES_ERRORS_PATH)
    test_df = pl.read_csv(TEST_FINAL_SES_ERRORS_PATH)

    # cast into string
    train_df = train_df.with_columns(pl.col("user_id").cast(pl.String))
    dev_df = dev_df.with_columns(pl.col("user_id").cast(pl.String))
    test_df = test_df.with_columns(pl.col("user_id").cast(pl.String))

    # filter useful columns
    income_ses_df = income_ses_df.select(['user_id', 'ind_snv_mean'])

    # merge
    train_tot_df = income_ses_df.join(train_df, on='user_id', how='inner')
    dev_tot_df = income_ses_df.join(dev_df, on='user_id', how='inner')
    test_tot_df = income_ses_df.join(test_df, on='user_id', how='inner')

    # save
    train_tot_df.write_csv('dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION.csv')
    dev_tot_df.write_csv('dev_phase_normal/input with medium large/test_dev_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION.csv')
    test_tot_df.write_csv('dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION.csv')'''


    # RE-MAP MEDIUM CLASS INTO LOWER/HIGH

    '''# set thresholds
    # for 3 classes
    low_thresh = 23171
    high_thresh = 32918.5
    # for 2 non-separated classes
    thresh = 27590

    # input paths
    input_train = 'dev_phase_normal/intermediate df/tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_income.csv'
    input_test = 'dev_phase_normal/intermediate df/tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_income.csv'

    # output paths
    output_train_class_2 = 'dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium_2_classes_chosen.csv'
    output_test_class_2 = 'dev_phase_normal/input with medium large/test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes_chosen.csv'

    # read files
    input_train_df = pl.read_csv(input_train)
    input_test_df = pl.read_csv(input_test)
    # rename columns
    input_train_df = input_train_df.rename({'SES_users': 'SES_users_old'})
    input_test_df = input_test_df.rename({'SES_users': 'SES_users_old'})

    print(input_train_df.schema['ind_snv_mean'])

    # move medium class into lower/high according to income
    # train
    input_train_df = input_train_df.with_columns(
        pl.when(pl.col('ind_snv_mean') < thresh)
        .then(pl.lit('lower'))
        .otherwise(pl.lit('high'))
        .alias('SES_users')
    )
    # test
    input_test_df = input_test_df.with_columns(
        pl.when(pl.col('ind_snv_mean') < thresh)
        .then(pl.lit('lower'))
        .otherwise(pl.lit('high'))
        .alias('SES_users')
    )

    print(input_train_df.filter(pl.col('SES_users') == 'lower'))
    print(input_train_df.filter(pl.col('SES_users') == 'high'))
    #print(input_test_df.filter(pl.col('SES_users') == 'lower'))
    #print(input_test_df.filter(pl.col('SES_users') == 'high'))'''


    # DIVIDE TEST SET INTO TEST AND DEV ACCORDING TO PREVIOUS DIVISION FOR RANDOM TWEETS PER USER
    '''df = pl.read_csv('dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv')
    print(df)
    print(df.filter(pl.col('SES_users') == 'lower'))
    print(df.filter(pl.col('SES_users') == 'high'))
    df1 = pl.read_csv('dev_phase_normal/input with medium large/test_dev_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv')
    print(df1)
    print(df1.filter(pl.col('SES_users') == 'lower'))
    print(df1.filter(pl.col('SES_users') == 'high'))'''
    '''df2 = pl.read_csv('dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv')
    print(df2)
    print(df2.filter(pl.col('SES_users') == 'lower'))
    print(df2.filter(pl.col('SES_users') == 'high'))'''

