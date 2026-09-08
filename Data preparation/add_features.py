import pandas as pd
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import re


# IN PANDAS

def tf_idf(text_list, vectorizer):

    tf_idf_vec = vectorizer.fit_transform(text_list)  # tweets = list of strings
    feature_names = vectorizer.get_feature_names_out()

    return tf_idf_vec, feature_names

def split_words(sentence):

    # lower case
    sentence = sentence.lower()
    # split into words using regex
    words = re.findall(r'\b\w+\b', sentence)

    return words

def count_words(sentence):

    words = re.findall(r'\b\w+\b', sentence)
    counted_words = len(words)

    return counted_words

def detect_negation(text):

    text = str(text).lower()

    if re.search(STANDARD_NEG_REGEX, text):
        return "standard"

    elif re.search(NEG_ONLY_REGEX, text):
        return "non_standard"

    else:
        return "no_negation"

def add_features_pandas(df, user_col):

    # 1) count number of tweet words
    df['TWEET_NUM_WORDS'] = df['text'].apply(count_words)

    # 2) group by user -> create set of words per each user and count number of tweets per user
    grouped_df = df.groupby(user_col).agg(
        vocabulary=('tweet_words', lambda x: set(word for tweet in x for word in tweet)),
        num_tweets=('tweet_id', 'count')
    ).reset_index()

    df['VOCAB_SIZE'] = df_grouped['vocabulary'].apply(len) / df_grouped['num_tweets']

    # 3) negation
    df['negation_type'] = df['text'].map(lambda x: detect_negation(x))
    df["has_standard_neg"] = (df["negation_type"] == "standard").astype(int)
    df["has_nonstandard_neg"] = (df["negation_type"] == "non_standard").astype(int)    

    return df


# IN POLARS

import polars as pl

# precompiled regex
'''STANDARD_NEG_REGEX = r"..."
NEG_ONLY_REGEX = r"..."'''

# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text):

    text = str(text).lower()

    # normalize apostrophes
    text = text.replace("’", "'")

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove mentions
    text = re.sub(r"@\w+", "", text)

    # tokenize
    tokens = re.findall(r"\b\w+[\w'-]*\b", text)

    return tokens


# =========================================================
# PLURAL APPEARANCE DETECTION
# =========================================================

def looks_plural(word):

    return (
        word.endswith("s")
        or word.endswith("x")
        or word.endswith("aux")
    )


# =========================================================
# STANDARD PLURAL GENERATOR
# =========================================================

def make_plural(word):

    word = word.lower()

    # already invariant
    if word.endswith(INVARIABLE_ENDINGS):
        return word

    # irregular -ail
    if word in AIL_AUX_WORDS:
        return AIL_AUX_WORDS[word]

    # -al -> -aux
    if word.endswith("al"):
        return word[:-2] + "aux"

    # -eau / -eu -> x
    if word.endswith(("eau", "eu")):
        return word + "x"

    # special -ou nouns
    if word in OU_X_WORDS:
        return word + "x"

    # default
    return word + "s"


# =========================================================
# MAIN DETECTOR
# =========================================================

def detect_plural_errors(text):

    tokens = tokenize(text)

    errors = []

    for i in range(len(tokens) - 1):

        article = tokens[i]
        noun = tokens[i + 1]

        # skip if not plural determiner
        if article not in PLURAL_ARTICLES:
            continue

        # skip tiny tokens
        if len(noun) <= 2:
            continue

        # skip hashtags
        if noun.startswith("#"):
            continue

        # already plural-looking -> OK
        if looks_plural(noun):
            continue

        # otherwise generate expected plural
        expected_plural = make_plural(noun)

        errors.append({
            "article": article,
            "word": noun,
            "expected_plural": expected_plural,
            "error_type": "missing_plural"
        })

    return errors

def plural_category(text):

    errors = detect_plural_errors(text)

    # no plural article at all
    tokens = tokenize(text)

    has_plural_article = any(
        tok in PLURAL_ARTICLES
        for tok in tokens
    )

    if not has_plural_article:
        return "no_plural"

    # plural article + plural error
    if len(errors) > 0:
        return "non_standard_plural"

    # plural article + no error
    return "standard_plural"



# =========================================================
# BINARY FEATURE
# =========================================================

def has_plural_error(text):

    return int(
        len(detect_plural_errors(text)) > 0
    )


# =========================================================
# CATEGORICAL FEATURE
# =========================================================

def plural_status(text):

    errors = detect_plural_errors(text)

    if len(errors) == 0:
        return "standard_plural"

    return "non_standard_plural"


def add_features_polars(df, user_col: str):#: pl.LazyFrame | pl.DataFrame

    # make lazy if not already
    if isinstance(df, pl.DataFrame):
        df = df.lazy()
    
    # tokenize words
    df = df.with_columns(
        pl.col("text")
        .str.to_lowercase()
        .str.extract_all(r"\b\w+\b")
        .cast(pl.List(pl.Utf8))
        .alias("tweet_words")
    )

    # 1) number of words

    df = df.with_columns(
        pl.col("tweet_words")
        .list.len()
        .alias("TWEET_NUM_WORDS")
    )

    print('Done number of words')

    # 2) vocabulary size per user

    grouped_df = (
        df.group_by(user_col)
        .agg([
            # safer than flatten + unique combo issues
            pl.col("tweet_words")
            .explode()
            .n_unique()
            .alias("vocabulary_size"),

            pl.len().alias("num_tweets")
        ])
        .with_columns(
            (pl.col("vocabulary_size") / pl.col("num_tweets"))
            .alias("VOCAB_SIZE")
        )
        .select([user_col, "VOCAB_SIZE"])
    )

    print('Done vocabulary size')

    # join back
    df = df.join(grouped_df, on=user_col, how="left")
    
    print('vocabulary', df)


    # 3) negation detection

    df = df.with_columns(
        pl.when(
            pl.col("text")
            .str.to_lowercase()
            .str.contains(STANDARD_NEG_REGEX)
        )
        .then(pl.lit("standard"))

        .when(
            pl.col("text")
            .str.to_lowercase()
            .str.contains(NEG_ONLY_REGEX)
        )
        .then(pl.lit("non_standard"))

        .otherwise(pl.lit("no_negation"))
        .alias("negation_type")
    )

    # binary flag
    df = df.with_columns(
        (pl.col("negation_type") == "standard")
        .cast(pl.Int8)
        .alias("has_standard_neg")
    )
    df = df.with_columns(
        (pl.col("negation_type") == "non_standard")
        .cast(pl.Int8)
        .alias("has_nonstandard_neg")
    )
    
    print('Done negation')


    # 4) pluralization detection

    # =========================================================
    # CONFIG
    # =========================================================

    # plural determiners/articles
    PLURAL_ARTICLES = {
        "les", "des", "ces",
        "mes", "tes", "ses",
        "nos", "vos", "leurs",
        "quelques", "plusieurs"
    }

    # nouns ending in -ou taking x
    OU_X_WORDS = {
        "bijou", "caillou", "chou",
        "genou", "hibou", "joujou", "pou"
    }

    # irregular -ail
    AIL_AUX_WORDS = {
        "travail": "travaux",
        "vitrail": "vitraux",
        "corail": "coraux",
        "émail": "émaux",
        "soupirail": "soupiraux",
        "bail": "baux"
    }

    # invariant endings
    INVARIABLE_ENDINGS = ("s", "x", "z")

    df = df.with_columns(

    pl.col("text")
    .map_elements(
        plural_category,
        return_dtype=pl.Utf8
    )
    .alias("plural_type")
    )

    df = df.with_columns([

        (
            pl.col("plural_type")
            == "standard_plural"
        )
        .cast(pl.Int8)
        .alias("standard_plural"),

        (
            pl.col("plural_type")
            == "non_standard_plural"
        )
        .cast(pl.Int8)
        .alias("non_standard_plural"),

        (
            pl.col("plural_type")
            == "no_plural"
        )
        .cast(pl.Int8)
        .alias("no_plural")
    ])
    
    print('Done pluralization')

    # optional:
    # remove intermediate column
    df = df.drop(["tweet_words", "negation_type", "plural_type", "no_plural"])

    return df








'''# =========================================================
# EXAMPLES
# =========================================================

examples = [

    # incorrect
    "les chat jouent",
    "des cheval noirs",
    "les château sont beaux",
    "des travail difficiles",

    # correct
    "les chats jouent",
    "des chevaux noirs",
    "les châteaux sont beaux",
    "des travaux difficiles",

    # noisy twitter
    "les gars sont la",
    "des mec bizarre",
    "les bg arrivent",
    "des chevauxuuu"
]

for sent in examples:

    print("\\n====================")
    print(sent)

    print("status:",
          plural_status(sent))

    print("binary:",
          has_plural_error(sent))

    print("errors:",
          detect_plural_errors(sent))

    return df'''


























if __name__ == "__main__":

    # preprocessing (if needed)
    # 1) save into csv
    df = pl.read_parquet('tweets_langtool_to_add_features/new_users_tweets_test.parquet')
    df.write_csv('tweets_langtool_to_add_features/new_users_tweets_test.csv')
    

    STANDARD_NEG_REGEX = (
        r"\b(ne|n'|n’)\b.*\b(pas|jamais|rien|personne|aucun|aucune)\b"
    )

    NEG_ONLY_REGEX = (
        r"\b(pas|pa|aps|jamais|ni|personne|rien|ri1|r1|aucun|aucune)\b"
    )

    INPUT_PATH = 'tweets_langtool_to_add_features/new_users_tweets_test.csv' #'tweets_langtool_to_add_features/TWEETS_30_chosen_NEW_3_classes_LANGTOOLS_train.csv' #'tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets.csv'   #'chunks_paris_merged_enriched/chunks_paris_merged_enriched_until_192.parquet'

    OUTPUT_PATH_CSV = 'dev_phase_normal/input with medium large/intermediate test/tweets_users_new_test.csv' #tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets
    OUTPUT_PATH_PARQUET = 'dev_phase_normal/input with medium large/intermediate test/tweets_users_new_test.parquet' # tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets

    # plural determiners/articles
    PLURAL_ARTICLES = {
        "les", "des", "ces",
        "mes", "tes", "ses",
        "nos", "vos", "leurs",
        "quelques", "plusieurs"
    }

    # nouns ending in -ou taking x
    OU_X_WORDS = {
        "bijou", "caillou", "chou",
        "genou", "hibou", "joujou", "pou"
    }

    # irregular -ail
    AIL_AUX_WORDS = {
        "travail": "travaux",
        "vitrail": "vitraux",
        "corail": "coraux",
        "émail": "émaux",
        "soupirail": "soupiraux",
        "bail": "baux"
    }

    # invariant endings
    INVARIABLE_ENDINGS = ("s", "x", "z")

    df = pl.read_csv(INPUT_PATH)
    print(df)

    # replace null with zero
    df = df.fill_null(0)

    #df = df.rename({'clean_light': 'text'})

    df = df.sort('user_id')

    featured_df = add_features_polars(df, 'user_id')

    featured_df = featured_df.collect()

    print(featured_df)

    featured_df.write_csv(OUTPUT_PATH_CSV)
    featured_df.write_parquet(OUTPUT_PATH_PARQUET)




















    # check if we have some tweets with errors features
    '''df_test = pl.read_csv('dev_phase_normal/input with medium/TEST_DEV_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_negation.csv')
    df_dev = pl.read_csv('dev_phase_normal/input with medium/TEST_TEST_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_negation.csv')
    df_train = pl.read_csv('dev_phase_normal/input with medium/TRAIN_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_negation.csv')
    
    featured_df = pl.read_csv(OUTPUT_PATH_CSV)
    featured_df = featured_df.rename({'id': 'tweet_id'})

    # keep only useful columns
    df_train = df_train[['tweet_id', 'AGREEMENT','CASING','CAT_ELISION','CAT_GRAMMAIRE','CAT_HOMONYMES_PARONYMES','CAT_MAJUSCULES','CAT_PLEONASMES','CAT_REGIONALISMES','CAT_REGLES_DE_BASE','CAT_TOURS_CRITIQUES','CAT_TYPOGRAPHIE','MISC','MULTITOKEN_SPELLING','PONCTUATION_POINT','PONCTUATION_VIRGULE','PUNCTUATION','REPETITIONS_STYLE','SEMANTICS','STYLE','TYPOGRAPHY','TYPOS']]
    df_dev = df_dev[['tweet_id', 'AGREEMENT','CASING','CAT_ELISION','CAT_GRAMMAIRE','CAT_HOMONYMES_PARONYMES','CAT_MAJUSCULES','CAT_PLEONASMES','CAT_REGIONALISMES','CAT_REGLES_DE_BASE','CAT_TOURS_CRITIQUES','CAT_TYPOGRAPHIE','MISC','MULTITOKEN_SPELLING','PONCTUATION_POINT','PONCTUATION_VIRGULE','PUNCTUATION','REPETITIONS_STYLE','SEMANTICS','STYLE','TYPOGRAPHY','TYPOS']]
    df_test = df_test[['tweet_id', 'AGREEMENT','CASING','CAT_ELISION','CAT_GRAMMAIRE','CAT_HOMONYMES_PARONYMES','CAT_MAJUSCULES','CAT_PLEONASMES','CAT_REGIONALISMES','CAT_REGLES_DE_BASE','CAT_TOURS_CRITIQUES','CAT_TYPOGRAPHIE','MISC','MULTITOKEN_SPELLING','PONCTUATION_POINT','PONCTUATION_VIRGULE','PUNCTUATION','REPETITIONS_STYLE','SEMANTICS','STYLE','TYPOGRAPHY','TYPOS']]

    # merge
    df_tot = pl.concat([df_train, df_dev, df_test])
    merged_df = featured_df.join(df_tot, on='tweet_id', how='inner')

    print(merged_df)

    print('featured df')
    print(featured_df)
    print('df tot')
    print(df_tot)'''

    '''input_df = pl.scan_csv(INPUT_PATH)

    

    output_df = add_features_polars(input_df, 'user_id')
    output_df_csv = output_df.drop("tweet_words")

    # save output df
    # csv
    output_df_csv.collect().write_csv(OUTPUT_PATH_CSV)
    # parquet
    output_df.collect().write_parquet(OUTPUT_PATH_PARQUET)'''

    '''grouped_df = df.group_by('user_id').agg(
        pl.col('SES_users').first(),
        pl.len().alias('num_tweets')
    ).filter(pl.col('num_tweets') > 29).sort('num_tweets')

    grouped_grouped_df = grouped_df.group_by('SES_users').agg(
        pl.len().alias('num_users_per_class')
    )

    grouped_df = grouped_df.collect()
    grouped_grouped_df = grouped_grouped_df.collect()

    print(grouped_df)
    print(grouped_grouped_df)'''

    '''df = pl.read_parquet(OUTPUT_PATH_CSV)
    print(df)
    print(df.columns)
    print(df['non_standard_plural'].unique())
    print(df['standard_plural'].unique())'''

    '''print(df['no_plural'].unique())
    print(df['has_nonstandard_neg'].unique())'''

    '''df_sel = df.select(['text', 'no_plural', 'non_standard_plural', 'standard_plural'])
    print(df_sel.filter(pl.col('non_standard_plural') == 1))
    print(df_sel.filter(pl.col('standard_plural') == 1))
    print(df_sel.filter(pl.col('no_plural') == 1))'''

    '''
    
    print('ex 1')
    ex = plural_category('les jeux sont fait')
    print(ex)
    print('\n')

    print('ex 2')
    ex = plural_category('les chat son fait')
    print(ex)
    print('\n')

    print('ex 3')
    ex = plural_category('le chat est belle')
    print(ex)
    print('\n')'''





    '''df_train = pd.read_csv('dev_phase_normal/input with medium/TRAIN_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv')
    df_dev = pd.read_csv('dev_phase_normal/input with medium/TEST_DEV_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv')
    df_test = pd.read_csv('dev_phase_normal/input with medium/TEST_DEV_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv')

    df_train["negation_type"] = df_train['text'].map(lambda x: detect_negation(x))
    df_dev["negation_type"] = df_dev['text'].map(lambda x: detect_negation(x))
    df_test["negation_type"] = df_test['text'].map(lambda x: detect_negation(x))

    print(df_train)

    # train df
    df_train["has_standard_neg"] = (
        df_train["negation_type"] == "standard"
    ).astype(int)
    df_train["has_nonstandard_neg"] = (
        df_train["negation_type"] == "non_standard"
    ).astype(int)

    # dev df
    df_dev["has_standard_neg"] = (
        df_dev["negation_type"] == "standard"
    ).astype(int)
    df_dev["has_nonstandard_neg"] = (
        df_dev["negation_type"] == "non_standard"
    ).astype(int)

    # test df
    df_test["has_standard_neg"] = (
        df_test["negation_type"] == "standard"
    ).astype(int)
    df_test["has_nonstandard_neg"] = (
        df_test["negation_type"] == "non_standard"
    ).astype(int)

    print(df_train)
    df_train.to_csv('dev_phase_normal/input with medium/TRAIN_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_negation.csv', index=False)
    df_dev.to_csv('dev_phase_normal/input with medium/TEST_DEV_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_negation.csv', index=False)
    df_test.to_csv('dev_phase_normal/input with medium/TEST_TEST_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_negation.csv', index=False)

'''