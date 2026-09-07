from text_process import *
import language_tool_python
from pathlib import Path
import pycld2
import cld3
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import pyarrow.parquet as pq


'''This script is used to run LanguageTool on the tweets and create a parquet file with the number of mistakes for each category, rule and SES level.
The input is a parquet file with the tweets and the output is a parquet file with the mistakes counted.'''


def lang_tool_chunk(input_path, output_dir):

    batchsize = 10000  # tune this

    # use lang_tool
    tool = language_tool_python.LanguageTool('fr')

    # file
    parquet_file = pq.ParquetFile(input_path)#.slice(0, 10)

    # iterator
    i = 0

    for batch in parquet_file.iter_batches(batch_size=batchsize):

        print(i)

        chunk = batch.to_pandas()

        user_mistakes = count_mistakes_single_tweets(chunk, tool)
        # save mistakes df
        user_mistakes.to_parquet(output_dir + 'chunk_' + str(i) + '.parquet')
        # count number of mistakes per ses level, rule, category
        #train_counts = user_mistakes_train.groupby(["category", "rule_id", 'SES_users']).sum()
        # save counted mistakes
        #train_counts.to_parquet('../classification_data/FR/TRAIN_users_mistakes_PARIS_count_no_spammers_OK.parquet')

        i += 1


if __name__ == "__main__":

    TRAIN_INPUT_PATH =  'to_lang_tool.parquet'#'../errors df/train_for_langtool_30_tweets_chosen_correct.parquet'
    TEST_INPUT_PATH = '../errors df/test_for_langtool_30_tweets_chosen_correct.parquet'

    OUTPUT_PATH_TRAIN = '../classification_data/FR/new_users_tweets_langtool_done/'#'../classification_data/FR/TWEETS_TRAIN_30_chosen_correct/'
    OUTPUT_PATH_TEST = '../classification_data/FR/TWEETS_TEST_30_chosen_correct/' #tweets_NEW_TEST_OK

    df_train = pl.read_parquet(TRAIN_INPUT_PATH)
    df_test = pl.read_parquet(TEST_INPUT_PATH)

    lang_tool_chunk(TRAIN_INPUT_PATH, OUTPUT_PATH_TRAIN)
    lang_tool_chunk(TEST_INPUT_PATH, OUTPUT_PATH_TEST)