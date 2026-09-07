import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score


def evaluation(class_num, col_pred, TEST_PATH, PRED_PATH):

    # set prediction and true labels vectors
    y_true = pd.read_csv(TEST_PATH)#, sep='\t'
    y_pred = pd.read_csv(PRED_PATH)

    print('pred path', PRED_PATH)
    print('test path', TEST_PATH)
    print(y_true)
    print(y_pred)
    
    # merge true and pred to have labels on each tweet
    #y_tot = pd.merge(y_true, y_pred, on=col_pred)
    
    # low df
    y_true_low = y_true[y_true['SES_users'] == 'lower']
    y_true_medium = y_true[y_true['SES_users'] == 'medium']
    y_true_high = y_true[y_true['SES_users'] == 'high']

    print(y_true)
    # set columns predictions df
    #y_pred.columns = COLUMNS_TOT_ID_TEXT

    if col_pred == 'user_id':
        print('class_num', class_num)
        # keep only one example per user since we check the user's label
        y_true = y_true.drop_duplicates(col_pred, keep='first')
        # change labels
        if class_num == '2' or class_num == '2_2':
            label_map = {"lower": 0, "high": 1}
        elif class_num == '3':
            label_map = {"lower": 0, "medium": 1, "high": 2}
        y_true["SES_users"] = y_true["SES_users"].map(label_map)
        y_true_low["SES_users"] = y_true_low["SES_users"].map(label_map)
        y_true_medium["SES_users"] = y_true_medium["SES_users"].map(label_map)
        y_true_high["SES_users"] = y_true_high["SES_users"].map(label_map)
    else:
        print('class_num', class_num)
        # keep only one example per user since we check the user's label
        #y_true = y_true.drop_duplicates('user_id', keep='first')
        # change labels
        if class_num == '2' or class_num == '2_2':
            label_map = {"lower": 0, "high": 1}
        elif class_num == '3':
            label_map = {"lower": 0, "medium": 1, "high": 2}
        y_true["SES_users"] = y_true["SES_users"].map(label_map)
        y_true_low["SES_users"] = y_true_low["SES_users"].map(label_map)
        y_true_medium["SES_users"] = y_true_medium["SES_users"].map(label_map)
        y_true_high["SES_users"] = y_true_high["SES_users"].map(label_map)

    print('true', y_true)
    print('pred', y_pred)

    y_pred = y_pred.rename(columns={'label': 'SES_users'})
    print(y_pred['SES_users'].unique)

    # total df
    y_tot = pd.merge(y_true, y_pred, on=col_pred, suffixes=('_true', '_pred'))
    y_tot = y_tot[['SES_users_true', 'SES_users_pred', col_pred, 'tweet_id']]
    y_low_tot = pd.merge(y_true_low, y_pred, on=col_pred, suffixes=('_true', '_pred'))
    y_low_tot = y_low_tot[['SES_users_true', 'SES_users_pred', col_pred, 'tweet_id']]
    y_medium_tot = pd.merge(y_true_medium, y_pred, on=col_pred, suffixes=('_true', '_pred'))
    y_medium_tot = y_medium_tot[['SES_users_true', 'SES_users_pred', col_pred, 'tweet_id']]
    y_high_tot = pd.merge(y_true_high, y_pred, on=col_pred, suffixes=('_true', '_pred'))
    y_high_tot = y_high_tot[['SES_users_true', 'SES_users_pred', col_pred, 'tweet_id']]

    print(y_tot)

    # sort by tweet_id
    '''y_true = y_true.sort_values(by='tweet_id')
    y_pred = y_pred.sort_values(by='tweet_id')'''

    # compute F1 score and accuracy
    if class_num == '2' or class_num == '2_2':
        f1_score_num = f1_score(y_tot['SES_users_true'], y_tot['SES_users_pred'], average='macro')
        acc = accuracy_score(y_tot['SES_users_true'], y_tot['SES_users_pred'])
        acc_low = accuracy_score(y_low_tot['SES_users_true'], y_low_tot['SES_users_pred'])
        acc_medium = accuracy_score(y_medium_tot['SES_users_true'], y_medium_tot['SES_users_pred'])
        acc_high = accuracy_score(y_high_tot['SES_users_true'], y_high_tot['SES_users_pred'])
    elif class_num == '3':
        f1_score_num = f1_score(y_tot['SES_users_true'], y_tot['SES_users_pred'], average='macro')
        acc = accuracy_score(y_tot['SES_users_true'], y_tot['SES_users_pred'])
        acc_low = accuracy_score(y_low_tot['SES_users_true'], y_low_tot['SES_users_pred'])
        acc_medium = accuracy_score(y_medium_tot['SES_users_true'], y_medium_tot['SES_users_pred'])
        acc_high = accuracy_score(y_high_tot['SES_users_true'], y_high_tot['SES_users_pred'])

    return f1_score_num, acc, acc_low, acc_medium, acc_high, PRED_PATH


def evaluation_direct(col_pred, col_true, PRED_PATH):

    # read df with labels
    df = pd.read_csv(PRED_PATH)

    # create dub df
    lower_values = ['lower', 'low']
    medium_values = ['medium']
    high_values = ['high']
    df_low = df[df[col_true].isin(lower_values)]
    df_medium = df[df[col_true].isin(medium_values)]
    df_high = df[df[col_true].isin(high_values)]

    # compute F1 score and accuracy
    f1_score_num = f1_score(df[col_true], df[col_pred], average='macro')
    acc = accuracy_score(df[col_true], df[col_pred])
    acc_low = accuracy_score(df_low[col_true], df_low[col_pred])
    acc_medium = accuracy_score(df_medium[col_true], df_medium[col_pred])
    acc_high = accuracy_score(df_high[col_true], df_high[col_pred])

    return f1_score_num, acc, acc_low, acc_medium, acc_high, PRED_PATH


if __name__ == "__main__":


    #pred_regression = pd.read_csv('predictions_regression_ILE_DE_FRANCE/all_features_negation_attention/run_1/all_features_negation_attention.csv')
    '''pred_regression = pred_regression[pred_regression['true_class'].isin(['lower', 'high'])]
    pred_regression['new_pred_class'] = ''
    pred_regression['new_pred_class'] = np.where(pred_regression['pred_income'] < 28044, 'low', 'high')
    print(pred_regression)'''
    #f1_score_num = f1_score(pred_regression['pred_class'], pred_regression['true_class'], average='macro')
    #print(f1_score_num)

    CLASS_NUM_LIST = ['2_2']#'2', , '3'
    NUM_TWEETS_LIST = [5, 15, 30] #[1, 5, 15, 25, 30] 5, 15, 

    CONFIG_TYPE = 'pre_avg_v2_ILE_DE_FRANCE' # llama | pre_avg_v2_ILE_DE_FRANCE | pre_avg_v2_3_classes_ILE_DE_FRANCE | post_ILE_DE_FRANCE | regression_ILE_DE_FRANCE | gpt

    prediction_type = 'classification' # classification | regression | llm

    masked_location = False # True | False

    #CLASS_NUM = 3
    COL_PRED = '' # user_id | tweet_id
    # NB: file for 2 classes separated and for 3 classes: dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium.csv | file for 2 classes not separated: dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv
    #TEST_PATH = 'dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium.csv' # _2_classes
    '''PRED_DIR = 'predictions_gpt/classes_3_5/' #'predictions_pre_avg_v2_ILE_DE_FRANCE/classes_2/' 'predictions_pre_avg_v2_3_classes_ILE_DE_FRANCE/classes_3/' 'predictions_post_ILE_DE_FRANCE/classes_2/' #'predictions_post_ILE_DE_FRANCE/classes_3/' #'predictions_llama/' predictions_regression_ILE_DE_FRANCE 
    F1_SCORE_DIR = 'F1_score_gpt/classes_3_5/' # 'F1_score_dev_normal_pre_avg_v2_ILE_DE_FRANCE/classes_2/' 'F1_score_dev_normal_pre_avg_v2_3_classes_ILE_DE_FRANCE/classes_3/' #'F1_score_dev_normal_post_ILE_DE_FRANCE/classes_2/' #F1_score_dev_normal_post_ILE_DE_FRANCE/classes_3/' #'F1_score_llama/classes_2/' F1_score_regression_ILE_DE_FRANCE
    LLAMA_SCORE_PATH = 'F1_score_gpt/tot_scores_classes_3_5.csv'''

    

    for CLASS_NUM in CLASS_NUM_LIST:

        for num_tweets in NUM_TWEETS_LIST:

            print(f"Processing CLASS_NUM: {CLASS_NUM}, num_tweets: {num_tweets}")

            try:
                
                if CONFIG_TYPE == 'gpt' or CONFIG_TYPE == 'llama':
                    PRED_DIR = f'predictions_{CONFIG_TYPE}/classes_{CLASS_NUM}/'
                    F1_SCORE_DIR = f'F1_score_{CONFIG_TYPE}/classes_{CLASS_NUM}/'
                    LLM_SCORE_PATH = f'F1_score_{CONFIG_TYPE}/tot_scores_classes_{CLASS_NUM}.csv'
                elif CONFIG_TYPE == 'post_ILE_DE_FRANCE':
                    if masked_location == True:
                        PRED_DIR = f'predictions_{CONFIG_TYPE}_MASKED_LOC/classes_{CLASS_NUM}/'
                        F1_SCORE_DIR = f'F1_score_dev_normal_{CONFIG_TYPE}_MASKED_LOC/classes_{CLASS_NUM}/'
                        LLM_SCORE_PATH = f'F1_score_dev_normal_{CONFIG_TYPE}_MASKED_LOC/tot_scores_classes_{CLASS_NUM}.csv'
                    else:
                        PRED_DIR = f'predictions_{CONFIG_TYPE}/classes_{CLASS_NUM}/'
                        F1_SCORE_DIR = f'F1_score_dev_normal_{CONFIG_TYPE}/classes_{CLASS_NUM}/'
                        LLM_SCORE_PATH = f'F1_score_dev_normal_{CONFIG_TYPE}/tot_scores_classes_{CLASS_NUM}.csv'
                else:
                    if masked_location == True:
                        PRED_DIR = f'predictions_{CONFIG_TYPE}_MASKED_LOC/classes_{CLASS_NUM}_{num_tweets}/'
                        F1_SCORE_DIR = f'F1_score_dev_normal_{CONFIG_TYPE}_MASKED_LOC/classes_{CLASS_NUM}_{num_tweets}/'
                        LLM_SCORE_PATH = f'F1_score_dev_normal_{CONFIG_TYPE}_MASKED_LOC/tot_scores_classes_{CLASS_NUM}_{num_tweets}.csv'
                    else:    
                        PRED_DIR = f'predictions_{CONFIG_TYPE}/classes_{CLASS_NUM}_{num_tweets}/'
                        F1_SCORE_DIR = f'F1_score_dev_normal_{CONFIG_TYPE}/classes_{CLASS_NUM}_{num_tweets}/'
                        LLM_SCORE_PATH = f'F1_score_dev_normal_{CONFIG_TYPE}/tot_scores_classes_{CLASS_NUM}_{num_tweets}.csv'

                if CLASS_NUM == '2' or CLASS_NUM == '3':
                    if masked_location == True:
                        TEST_PATH = f'dev_phase_normal/input with medium large/tweets_ner_stanza_test_test_location_3_classes.csv'
                    else:
                        TEST_PATH = f'dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium.csv'
                else:
                    if masked_location == True:
                        TEST_PATH = f'dev_phase_normal/input with medium large/tweets_ner_stanza_test_test_location_2_classes.csv'
                    else:
                        TEST_PATH = f'dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv'
        
                if prediction_type == 'classification':

                    pred_type_directories = [d for d in os.listdir(PRED_DIR) if os.path.isdir(os.path.join(PRED_DIR, d))]
                    print(pred_type_directories)

                    for d in pred_type_directories:

                        df_eval = pd.DataFrame()
                        #if d in ['corr_features_with_length_negation', 'corr_features_with_length_vocab_negation', 'corr_features_with_length_vocab_pluralization']:
                        #    continue
                        print(d)
                        print(PRED_DIR + d)
                        r = os.listdir(PRED_DIR + d)
                        print('r', r)
                        runs_dirs = [run for run in os.listdir(PRED_DIR + d) if os.path.isdir(os.path.join(PRED_DIR + d, run))]
                        i = 0
                        for run in runs_dirs:
                            print('i', i)
                            run_files = [f for f in os.listdir(PRED_DIR + d + '/' + run) if os.path.isfile(os.path.join(PRED_DIR + d + '/' + run, f))]
                            for run_file in run_files:
                                print('run file', run_file)
                                if run_file[-5] == 's': # if tweets files -> final letters are ...tweets.csv
                                    COL_PRED = 'tweet_id'
                                else:
                                    COL_PRED = 'user_id'
                                # compute total accuracy and by class
                                PRED_PATH = PRED_DIR + d + '/' + run + '/' + run_file
                                print('EVAL')
                                f1_score_num, acc, acc_low, acc_medium, acc_high, pred_path = evaluation(CLASS_NUM, COL_PRED, TEST_PATH, PRED_PATH)
                                df_eval = pd.concat([df_eval, pd.DataFrame({'run_num': [i+1], 'F1_score': [f1_score_num], 'accuracy': [acc], 'accuracy_low': [acc_low], 'accuracy_medium': [acc_medium], 'accuracy_high': [acc_high], 'Prediction_path': [pred_path]})])
                            i += 1
                        df_eval.to_csv(F1_SCORE_DIR + d + '.csv')
                
                elif prediction_type == 'regression':

                    pred_type_directories = [d for d in os.listdir(PRED_DIR) if os.path.isdir(os.path.join(PRED_DIR, d))]

                    COL_PRED = 'pred_class'
                    COL_TRUE = 'true_class'

                    for d in pred_type_directories:
                        df_eval = pd.DataFrame()
                        runs_dirs = [run for run in os.listdir(PRED_DIR + d) if os.path.isdir(os.path.join(PRED_DIR + d, run))]
                        i = 0
                        for run in runs_dirs:
                            run_files = [f for f in os.listdir(PRED_DIR + d + '/' + run) if os.path.isfile(os.path.join(PRED_DIR + d + '/' + run, f))]
                            for run_file in run_files:
                                # compute total accuracy and by class
                                PRED_PATH = PRED_DIR + d + '/' + run + '/' + run_file
                                f1_score_num, acc, acc_low, acc_medium, acc_high, pred_path = evaluation_direct(COL_PRED, COL_TRUE, PRED_PATH)
                                df_eval = pd.concat([df_eval, pd.DataFrame({'run_num': [i+1], 'F1_score': [f1_score_num], 'accuracy': [acc], 'accuracy_low': [acc_low], 'accuracy_medium': [acc_medium], 'accuracy_high': [acc_high], 'Prediction_path': [pred_path]})])
                        df_eval.to_csv(F1_SCORE_DIR + d + '.csv')

                elif prediction_type == 'llm':
                    
                    files = [f for f in os.listdir(PRED_DIR) if os.path.isfile(os.path.join(PRED_DIR, f))]

                    for file in files:

                        COL_PRED = 'pred'
                        COL_TRUE = 'true'

                        f1_score_num, acc, acc_low, acc_medium, acc_high, pred_path = evaluation_direct(COL_PRED, COL_TRUE, PRED_DIR + file)
                        df_eval = pd.DataFrame({'F1_score': [f1_score_num], 'accuracy': [acc], 'accuracy_low': [acc_low], 'accuracy_medium': [acc_medium], 'accuracy_high': [acc_high], 'Prediction_path': [pred_path]})

                        df_eval.to_csv(F1_SCORE_DIR + file + '.csv')


                        # concatenate llama results

                        scores_files = files = [f for f in os.listdir(F1_SCORE_DIR) if os.path.isfile(os.path.join(F1_SCORE_DIR, f))]

                        scores_tot_df = pd.DataFrame()

                        for score_file in scores_files:
                            df_scores = pd.read_csv(F1_SCORE_DIR + score_file)
                            scores_tot_df = pd.concat([scores_tot_df, df_scores])
                        
                        scores_tot_df = scores_tot_df.sort_values(by='F1_score')
                        scores_tot_df.to_csv(LLM_SCORE_PATH)

            except Exception as e:
                print(f"An error occurred for class_num {CLASS_NUM} and num_tweets {num_tweets}: {e}")
                continue
    
    mean_tot_df = pd.DataFrame()

    for CLASS_NUM in CLASS_NUM_LIST:
        
        if CONFIG_TYPE == 'gpt' or CONFIG_TYPE == 'llama':
            OUTPUT_PATH_MEAN = f'F1_score_{CONFIG_TYPE}/tot_mean_scores_classes_{CLASS_NUM}.csv' #'F1_score_dev_normal_post_ILE_DE_FRANCE/tot_scores_classes_2/' #'F1_score_dev_normal_pre_avg_v2_3_classes_ILE_DE_FRANCE/tot_scores_classes_2/'
        else:
            if masked_location == True:
                OUTPUT_PATH_MEAN = f'F1_score_dev_normal_{CONFIG_TYPE}_MASKED_LOC/tot_mean_scores_classes_{CLASS_NUM}.csv' #'F1_score_dev_normal_post_ILE_DE_FRANCE/tot_scores_classes_2/' #'F1_score_dev_normal_pre_avg_v2_3_classes_ILE_DE_FRANCE/tot_scores_classes_2/'
            else:
                OUTPUT_PATH_MEAN = f'F1_score_dev_normal_{CONFIG_TYPE}/tot_mean_scores_classes_{CLASS_NUM}.csv' #'F1_score_dev_normal_post_ILE_DE_FRANCE/tot_scores_classes_2/' #'F1_score_dev_normal_pre_avg_v2_3_classes_ILE_DE_FRANCE/tot_scores_classes_2/'

        for num_tweets in NUM_TWEETS_LIST:

            try:

                if CONFIG_TYPE == 'gpt' or CONFIG_TYPE == 'llama':
                    #PRED_DIR = f'predictions_{CONFIG_TYPE}/classes_{CLASS_NUM}/'
                    F1_SCORE_DIR = f'F1_score_{CONFIG_TYPE}/classes_{CLASS_NUM}/'
                    LLM_SCORE_PATH = f'F1_score_{CONFIG_TYPE}/tot_scores_classes_{CLASS_NUM}.csv'
                elif CONFIG_TYPE == 'post_ILE_DE_FRANCE':
                    if masked_location == True:
                        F1_SCORE_DIR = f'F1_score_dev_normal_{CONFIG_TYPE}_MASKED_LOC/classes_{CLASS_NUM}/'
                    else:
                        F1_SCORE_DIR = f'F1_score_dev_normal_{CONFIG_TYPE}/classes_{CLASS_NUM}/'
                else:
                    #PRED_DIR = f'predictions_{CONFIG_TYPE}/classes_{CLASS_NUM}_{num_tweets}/'
                    if masked_location == True:
                        F1_SCORE_DIR = f'F1_score_dev_normal_{CONFIG_TYPE}_MASKED_LOC/classes_{CLASS_NUM}_{num_tweets}/'
                    else:
                        F1_SCORE_DIR = f'F1_score_dev_normal_{CONFIG_TYPE}/classes_{CLASS_NUM}_{num_tweets}/'

                INPUT_DIR = F1_SCORE_DIR # 'F1_regression_ILE_DE_FRANCE/classes_3/' #'F1_score_dev_normal_post_ILE_DE_FRANCE/classes_2/' #'F1_score_dev_normal_pre_avg_v2_3_classes_ILE_DE_FRANCE/classes_2/'

                if CONFIG_TYPE == 'gpt' or CONFIG_TYPE == 'llama':
                    OUTPUT_PATH = f'F1_score_{CONFIG_TYPE}/tot_scores_classes_{CLASS_NUM}.csv'
                elif CONFIG_TYPE == 'post_ILE_DE_FRANCE':
                    if masked_location == True:
                        OUTPUT_PATH = f'F1_score_dev_normal_{CONFIG_TYPE}_MASKED_LOC/tot_scores_classes_{CLASS_NUM}.csv'
                    else:
                        OUTPUT_PATH = f'F1_score_dev_normal_{CONFIG_TYPE}/tot_scores_classes_{CLASS_NUM}.csv'
                else:                
                    if masked_location == True:
                        OUTPUT_PATH = f'F1_score_dev_normal_{CONFIG_TYPE}_MASKED_LOC/tot_scores_classes_{CLASS_NUM}_{num_tweets}.csv'
                    else:
                        OUTPUT_PATH = f'F1_score_dev_normal_{CONFIG_TYPE}/tot_scores_classes_{CLASS_NUM}_{num_tweets}.csv' #'F1_score_dev_normal_post_ILE_DE_FRANCE/tot_scores_classes_2.csv' #'F1_score_dev_normal_pre_avg_v2_3_classes_ILE_DE_FRANCE/tot_scores_classes_2.csv'
                
                files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]

                list_df = []
                output_df = pd.DataFrame()

                for file in files:

                    #print(file)

                    df = pd.read_csv(INPUT_DIR + file)

                    #print(df)

                    #print(df.columns)

                    # Prediction_path -> if classification | pred_path -> if regression
                    #if (INPUT_DIR == 'F1_score_llama/classes_2/') or (INPUT_DIR == 'F1_score_llama/classes_3/'):
                    #    df['only_file_name'] = df['pred_path'].map(lambda x: x.split('/')[-1])
                    #else:
                    df['only_file_name'] = df['Prediction_path'].map(lambda x: x.split('/')[-1]) 

                    res_df = df.groupby('only_file_name').agg(F1_score_mean=pd.NamedAgg(column='F1_score', aggfunc='mean'), F1_score_st_dev=pd.NamedAgg(column='F1_score', aggfunc='std'), accuracy_mean=pd.NamedAgg(column='accuracy',  aggfunc='mean'), accuracy_std_dev=pd.NamedAgg(column='accuracy',  aggfunc='std'), 
                    accuracy_mean_low=pd.NamedAgg(column='accuracy_low',  aggfunc='mean'), accuracy_mean_medium=pd.NamedAgg(column='accuracy_medium',  aggfunc='mean'), accuracy_mean_high=pd.NamedAgg(column='accuracy_high',  aggfunc='mean'),
                    st_dev_mean_low=pd.NamedAgg(column='accuracy_low',  aggfunc='std'), std_dev_mean_medium=pd.NamedAgg(column='accuracy_medium',  aggfunc='std'), st_dev_mean_high=pd.NamedAgg(column='accuracy_high',  aggfunc='std')).reset_index()# F1_score -> if classification | f1_bucketed -> if regression

                    #mean_f1_score = grouped_df['F1_score'].mean()

                    list_df += [res_df]

                    #print(list_df)

                    #print('\n\n')

                print(OUTPUT_PATH)
                
                output_df = pd.concat(list_df).sort_values('F1_score_mean')
                output_df.to_csv(OUTPUT_PATH, index=False)

                # total mean per tweets num
                mean_df = df.mean(numeric_only=True).to_frame(name='mean').T
                mean_df['num_tweets'] = num_tweets
                mean_tot_df = pd.concat([mean_tot_df, mean_df])
            
            except Exception as e:
                print(f"An error occurred for class_num {CLASS_NUM} and num_tweets {num_tweets}: {e}")
                continue
            
            mean_tot_df.to_csv(OUTPUT_PATH_MEAN, index=False)
 
    
    '''# concatenate F1 pre and post

    tot_score_pre = pd.read_csv('F1_score_dev_normal_pre_avg_v2_ILE_DE_FRANCE/pre_avg_tot_scores_classes_2.csv')
    tot_score_post = pd.read_csv('F1_score_dev_normal_post_ILE_DE_FRANCE/post_tot_scores_classes_2.csv')
    
    # add column to identify whether they come from pre or post
    tot_score_pre['aggregation_type'] = 'pre'
    tot_score_post['aggregation_type'] = 'post'

    # concatenate
    tot_score = pd.concat([tot_score_pre, tot_score_post]).sort_values('F1_score_mean')

    # save total results
    tot_score.to_csv('F1_scores_bert.csv', index=False)'''








    