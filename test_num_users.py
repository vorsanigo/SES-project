import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def sample_tweets_per_user(df, m, user_col='user_id'):
    return (
        df.groupby(user_col)
        .apply(lambda x: x.sample(m))
        .reset_index(drop=True)
    )


def evaluation(class_num, col_pred, y_true, y_pred):

    # set prediction and true labels vectors
    #y_true = pd.read_csv(TEST_PATH)#, sep='\t'
    #y_pred = pd.read_csv(PRED_PATH)

    #print('pred path', PRED_PATH)
    #print('test path', TEST_PATH)
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
    print('y tot', y_tot)
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

    return f1_score_num, acc, acc_low, acc_medium, acc_high #, PRED_PATH


if __name__ == '__main__':

    print('Starting test_num_users.py')

    # 1) compute f1-score and accuracy by taking subsamples of tweets (1, 5, 15, 25, 30), repeating it 50 times for each number to take random combinations of tweets
    # we compute it on each run separately

    CLASS_NUM = '2_2'

    PRED_DIR = f'predictions_post_ILE_DE_FRANCE/classes_{CLASS_NUM}/'
    if CLASS_NUM == '2_2':
        TEST_PATH = 'dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv'
    else:
        TEST_PATH = 'dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium.csv'
    #F1_SCORE_DIR = f'F1_score_dev_normal_post_ILE_DE_FRANCE/classes_{CLASS_NUM}/' # 'F1_score_dev_normal_pre_avg_v2_ILE_DE_FRANCE/classes_2/' 'F1_score_dev_normal_pre_avg_v2_3_classes_ILE_DE_FRANCE/classes_3/' #'F1_score_dev_normal_post_ILE_DE_FRANCE/classes_2/' #F1_score_dev_normal_post_ILE_DE_FRANCE/classes_3/' #'F1_score_llama/classes_2/' F1_score_regression_ILE_DE_FRANCE

    OUTPUT_DIR = f'predictions_post_ILE_DE_FRANCE/0_TEST_NUM_TWEETS_TEST_CLASSES_{CLASS_NUM}/all_files/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_df = pd.read_csv(TEST_PATH)

    pred_type_directories = [d for d in os.listdir(PRED_DIR) if os.path.isdir(os.path.join(PRED_DIR, d))]
    print(pred_type_directories)

    num_iterations = 50
    num_tweets_list = [1, 5, 15, 25, 30]

    COL_PRED = 'user_id'
    #i=0
    count = 0

    '''for d in pred_type_directories:

        try:

        
            df_eval = pd.DataFrame()
            runs_dirs = os.listdir(PRED_DIR + d)
            r = os.listdir(PRED_DIR + d)
            print('r', r)
            runs_dirs = [run for run in os.listdir(PRED_DIR + d) if os.path.isdir(os.path.join(PRED_DIR + d, run))]
                    

            results_df_mean = pd.DataFrame()
            results_df_max = pd.DataFrame()

            #i = 0
            for run in runs_dirs:
                #print('i', i)
                run_file = d + '_tweets.csv'
                #run_files = [f for f in os.listdir(PRED_DIR + d + '/' + run) if os.path.isfile(os.path.join(PRED_DIR + d + '/' + run, f))]
                #for run_file in run_files:
                print('run file', run_file)
                #if run_file[-5] == 's': # if tweets files -> final letters are ...tweets.csv
                #    COL_PRED = 'tweet_id'
                
                #else:
                #    COL_PRED = 'user_id'
                # compute total accuracy and by class
                PRED_PATH = PRED_DIR + d + '/' + run + '/' + run_file

                tweets_pred_df = pd.read_csv(PRED_PATH)

                print(tweets_pred_df)
                print(test_df)

                # add user id
                tweets_pred_df = tweets_pred_df.merge(test_df[['user_id', 'tweet_id']], on='tweet_id')


                for num_tweets in num_tweets_list:

                    f1_scores_mean, accs_mean, accs_low_mean, accs_medium_mean, accs_high_mean = [], [], [], [], []
                    f1_scores_max, accs_max, accs_low_max, accs_medium_max, accs_high_max = [], [], [], [], []


                    for i in range(num_iterations):
                        print('num tweets', num_tweets)
                        print('num iterations', num_iterations)
                        print(i)

                        sampled = sample_tweets_per_user(tweets_pred_df, num_tweets)

                        if CLASS_NUM == '2' or CLASS_NUM == '2_2':

                            # aggregate per user and compute the mean of prob per class
                            user_scores = sampled.groupby("user_id")[["prob_0", "prob_1"]].mean().reset_index()
                            user_scores['winning_prob'] = user_scores[['prob_0', 'prob_1']].max(axis=1)
                            user_scores["final_label"] = np.where(
                                user_scores["prob_1"] >= user_scores["prob_0"],
                                1, 0
                            )

                            # aggregate per user and get as label the one appearing more often
                            user_preds = (
                                sampled
                                .groupby("user_id")["label"]
                                .agg(
                                    final_label=lambda x: x.mode()[0], # take most frequent value (0 or 1)
                                    n_tweets="count"
                                )
                                .reset_index()
                            )

                            #print('user scores', user_scores)

                            #user_scores["confidence"] = user_scores.max(axis=1)

                            # compute uncertainty as std of the predicted probabilities for each user
                            #test_df["pred_probs"] = probs.max(axis=1)
                            #user_uncertainty = test_df.groupby("user_id")["pred_probs"].std().reset_index(name="uncertainty")

                            
                            # df with predictions on single users (aggregated tweets) computing the mean of the probabilities for each class and taking as final label the one with highest mean probability, confidence score as the highest mean probability and uncertainty as the std of the predicted probabilities for each user
                            user_mean_df = pd.DataFrame({
                                'user_id': user_scores['user_id'],
                                'SES_users': user_scores['final_label'],
                                'mean_prob_0': user_scores['prob_0'],
                                'mean_prob_1': user_scores['prob_1'],
                            })#.to_csv(SAVE_PREDICTIONS_USERS_PATH, index=False)

                            # df with predictions on single users (aggregated tweets) copmuting as final label the one appearing more often among the tweets of the user and number of tweets for each user
                            user_max_df = pd.DataFrame({
                                'user_id': user_preds['user_id'],
                                'SES_users': user_preds['final_label'],
                                'n_tweets': user_preds['n_tweets']
                            })#.to_csv(SAVE_PREDICTIONS_USERS_MAX_PATH, index=False)
                        
                        elif CLASS_NUM == '3':

                            # aggregate per user — mean of probabilities per class
                            user_scores = sampled.groupby("user_id")[["prob_0", "prob_1", "prob_2"]].mean().reset_index()
                            user_scores["final_label"] = user_scores[["prob_0", "prob_1", "prob_2"]].values.argmax(axis=1)
                            user_scores["confidence"]  = user_scores[["prob_0", "prob_1", "prob_2"]].max(axis=1)

                            # aggregate per user — most frequent predicted label
                            user_preds = (
                                sampled
                                .groupby("user_id")["label"]
                                .agg(
                                    final_label=lambda x: x.mode()[0],
                                    n_tweets="count"
                                )
                                .reset_index()
                            )
                            
                            # df with predictions on single users (aggregated tweets) computing the mean of the probabilities for each class and taking as final label the one with highest mean probability, confidence score as the highest mean probability and uncertainty as the std of the predicted probabilities for each user
                            user_mean_df = pd.DataFrame({
                                'user_id':        user_scores['user_id'],
                                'label':          user_scores['final_label'],
                                'mean_prob_0':    user_scores['prob_0'],
                                'mean_prob_1':    user_scores['prob_1'],
                                'mean_prob_2':    user_scores['prob_2'],    # CHANGE 5
                            })#.to_csv(SAVE_PREDICTIONS_USERS_PATH, index=False)

                            # df with predictions on single users (aggregated tweets) copmuting as final label the one appearing more often among the tweets of the user and number of tweets for each user
                            user_max_df = pd.DataFrame({
                                'user_id':  user_preds['user_id'],
                                'label':    user_preds['final_label'],
                                'n_tweets': user_preds['n_tweets']
                            })#.to_csv(SAVE_PREDICTIONS_USERS_MAX_PATH, index=False)

                        f1_mean, acc_mean, acc_low_mean, acc_medium_mean, acc_high_mean = evaluation(CLASS_NUM, COL_PRED, test_df, user_mean_df)
                        f1_scores_mean.append(f1_mean)
                        accs_mean.append(acc_mean)
                        accs_low_mean.append(acc_low_mean)
                        accs_medium_mean.append(acc_medium_mean)
                        accs_high_mean.append(acc_high_mean)

                        f1_max, acc_max, acc_low_max, acc_medium_max, acc_high_max = evaluation(CLASS_NUM, COL_PRED, test_df, user_max_df)
                        f1_scores_max.append(f1_max)
                        accs_max.append(acc_max)
                        accs_low_max.append(acc_low_max)
                        accs_medium_max.append(acc_medium_max)
                        accs_high_max.append(acc_high_max)

                    results_df_mean = pd.concat([results_df_mean, pd.DataFrame({'num_run': run, 'num_tweets': num_tweets, 'f1_score_num': [np.mean(f1_scores_mean)], 
                    'acc': [np.mean(accs_mean)], 'acc_low': [np.mean(accs_low_mean)], 'acc_medium': [np.mean(accs_medium_mean)], 'acc_high': [np.mean(accs_high_mean)]})])         
                    
                    results_df_max = pd.concat([results_df_max, pd.DataFrame({'num_run': run, 'num_tweets': num_tweets, 'f1_score_num': [np.mean(f1_scores_max)], 
                    'acc': [np.mean(accs_max)], 'acc_low': [np.mean(accs_low_max)], 'acc_medium': [np.mean(accs_medium_max)], 'acc_high': [np.mean(accs_high_max)]})])         

            results_df_mean.to_csv(OUTPUT_DIR + d + '_MEAN.csv')
            results_df_max.to_csv(OUTPUT_DIR + d + '_MAX.csv')
            #y_true = pd.read_csv(TEST_PATH)
        

        except Exception as e:
            print(f"Error: {e}")
            continue'''
     

    # 2) average pwr num tweets over the runs for each strategy settinng

    '''CLASSES = '3' # 2 | 3 | 2_2 
    INPUT_DIR = f'predictions_post_ILE_DE_FRANCE/0_TEST_NUM_TWEETS_TEST_CLASSES_{CLASSES}/all_files/'
    OUTPUT_DIR = f'predictions_post_ILE_DE_FRANCE/0_TEST_NUM_TWEETS_TEST_CLASSES_{CLASSES}/'
    output_file_strategy = f'mean_results_per_strategy_{CLASSES}.csv'
    output_file_tot = f'mean_results_tot_{CLASSES}.csv'

    files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]

    df_tot = pd.DataFrame()

    for file in files:
        print(file)
        df = pd.read_csv(INPUT_DIR + file)
        avg_per_num_tweets = df.groupby('num_tweets').agg(f1_score_num=pd.NamedAgg(column='f1_score_num', aggfunc='mean'),
                                                            acc=pd.NamedAgg(column='acc', aggfunc='mean'),
                                                            acc_low=pd.NamedAgg(column='acc_low', aggfunc='mean'),
                                                            acc_medium=pd.NamedAgg(column='acc_medium', aggfunc='mean'),
                                                            acc_high=pd.NamedAgg(column='acc_high', aggfunc='mean')).reset_index()
        avg_per_num_tweets['type'] = file
        df_tot = pd.concat([df_tot, avg_per_num_tweets])

        print(avg_per_num_tweets)
        print('\n')

    # avg over tweets number over all the strategies
    df_tot_mean = df_tot.groupby('num_tweets').agg(f1_score_num=pd.NamedAgg(column='f1_score_num', aggfunc='mean'),
                                                            acc=pd.NamedAgg(column='acc', aggfunc='mean'),
                                                            acc_low=pd.NamedAgg(column='acc_low', aggfunc='mean'),
                                                            acc_medium=pd.NamedAgg(column='acc_medium', aggfunc='mean'),
                                                            acc_high=pd.NamedAgg(column='acc_high', aggfunc='mean')).reset_index()
    
    # save tot df
    df_tot.to_csv(OUTPUT_DIR + output_file_strategy, index=False)

    # save tot mean df
    df_tot_mean.to_csv(OUTPUT_DIR + output_file_tot, index=False)'''


    # 2.2) plot f1/accuracy scores per tweets number -> with tot mean df -> a posteriori case

    #CLASSES = '2' # 2 | 2_2 | 3
    #INPUT_FILE_2 = 'predictions_post_ILE_DE_FRANCE/0_TEST_NUM_TWEETS_TEST_CLASSES_2/mean_results_tot_2.csv'
    '''INPUT_FILE_2_2 = 'predictions_post_ILE_DE_FRANCE/0_TEST_NUM_TWEETS_TEST_CLASSES_2_2/mean_results_tot_2_2.csv'
    INPUT_FILE_3 = 'predictions_post_ILE_DE_FRANCE/0_TEST_NUM_TWEETS_TEST_CLASSES_3/mean_results_tot_3.csv'

    PLOT_COLUMN = 'acc' # f1_score_num | acc
    
    if PLOT_COLUMN == 'acc':
        y_label = 'Accuracy'
        OUTPUT_FILE = 'predictions_post_ILE_DE_FRANCE/mean_results_tot_accuracy.png'
    else:
        y_label = 'F1 score'
        OUTPUT_FILE = 'predictions_post_ILE_DE_FRANCE/mean_results_tot_f1_score.png'

    # font size
    ticks_size = 17
    labels_size = 25
    legend_size = 19

    #scores_df_2 = pd.read_csv(INPUT_FILE_2)
    scores_df_2_2 = pd.read_csv(INPUT_FILE_2_2)
    scores_df_3 = pd.read_csv(INPUT_FILE_3)

    #scores_df_2 = scores_df_2[scores_df_2['num_tweets'] != 25]
    scores_df_2_2 = scores_df_2_2[scores_df_2_2['num_tweets'] != 25]
    scores_df_3 = scores_df_3[scores_df_3['num_tweets'] != 25]

    fig,ax = plt.subplots(figsize = (9,9))

    #ax.plot(scores_df_2['num_tweets'], scores_df_2[PLOT_COLUMN], '-o', label='2 separated classes')
    ax.plot(scores_df_2_2['num_tweets'], scores_df_2_2[PLOT_COLUMN], '-o', label='2 classes')
    ax.plot(scores_df_3['num_tweets'], scores_df_3[PLOT_COLUMN], '-o', label='3 classes')

    ax.tick_params(axis='both', which='major', labelsize=ticks_size)
    ax.set_xlabel('Tweets number', fontsize=labels_size)
    ax.set_ylabel(y_label, fontsize=labels_size)

    ax.legend(fontsize=legend_size)
    plt.tight_layout()

    fig.savefig(OUTPUT_FILE)
    '''
    
    # 2.3) plot f1/accuracy scores per tweets number -> with tot mean df -> a priori case

    PLOT_COLUMN = 'accuracy' # F1_score | accuracy
    OUTPUT_FILE = f'mean_results_tot_{PLOT_COLUMN}_priori.png'

    if PLOT_COLUMN == 'accuracy':
        y_label = 'Accuracy'
        #OUTPUT_FILE = 'predictions_post_ILE_DE_FRANCE/mean_results_tot_accuracy.png'
    else:
        y_label = 'F1 score'
        #OUTPUT_FILE = 'predictions_post_ILE_DE_FRANCE/mean_results_tot_f1_score.png'

    df_priori_list = []

    for CLASS_NUM in ['2_2', '3']:#'2', 

        if CLASS_NUM == '2':
            df_posteriori = pd.read_csv('predictions_post_ILE_DE_FRANCE/0_TEST_NUM_TWEETS_TEST_CLASSES_2/mean_results_tot_2.csv')
            df_priori = pd.read_csv('F1_score_dev_normal_pre_avg_v2_ILE_DE_FRANCE/tot_mean_scores_classes_2.csv')
        elif CLASS_NUM == '2_2':
            df_posteriori = pd.read_csv('predictions_post_ILE_DE_FRANCE/0_TEST_NUM_TWEETS_TEST_CLASSES_2_2/mean_results_tot_2_2.csv')
            df_priori = pd.read_csv('F1_score_dev_normal_pre_avg_v2_ILE_DE_FRANCE/tot_mean_scores_classes_2_2.csv')
        elif CLASS_NUM == '3':
            df_posteriori = pd.read_csv('predictions_post_ILE_DE_FRANCE/0_TEST_NUM_TWEETS_TEST_CLASSES_3/mean_results_tot_3.csv')
            df_priori = pd.read_csv('F1_score_dev_normal_pre_avg_v2_3_classes_ILE_DE_FRANCE/tot_mean_scores_classes_3.csv')

        print(CLASS_NUM)
        print(df_posteriori)
        print(df_priori)
        df_posteriori = df_posteriori.rename(columns={'f1_score_num': 'F1_score', 'acc': 'accuracy'})
        df_posteriori = df_posteriori[['num_tweets', 'F1_score', 'accuracy']]
        df_posteriori = df_posteriori[df_posteriori['num_tweets'] == 1]
        df_priori = df_priori[['num_tweets', 'F1_score', 'accuracy']]

        df_priori_tot = pd.concat([df_priori, df_posteriori], ignore_index=True).sort_values(by='num_tweets').reset_index(drop=True)
        print('tot', df_priori_tot)

        df_priori_list.append(df_priori_tot)

    # font size
    ticks_size = 17
    labels_size = 25
    legend_size = 19

    fig,ax = plt.subplots(figsize = (9,9))

    print(df_priori_list[0]['num_tweets'])
    ax.plot(df_priori_list[0]['num_tweets'], df_priori_list[0][PLOT_COLUMN], '-o', label='2 classes')
    #ax.plot(df_priori_list[1]['num_tweets'], df_priori_list[1][PLOT_COLUMN], '-o', label='2 non-separated classes')
    ax.plot(df_priori_list[1]['num_tweets'], df_priori_list[1][PLOT_COLUMN], '-o', label='3 classes')

    ax.tick_params(axis='both', which='major', labelsize=ticks_size)
    ax.set_xlabel('Tweets number', fontsize=labels_size)
    ax.set_ylabel(y_label, fontsize=labels_size)

    ax.legend(fontsize=legend_size)
    plt.tight_layout()

    fig.savefig(OUTPUT_FILE)

    print('Done')


'''if count == 0:
break'''
    

'''if count == 0:
    break'''




'''y_pred = pd.read_csv(PRED_PATH)
y_pred = y_pred.drop('use')
f1_score_num, acc, acc_low, acc_medium, acc_high = evaluation(CLASS_NUM, COL_PRED, y_true, y_pred)
df_eval = pd.concat([df_eval, pd.DataFrame({'run_num': [i+1], 'F1_score': [f1_score_num], 'accuracy': [acc], 'accuracy_low': [acc_low], 'accuracy_medium': [acc_medium], 'accuracy_high': [acc_high], 'Prediction_path': [PRED_PATH]})])
i += 1
df_eval.to_csv(F1_SCORE_DIR + d + '_PROVAAAAAAA_ZZZZZZ.csv')
'''
    