from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
import numpy as np
#from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import polars as pl
import random
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from safetensors.torch import load_file
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os


# FUNCTIONS

# with features
class CamembertWithFeatures(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.bert = AutoModel.from_pretrained("camembert-base")

        #self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, features, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]

        if scaling == True:
            combined = torch.cat([cls, features.float()], dim=1)
        else:
            combined = torch.cat([cls, features], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)


        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


# without features
class CamembertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("camembert-base")

        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


def add_features(example):
    example["features"] = [example[col] for col in feature_cols]
    return example


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def compute_f1(y_true, y_pred):
    
    f1 = f1_score(y_true, y_pred)

    return f1


def compute_f1_macro(y_true, y_pred, classes):

    if len(classes) == 0:
        class_name = classes[0]
        print(np.array(list(y_true[class_name])))
        print(np.array(list(y_pred[class_name])))
        global_f1 = f1_score(np.array(list(y_true[class_name])), np.array(list(y_pred[class_name])), average='macro')
    
    else:

        y_true_classes = []
        y_pred_classes = []

        for class_name in classes:
            y_true_classes += [list(y_true[class_name])]
            y_pred_classes += [list(y_pred[class_name])]
        
        # Global macro F1
        global_f1 = f1_score(np.array(y_true_classes).T, np.array(y_pred_classes).T, average='macro')

    return global_f1



# read df

def prepare_data(with_features, feature_cols, TRAIN_FINAL_SES_ERRORS_PATH, DEV_FINAL_SES_ERRORS_PATH):

    print('Reading files')

    train_df = pd.read_csv(TRAIN_FINAL_SES_ERRORS_PATH)#, sep='\t'
    dev_df = pd.read_csv(DEV_FINAL_SES_ERRORS_PATH)#, sep='\t'

    if with_features == False:
        # keep only useful columns
        train_df = train_df[['tweet_id', 'user_id', 'text', 'SES_users']]
        dev_df = dev_df[['tweet_id', 'user_id', 'text', 'SES_users']]

    print(train_df)
    print(dev_df)

    # rename column ses labels for right format for training
    train_df = train_df.rename(columns={'SES_users': 'labels'})
    dev_df = dev_df.rename(columns={'SES_users': 'labels'})

    # keep only low and high class
    train_df = train_df[train_df['labels'].isin(['lower', 'high'])]
    dev_df = dev_df[dev_df['labels'].isin(['lower', 'high'])]

    label_map = {"lower": 0, "high": 1}

    train_df["labels"] = train_df["labels"].map(label_map)
    dev_df["labels"]   = dev_df["labels"].map(label_map)

    print(train_df["labels"].isna().sum())

    if with_features == True and scaling == True:

        print("Standardizing features")

        scaler = StandardScaler()

        # extract feature matrix
        X_train = train_df[feature_cols].values
        X_dev   = dev_df[feature_cols].values
        #X_test  = test_df[feature_cols].values

        # fit ONLY on train
        scaler.fit(X_train)

        # transform all
        train_df[feature_cols] = scaler.transform(X_train)
        dev_df[feature_cols]   = scaler.transform(X_dev)
        #test_df[feature_cols]  = scaler.transform(X_test)

        # save scaler
        joblib.dump(scaler, f"{SAVE_MODEL_DIR}/scaler.pkl")


    # 2) tokenization

    print('Preparing data')

    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    # huggingface df
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    def tokenize(batch, text_column="text"):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=128
    )

    train_dataset = train_dataset.map(tokenize, batched=True)#, remove_columns=train_dataset.column_names
    dev_dataset   = dev_dataset.map(tokenize, batched=True)#, remove_columns=dev_dataset.column_names

    # add features vector
    if with_features == True:
        train_dataset = train_dataset.map(add_features)
        dev_dataset   = dev_dataset.map(add_features)

    print(train_dataset)
    print(dev_dataset)


    if with_features == True:
        # set format pytorch
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "features", "labels"]
        )
        dev_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "features", "labels"]
        )
    else:
        # set format pytorch
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        dev_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

    print(train_dataset)
    print(dev_dataset)

    return train_dataset, dev_dataset, tokenizer


# 1) TRAIN
def train(with_features, training_args, train_dataset, dev_dataset, tokenizer, SAVE_MODEL_DIR):
    print('TRAIN: Preparing functions and defining model')


    if with_features == True:
        n_features=len(feature_cols)
        model = CamembertWithFeatures(n_features=len(feature_cols))
        print('model CamembertWithFeatures with features')
    else:
        print('model CamembertClassifier without features')
        model = CamembertClassifier()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )

    print('TRAIN: Training the model')
    trainer.train()

    try:
        # save model weights
        torch.save(model.state_dict(), f"{SAVE_MODEL_DIR}/model.pt")
        # save tokenizer
        tokenizer.save_pretrained(SAVE_MODEL_DIR)
        # (optional) save feature config

        # save config
        config = {
            "with_features": with_features,
        }

        if with_features:
            config["n_features"] = n_features #len(feature_cols)

        with open(f"{SAVE_MODEL_DIR}/config.json", "w") as f:
            json.dump(config, f)

    except Exception as e:
        print('Not able to save model:', e)


# 2) TEST

def test(with_features, training_args, feature_cols, TEST_FINAL_SES_ERRORS_PATH, SAVE_MODEL_DIR, SAVE_PREDICTIONS_TWEETS_PATH, SAVE_PREDICTIONS_USERS_PATH, SAVE_PREDICTIONS_USERS_MAX_PATH): 
    
    test_df = pd.read_csv(TEST_FINAL_SES_ERRORS_PATH)#, sep='\t'

    if with_features == False:
        test_df = test_df[['tweet_id', 'user_id', 'text', 'SES_users']]

    test_df = test_df.rename(columns={'SES_users': 'labels'})

    test_df = test_df[test_df['labels'].isin(['lower', 'high'])]

    label_map = {"lower": 0, "high": 1}
    test_df["labels"]  = test_df["labels"].map(label_map)


    # reload config
    with open(f"{SAVE_MODEL_DIR}/config.json") as f:
        config = json.load(f)
        
    # rebuild model architecture
    if with_features == True:
        model = CamembertWithFeatures(n_features=config["n_features"])
        print('model CamembertWithFeatures with features')
    else:
        model = CamembertClassifier()
        print('model CamembertClassifier without features')

    # load weights
    model.load_state_dict(torch.load(f"{SAVE_MODEL_DIR}/model.pt"))

    if with_features == True and scaling == True:
        # load scaler
        scaler = joblib.load(f"{SAVE_MODEL_DIR}/scaler.pkl")
        # apply loader
        X_test = test_df[feature_cols].values
        test_df[feature_cols] = scaler.transform(X_test)

    model.eval()

    print('TEST: Preparing test data')

    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    def tokenize(batch, text_column="text"):

        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=128
    )

    test_dataset = test_dataset.map(tokenize, batched=True)

    if with_features == True:
        test_dataset = test_dataset.map(add_features)
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "features"]#, "labels"
        )
    else:
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"]#, "labels"
        )

    print('TEST: Prediction')

    trainer = Trainer(
        model=model,
        args=training_args
    )

    predictions = trainer.predict(test_dataset)

    logits = predictions.predictions
    #preds = logits.argmax(axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)

    test_df["predicted_label"] = preds
    test_df["prob_0"] = probs[:, 0]
    test_df["prob_1"] = probs[:, 1]

    print('test df', test_df)

    # aggregate per user and compute the mean of prob per class
    user_scores = test_df.groupby("user_id")[["prob_0", "prob_1"]].mean().reset_index()
    user_scores['winning_prob'] = user_scores[['prob_0', 'prob_1']].max(axis=1)
    user_scores["final_label"] = np.where(
        user_scores["prob_1"] >= user_scores["prob_0"],
        1, 0
    )

    # aggregate per user and get as label the one appearing more often
    user_preds = (
        test_df
        .groupby("user_id")["predicted_label"]
        .agg(
            final_label=lambda x: x.mode()[0], # take most frequent value (0 or 1)
            n_tweets="count"
        )
        .reset_index()
    )

    print('user scores', user_scores)

    user_scores["confidence"] = user_scores.max(axis=1)

    # compute uncertainty as std of the predicted probabilities for each user
    test_df["pred_probs"] = probs.max(axis=1)
    user_uncertainty = test_df.groupby("user_id")["pred_probs"].std().reset_index(name="uncertainty")

    test_evaluation = trainer.evaluate(test_dataset)
    print(test_evaluation)
        
    print(user_scores.columns)

    # df with predictions on single tweets
    pd.DataFrame({
        'tweet_id': test_df['tweet_id'],
        'label': preds,
        'prob_0': test_df['prob_0'],
        'prob_1': test_df['prob_1']
    }).to_csv(SAVE_PREDICTIONS_TWEETS_PATH, index=False)

    # df with predictions on single users (aggregated tweets) computing the mean of the probabilities for each class and taking as final label the one with highest mean probability, confidence score as the highest mean probability and uncertainty as the std of the predicted probabilities for each user
    pd.DataFrame({
        'user_id': user_scores['user_id'],
        'label': user_scores['final_label'],
        'mean_prob_0': user_scores['prob_0'],
        'mean_prob_1': user_scores['prob_1'],
        'confidence': user_scores['confidence'],
        'uncertainty_std': user_uncertainty['uncertainty']
    }).to_csv(SAVE_PREDICTIONS_USERS_PATH, index=False)

    # df with predictions on single users (aggregated tweets) copmuting as final label the one appearing more often among the tweets of the user and number of tweets for each user
    pd.DataFrame({
        'user_id': user_preds['user_id'],
        'label': user_preds['final_label'],
        'n_tweets': user_preds['n_tweets']
    }).to_csv(SAVE_PREDICTIONS_USERS_MAX_PATH, index=False)


# 4) EVALUATION

def evaluation(class_num, col_pred, TEST_PATH, PRED_PATH):
    
    # set paths
    #TEST_PATH = 'dev_phase_normal/input with medium/TEST_TEST_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv' #'dev_phase_normal/input with medium/TEST_TEST_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length.csv'#TEST_ground_truth_users_mistakes_PARIS_no_spammers_OK_INPUT_FEAT_500_users_NEW.tsv

    # select column: tweet_id or user_id
    #col_pred = 'user_id' # tweet_id | user_id

    # set prediction and true labels vectors
    y_true = pd.read_csv(TEST_PATH)#, sep='\t'
    y_pred = pd.read_csv(PRED_PATH)

    print('y_true initial', y_true)

    # set columns predictions df
    #y_pred.columns = COLUMNS_TOT_ID_TEXT

    if col_pred == 'user_id':
        print('user')
        # keep only one example per user since we check the user's label
        #y_true = y_true.drop_duplicates(col_pred, keep='first')
        # change labels
        if class_num == 2:
            label_map = {"lower": 0, "high": 1}
        elif class_num == 3:
            label_map = {"lower": 0, "medium": 1, "high": 2}
        y_true["SES_users"] = y_true["SES_users"].map(label_map)

    else:
        print('tweet')
        # keep only one example per user since we check the user's label
        #y_true = y_true.drop_duplicates('user_id', keep='first')
        # change labels
        if class_num == 2:
            label_map = {"lower": 0, "high": 1}
        elif class_num == 3:
            label_map = {"lower": 0, "medium": 1, "high": 2}
        y_true["SES_users"] = y_true["SES_users"].map(label_map)
        

    print('y_true filtered', y_true)
    print('y_pred', y_pred)

    y_pred = y_pred.rename(columns={'label': 'SES_users'})

    y_tot = pd.merge(y_true, y_pred, on=col_pred, suffixes=('_true', '_pred'))
    y_tot = y_tot[['SES_users_true', 'SES_users_pred', col_pred]]

    print('y_tot', y_tot)

    # sort by tweet_id
    '''y_true = y_true.sort_values(by='tweet_id')
    y_pred = y_pred.sort_values(by='tweet_id')'''

    # compute F1 score
    if class_num == 2:
        f1_score_num = compute_f1(y_tot['SES_users_true'], y_tot['SES_users_pred'])
    elif class_num == 3:
        f1_score_num = f1_score(y_tot['SES_users_true'], y_tot['SES_users_pred'], average='macro')

    return f1_score_num, PRED_PATH
    '''# save F1 scores
    scores = {'F1': [f1_score_num], 'configuration': [PRED_PATH]}
    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(OUTPUT_PATH, index=False)'''


if __name__ == "__main__":

    num_runs = 3

    seeds = [150, 300, 999]#

    df_eval = pd.DataFrame()

    # variable of num classes
    classes = '2_2' # 2 | 2_2 | 3
    if classes == '2' or classes == '2_2':
        class_num = 2 # 2 | 3
    else:
        class_num = 3

    features_macro = 'only_text' # all_features | corr_features | only_text | neg_plur | len_vocab_neg_plur

    masked_location = True

    tweets_selection = 'random_tweets' # random | chosen
    
    # region to use -> df
    region = 'ILE_DE_FRANCE' # PARIS | ILE_DE_FRANCE

    # variable to decide if keeping features
    if features_macro == 'only_text':
        with_features = False # True | False
    else:
        with_features = True

    # variable to decide if scaling features
    scaling = False # True | False
    if scaling == True:
        scaling_name = '_scaling'
    else:
        scaling_name = ''

    # predicted column
    col_pred_tweet = 'tweet_id'
    col_pred_user = 'user_id'

    # input paths
    if masked_location == True:
        TRAIN_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_ner_stanza_train_location_2_classes.csv'#tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classestrain_paris_LARGE_LARGE_tot_errors_with_medium_2_classes
        DEV_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_ner_stanza_test_dev_location_2_classes.csv'#tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT
        TEST_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_ner_stanza_test_test_location_2_classes.csv'#tweets_TEST_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes
    elif region == 'PARIS':
        TRAIN_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium/TRAIN_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv'
        DEV_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium/TEST_DEV_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv'
        TEST_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium/TEST_TEST_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv'
    elif region == 'ILE_DE_FRANCE':
        if classes == '2_2':
            '''TRAIN_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes.csv'#train_paris_LARGE_LARGE_tot_errors_with_medium_2_classes
            DEV_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_TEST_DEV_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes.csv'#tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT
            TEST_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_TEST_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes.csv'#'''
            TRAIN_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv'#tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classestrain_paris_LARGE_LARGE_tot_errors_with_medium_2_classes
            DEV_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/test_dev_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv'#tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT
            TEST_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv'#tweets_TEST_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes
        else:
            '''TRAIN_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT.csv'#_2_classes
            DEV_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_TEST_DEV_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes.csv'#_2_classes test_dev_paris_LARGE_LARGE_tot_errors_with_medium
            TEST_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_TEST_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes.csv'#_2_classes'''
            TRAIN_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium.csv'#tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes
            DEV_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/test_dev_paris_LARGE_LARGE_tot_errors_with_medium.csv'#tweets_TEST_DEV_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes 
            TEST_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium.csv'#tweets_TEST_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes
            
    # features (error features) on which we apply normalization 
    # smaller df (Paris large dataset)
    NORM_FEATURES = []
    if region == 'PARIS':
        if features_macro == 'all_features':
            NORM_FEATURES = ['AGREEMENT', 'CASING', 'CAT_ELISION', 'CAT_GRAMMAIRE', 'CAT_HOMONYMES_PARONYMES', 'CAT_MAJUSCULES', 'CAT_PLEONASMES', 'CAT_REGIONALISMES', 'CAT_REGLES_DE_BASE', 'CAT_TOURS_CRITIQUES', 'CAT_TYPOGRAPHIE', 'MISC', 'MULTITOKEN_SPELLING', 'PONCTUATION_POINT', 'PONCTUATION_VIRGULE', 'PUNCTUATION', 'REPETITIONS_STYLE', 'SEMANTICS', 'STYLE', 'TYPOGRAPHY', 'TYPOS']
        elif features_macro == 'corr_features':
            NORM_FEATURES = ['CAT_GRAMMAIRE', 'TYPOS', 'CAT_HOMONYMES_PARONYMES', 'AGREEMENT', 'PONCTUATION_VIRGULE', 'CAT_ELISION', 'CAT_TYPOGRAPHIE', 'CAT_TOURS_CRITIQUES', 'MISC']            
    # large df (Ile-de-France dataset)
    elif region == 'ILE_DE_FRANCE':
        if features_macro == 'all_features':
            NORM_FEATURES = ['AGREEMENT', 'CASING', 'CAT_ELISION', 'CAT_GRAMMAIRE', 'CAT_HOMONYMES_PARONYMES', 'CAT_MAJUSCULES', 'CAT_PLEONASMES', 'CAT_REGIONALISMES', 'CAT_REGLES_DE_BASE', 'CAT_TOURS_CRITIQUES', 'CAT_TYPOGRAPHIE', 'MISC', 'MULTITOKEN_SPELLING', 'PONCTUATION_POINT', 'PONCTUATION_VIRGULE', 'PUNCTUATION', 'REPETITIONS_STYLE', 'SEMANTICS', 'STYLE', 'TYPOGRAPHY', 'TYPOS', 'CAT_MARQUES_DE_COMMERCE']
            #['AGREEMENT', 'CASING', 'CAT_ELISION', 'CAT_GRAMMAIRE', 'CAT_HOMONYMES_PARONYMES', 'CAT_MAJUSCULES', 'CAT_PLEONASMES', 'CAT_REGIONALISMES', 'CAT_REGLES_DE_BASE', 'CAT_TOURS_CRITIQUES', 'CAT_TYPOGRAPHIE', 'MISC', 'MULTITOKEN_SPELLING', 'PONCTUATION_POINT', 'PONCTUATION_VIRGULE', 'PUNCTUATION', 'REPETITIONS_STYLE', 'SEMANTICS', 'STYLE', 'TYPOGRAPHY', 'TYPOS']#, 'CAT_MARQUES_DE_COMMERCE'
        elif features_macro == 'corr_features':
            NORM_FEATURES = ['CAT_GRAMMAIRE', 'TYPOS', 'CAT_HOMONYMES_PARONYMES', 'AGREEMENT', 'PONCTUATION_VIRGULE', 'CAT_ELISION', 'CAT_TYPOGRAPHIE', 'CAT_TOURS_CRITIQUES', 'MISC']
        
    if features_macro == 'all_features':#
        FEATURES = ['all_features_with_length_vocab_negation', 'all_features_with_length_vocab_pluralization', 'all_features_with_length_vocab_negation_pluralization']
        # 'all_features_negation', 'all_features_with_length_negation', 'all_features_with_length_pluralization', 
    elif features_macro == 'corr_features':#
        FEATURES = ['corr_features_with_length_vocab_negation', 'corr_features_with_length_vocab_negation_pluralization', 'corr_features_with_length_vocab_pluralization']
        # 'corr_features_negation', 'corr_features_with_length_negation', 'corr_features_with_length_pluralization', 
    elif features_macro == 'neg_plur':
        FEATURES = ['neg_plur']
    elif features_macro == 'len_vocab_neg_plur':
        FEATURES = ['len_vocab_neg_plur']
    elif features_macro == 'only_text':
        FEATURES = ['only_text']

    for features in FEATURES:

        df_eval = pd.DataFrame()

        if features == 'only_text':
            with_features = False
        else:
            with_features = True
        
        # output eval
        if masked_location == True:
            SAVE_DIR_EVAL = f'F1_score_dev_normal_post_ILE_DE_FRANCE_MASKED_LOC/classes_{classes}'#_{tweets_selection}
            os.makedirs(SAVE_DIR_EVAL, exist_ok=True)
            OUTPUT_PATH_EVAL = f'{SAVE_DIR_EVAL }/{features}{scaling_name}.csv'
        else:
            SAVE_DIR_EVAL = f'F1_score_dev_normal_post_ILE_DE_FRANCE/classes_{classes}'#_{tweets_selection}
            os.makedirs(SAVE_DIR_EVAL, exist_ok=True)
            OUTPUT_PATH_EVAL = f'{SAVE_DIR_EVAL }/{features}{scaling_name}.csv'
     
        for i in range(num_runs):
        #for i in range(1):
        
            if num_runs == 1:
                
                # output paths
                CHECKPOINTS_DIR = './checkpoints_post_ILE_DE_FRANCE/classes_2/' + features + scaling_name
                SAVE_MODEL_DIR = './models_post_ILE_DE_FRANCE/classes_2_2/' + features + scaling_name
                SAVE_PREDICTIONS_TWEETS_PATH = './predictions_post_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_tweets.csv'
                SAVE_PREDICTIONS_USERS_PATH = './predictions_post_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_users_mean.csv'
                SAVE_PREDICTIONS_USERS_MAX_PATH = './predictions_post_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_users_max.csv'
                
                # evaluation paths
                # single tweets
                PRED_PATH_TWEETS = 'predictions_post_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_tweets.csv'#predictions_normal_training_features_corr_NEW.csv
                OUTPUT_PATH_TWEETS = 'F1_score_dev_normal_post_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_tweets.csv'#predictions_normal_training_features_corr_NEW.tsv
                # users avg
                PRED_PATH_USERS = 'predictions_post_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_users_mean.csv'#predictions_normal_training_features_corr_NEW.csv
                OUTPUT_PATH_USERS = 'F1_score_dev_normal_post_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_users_mean.csv'#predictions_normal_training_features_corr_NEW.tsv
                # users max
                PRED_PATH_USERS_MAX = 'predictions_post_ILE_DE_FRANCE/classes_2_2/' + features + scaling_name + '_users_max.csv'#predictions_normal_training_features_corr_NEW.csv
                OUTPUT_PATH_USERS_MAX = 'F1_score_dev_normal_post_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_users_max.csv'#predictions_normal_training_features_corr_NEW.tsv
            
            else:
                
                # output paths

                if masked_location == True:

                    CHECKPOINTS_DIR = f'./checkpoints_post_ILE_DE_FRANCE_MASKED_LOC/classes_{classes}/{features}{scaling_name}/run_{i+1}'#_{tweets_selection}
                    
                    SAVE_MODEL_DIR = f'./models_post_ILE_DE_FRANCE_MASKED_LOC/classes_{classes}/{features}{scaling_name}/run_{i+1}'#_{tweets_selection}
                    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

                    SAVE_PREDICTIONS_DIR = f'./predictions_post_ILE_DE_FRANCE_MASKED_LOC/classes_{classes}/{features}{scaling_name}/run_{i+1}'#_{tweets_selection}
                    os.makedirs(SAVE_PREDICTIONS_DIR, exist_ok=True)
                    SAVE_PREDICTIONS_TWEETS_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_tweets.csv'
                    SAVE_PREDICTIONS_USERS_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_users_mean.csv'
                    SAVE_PREDICTIONS_USERS_MAX_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_users_max.csv'

                else:

                    CHECKPOINTS_DIR = f'./checkpoints_post_ILE_DE_FRANCE/classes_{classes}/{features}{scaling_name}/run_{i+1}'#_{tweets_selection}
                    
                    SAVE_MODEL_DIR = f'./models_post_ILE_DE_FRANCE/classes_{classes}/{features}{scaling_name}/run_{i+1}'#_{tweets_selection}
                    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

                    SAVE_PREDICTIONS_DIR = f'./predictions_post_ILE_DE_FRANCE/classes_{classes}/{features}{scaling_name}/run_{i+1}'#_{tweets_selection}
                    os.makedirs(SAVE_PREDICTIONS_DIR, exist_ok=True)
                    SAVE_PREDICTIONS_TWEETS_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_tweets.csv'
                    SAVE_PREDICTIONS_USERS_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_users_mean.csv'
                    SAVE_PREDICTIONS_USERS_MAX_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_users_max.csv'

                
                # evaluation paths
                # single tweets
                PRED_PATH_TWEETS = SAVE_PREDICTIONS_TWEETS_PATH #+ '/'  + features + scaling_name + '_tweets.csv'#predictions_normal_training_features_corr_NEW.csv
                #OUTPUT_PATH_TWEETS = 'F1_score_dev_normal/classes_2/' + features + scaling_name + '_tweets.csv'#predictions_normal_training_features_corr_NEW.tsv
                # users avg
                PRED_PATH_USERS = SAVE_PREDICTIONS_USERS_PATH #+ '/'  + features + scaling_name + '_users_mean.csv'#predictions_normal_training_features_corr_NEW.csv
                #OUTPUT_PATH_USERS = 'F1_score_dev_normal/classes_2/' + features + scaling_name + '_users_mean.csv'#predictions_normal_training_features_corr_NEW.tsv
                # users max
                PRED_PATH_USERS_MAX = SAVE_PREDICTIONS_USERS_MAX_PATH #+ '/'  + features + scaling_name + '_users_max.csv'#predictions_normal_training_features_corr_NEW.csv
                #OUTPUT_PATH_USERS_MAX = 'F1_score_dev_normal/classes_2/' + features + scaling_name + '_users_max.csv'#predictions_normal_training_features_corr_NEW.tsv
            

            '''# variable to decide which features keep
            #features = 'corr_features_with_length_vocab' #'features_corr'

            # variable to decide if keeping only one tweet per user
            one_tweet_user = False # True | False'''


            # variablesnorm
            feature_cols = []
            if features == 'all_features':
                feature_cols = NORM_FEATURES
            elif features == 'corr_features':
                feature_cols = NORM_FEATURES
            elif features == 'all_features_negation':
                feature_cols = NORM_FEATURES + ['has_standard_neg', 'has_nonstandard_neg']
            elif features == 'corr_features_negation':
                feature_cols = NORM_FEATURES + ['has_standard_neg', 'has_nonstandard_neg']
            elif features == 'all_features_with_length':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS']
            elif features == 'corr_features_with_length':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS']
            elif features == 'all_features_with_length_negation':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'has_standard_neg', 'has_nonstandard_neg']
            elif features == 'corr_features_with_length_negation':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'has_standard_neg', 'has_nonstandard_neg']
            elif features == 'all_features_with_length_vocab':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE']
            elif features == 'corr_features_with_length_vocab':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE']
            elif features == 'all_features_with_length_vocab_negation':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE', 'has_standard_neg', 'has_nonstandard_neg']
            elif features == 'corr_features_with_length_vocab_negation':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE', 'has_standard_neg', 'has_nonstandard_neg']
            elif features == 'all_features_with_length_pluralization':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'standard_plural', 'non_standard_plural']
            elif features == 'corr_features_with_length_pluralization':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'standard_plural', 'non_standard_plural']
            elif features == 'all_features_with_length_vocab_pluralization':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE', 'standard_plural', 'non_standard_plural']
            elif features == 'corr_features_with_length_vocab_pluralization':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE', 'standard_plural', 'non_standard_plural']
            elif features == 'all_features_with_length_vocab_negation_pluralization':   
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE', 'has_standard_neg', 'has_nonstandard_neg', 'standard_plural', 'non_standard_plural']
            elif features == 'corr_features_with_length_vocab_negation_pluralization':
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE', 'has_standard_neg', 'has_nonstandard_neg', 'standard_plural', 'non_standard_plural']
            elif features == 'neg_plur':
                feature_cols = ['has_standard_neg', 'has_nonstandard_neg', 'standard_plural', 'non_standard_plural']
            elif features == 'len_vocab_neg_plur':
                feature_cols = ['TWEET_NUM_WORDS', 'VOCAB_SIZE', 'has_standard_neg', 'has_nonstandard_neg', 'standard_plural', 'non_standard_plural']
            elif features == 'only_text':
                print('Only text')
            else:
                raise ValueError("Invalid value for 'features' variable")
            
            print('features_macro', features_macro)
            print('features', features)
            print('with_features', with_features)
            print('classes', classes)
            print('train path', TRAIN_FINAL_SES_ERRORS_PATH)


            training_args = TrainingArguments(

                # initial setting
                #output_dir="./results_normal_training_feat_corr", 
                #learning_rate=2e-5, per_device_train_batch_size=16, 
                #per_device_eval_batch_size=16, num_train_epochs=3, 
                #evaluation_strategy="epoch", 
                #save_strategy="epoch", 
                #logging_dir="./logs"

                output_dir=CHECKPOINTS_DIR,

                # core training
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=5,#10

                # evaluation and saving
                eval_strategy="epoch",
                save_strategy="epoch",

                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,

                # logging
                logging_dir="./logs",
                #logging_steps=50,

                # regularization
                weight_decay=0.01,

                # memory / safety
                save_total_limit=2,

                # reproducibility
                seed=seeds[i]
            )

            training_args = TrainingArguments(

                # initial setting
                #output_dir="./results_normal_training_feat_corr", 
                #learning_rate=2e-5, per_device_train_batch_size=16, 
                #per_device_eval_batch_size=16, num_train_epochs=3, 
                #evaluation_strategy="epoch", 
                #save_strategy="epoch", 
                #logging_dir="./logs"

                output_dir=CHECKPOINTS_DIR,

                # core training
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=5,#10

                # evaluation and saving
                eval_strategy="epoch",
                save_strategy="epoch",

                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,

                # logging
                logging_dir="./logs",
                #logging_steps=50,

                # regularization
                weight_decay=0.01,

                # memory / safety
                save_total_limit=2,

                # reproducibility
                seed=seeds[i]
            )

            test_args = TrainingArguments(
                output_dir=CHECKPOINTS_DIR,
                per_device_eval_batch_size=16,
                eval_strategy="no",
                save_strategy="no",
                load_best_model_at_end=False,
                seed=seeds[i]
            )

            print('with_features', with_features)

            # 1) read, prepare and tokenize data
            train_dataset, dev_dataset, tokenizer = prepare_data(with_features, feature_cols, TRAIN_FINAL_SES_ERRORS_PATH, DEV_FINAL_SES_ERRORS_PATH)

            # 2) TRAIN
            train(with_features, training_args, train_dataset, dev_dataset, tokenizer, SAVE_MODEL_DIR)
            
            # 3) TEST
            test(with_features, test_args, feature_cols, TEST_FINAL_SES_ERRORS_PATH, SAVE_MODEL_DIR, SAVE_PREDICTIONS_TWEETS_PATH, SAVE_PREDICTIONS_USERS_PATH, SAVE_PREDICTIONS_USERS_MAX_PATH)

            # 4) EVALUATION
            # with single tweets
            f1_tweets, pred_path_tweets = evaluation(class_num, col_pred_tweet, TEST_FINAL_SES_ERRORS_PATH, PRED_PATH_TWEETS)
            print('\n\n')
            # with user avg
            f1_users_avg, pred_path_users_avg = evaluation(class_num, col_pred_user, TEST_FINAL_SES_ERRORS_PATH, PRED_PATH_USERS)
            print('\n')
            # with user max
            f1_users_max, pred_path_users_max = evaluation(class_num, col_pred_user, TEST_FINAL_SES_ERRORS_PATH, PRED_PATH_USERS_MAX)

            df_eval = pd.concat([df_eval, pd.DataFrame({'run_num': [i+1, i+1, i+1], 'F1_score': [f1_tweets, f1_users_avg, f1_users_max], 'Aggregation_type': ['Single tweets', 'User avg', 'User max'], 'Prediction_path': [pred_path_tweets, pred_path_users_avg, pred_path_users_max]})]) 
        
            df_eval.to_csv(OUTPUT_PATH_EVAL, index=False)
            


            