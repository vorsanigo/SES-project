from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer, TrainingArguments
from safetensors.torch import load_file
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os


# ── CHANGE 1: output size 2 → 3 everywhere ───────────────────────────────────
# ── CHANGE 2: label_map 3 classes ────────────────────────────────────────────
# ── CHANGE 3: filter 3 classes ───────────────────────────────────────────────
# ── CHANGE 4: compute_metrics f1_macro instead of f1 binary ──────────────────
# ── CHANGE 5: probs shape (N, 3) everywhere ──────────────────────────────────


# ── MODELS ───────────────────────────────────────────────────────────────────

class CamembertWithFeatures(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.bert = AutoModel.from_pretrained("camembert-base")
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)          # CHANGE 1: 2 → 3
        )

    def forward(self, input_ids, attention_mask, features, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]

        if scaling:
            combined = torch.cat([cls, features.float()], dim=1)
        else:
            combined = torch.cat([cls, features], dim=1)

        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class CamembertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("camembert-base")
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)          # CHANGE 1: 2 → 3
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def add_features(example):
    example["features"] = [example[col] for col in feature_cols]
    return example


# CHANGE 4: f1_macro for 3 classes
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average='macro'),   # CHANGE 4
        "f1_lower": f1_score(labels, preds, average=None, labels=[0, 1, 2])[0],
        "f1_medium": f1_score(labels, preds, average=None, labels=[0, 1, 2])[1],
        "f1_high":  f1_score(labels, preds, average=None, labels=[0, 1, 2])[2],
    }


def compute_f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


# ── PREPARE DATA ──────────────────────────────────────────────────────────────

def prepare_data(with_features, feature_cols,
                 TRAIN_FINAL_SES_ERRORS_PATH, DEV_FINAL_SES_ERRORS_PATH):

    print('Reading files')
    train_df = pd.read_csv(TRAIN_FINAL_SES_ERRORS_PATH)
    dev_df   = pd.read_csv(DEV_FINAL_SES_ERRORS_PATH)

    if not with_features:
        train_df = train_df[['tweet_id', 'user_id', 'text', 'SES_users']]
        dev_df   = dev_df[['tweet_id',   'user_id', 'text', 'SES_users']]

    print(train_df)
    print(dev_df)

    train_df = train_df.rename(columns={'SES_users': 'labels'})
    dev_df   = dev_df.rename(columns={'SES_users':   'labels'})

    # CHANGE 3: keep all 3 classes instead of filtering 2
    train_df = train_df[train_df['labels'].isin(['lower', 'medium', 'high'])]
    dev_df   = dev_df[dev_df['labels'].isin(['lower',   'medium', 'high'])]

    # CHANGE 2: 3-class label map
    label_map = {"lower": 0, "medium": 1, "high": 2}
    train_df["labels"] = train_df["labels"].map(label_map)
    dev_df["labels"]   = dev_df["labels"].map(label_map)

    print(train_df["labels"].value_counts())

    if with_features and scaling:
        print("Standardizing features")

        scaler  = StandardScaler()
        X_train = train_df[feature_cols].values
        X_dev   = dev_df[feature_cols].values

        scaler.fit(X_train)
        train_df[feature_cols] = scaler.transform(X_train)
        dev_df[feature_cols]   = scaler.transform(X_dev)

        joblib.dump(scaler, f"{SAVE_MODEL_DIR}/scaler.pkl")

    print('Preparing data')
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset   = Dataset.from_pandas(dev_df)

    def tokenize(batch, text_column="text"):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    dev_dataset   = dev_dataset.map(tokenize,   batched=True)

    if with_features:
        train_dataset = train_dataset.map(add_features)
        dev_dataset   = dev_dataset.map(add_features)
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "features", "labels"])
        dev_dataset.set_format(type="torch",   columns=["input_ids", "attention_mask", "features", "labels"])
    else:
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        dev_dataset.set_format(type="torch",   columns=["input_ids", "attention_mask", "labels"])

    print(train_dataset)
    print(dev_dataset)

    return train_dataset, dev_dataset, tokenizer


# ── TRAIN ─────────────────────────────────────────────────────────────────────

def train(with_features, training_args, train_dataset, dev_dataset,
          tokenizer, SAVE_MODEL_DIR):

    print('TRAIN: Preparing functions and defining model')

    if with_features:
        n_features = len(feature_cols)
        model = CamembertWithFeatures(n_features=n_features)
    else:
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
        torch.save(model.state_dict(), f"{SAVE_MODEL_DIR}/model.pt")
        tokenizer.save_pretrained(SAVE_MODEL_DIR)

        config = {"with_features": with_features}
        if with_features:
            config["n_features"] = n_features

        with open(f"{SAVE_MODEL_DIR}/config.json", "w") as f:
            json.dump(config, f)

    except Exception as e:
        print('Not able to save model:', e)


# --- TEST ------------------------------------------------------------------------

def test(with_features, test_args, feature_cols,
         TEST_FINAL_SES_ERRORS_PATH, SAVE_MODEL_DIR,
         SAVE_PREDICTIONS_TWEETS_PATH,
         SAVE_PREDICTIONS_USERS_PATH,
         SAVE_PREDICTIONS_USERS_MAX_PATH):

    test_df = pd.read_csv(TEST_FINAL_SES_ERRORS_PATH)

    if not with_features:
        test_df = test_df[['tweet_id', 'user_id', 'text', 'SES_users']]

    test_df = test_df.rename(columns={'SES_users': 'labels'})

    # CHANGE 3: keep all 3 classes
    test_df = test_df[test_df['labels'].isin(['lower', 'medium', 'high'])]

    # CHANGE 2: 3-class label map
    label_map = {"lower": 0, "medium": 1, "high": 2}
    test_df["labels"] = test_df["labels"].map(label_map)

    with open(f"{SAVE_MODEL_DIR}/config.json") as f:
        config = json.load(f)

    if with_features:
        model = CamembertWithFeatures(n_features=config["n_features"])
    else:
        model = CamembertClassifier()

    model.load_state_dict(torch.load(f"{SAVE_MODEL_DIR}/model.pt"))

    if with_features and scaling:
        scaler = joblib.load(f"{SAVE_MODEL_DIR}/scaler.pkl")
        test_df[feature_cols] = scaler.transform(test_df[feature_cols].values)

    model.eval()

    print('TEST: Preparing test data')
    test_dataset = Dataset.from_pandas(test_df)
    tokenizer    = AutoTokenizer.from_pretrained("camembert-base")

    def tokenize(batch, text_column="text"):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    test_dataset = test_dataset.map(tokenize, batched=True)

    if with_features:
        test_dataset = test_dataset.map(add_features)
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "features"])
    else:
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    print('TEST: Prediction')
    trainer = Trainer(model=model, args=test_args)

    predictions  = trainer.predict(test_dataset)
    logits       = predictions.predictions

    # CHANGE 5: softmax over 3 classes
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()  # (N, 3)
    preds = probs.argmax(axis=1)

    test_df["predicted_label"] = preds
    test_df["prob_0"] = probs[:, 0]   # lower
    test_df["prob_1"] = probs[:, 1]   # medium
    test_df["prob_2"] = probs[:, 2]   # high

    print('test df', test_df)

    # aggregate per user — mean of probabilities per class
    user_scores = test_df.groupby("user_id")[["prob_0", "prob_1", "prob_2"]].mean().reset_index()
    user_scores["final_label"] = user_scores[["prob_0", "prob_1", "prob_2"]].values.argmax(axis=1)
    user_scores["confidence"]  = user_scores[["prob_0", "prob_1", "prob_2"]].max(axis=1)

    # aggregate per user — most frequent predicted label
    user_preds = (
        test_df
        .groupby("user_id")["predicted_label"]
        .agg(
            final_label=lambda x: x.mode()[0],
            n_tweets="count"
        )
        .reset_index()
    )

    print('user scores', user_scores)

    # uncertainty as std of max predicted probability per tweet
    test_df["pred_probs"]    = probs.max(axis=1)
    user_uncertainty = test_df.groupby("user_id")["pred_probs"].std().reset_index(name="uncertainty")

    # save predictions on single tweets
    pd.DataFrame({
        'tweet_id': test_df['tweet_id'],
        'label':    preds,
        'prob_0':   test_df['prob_0'],
        'prob_1':   test_df['prob_1'],
        'prob_2':   test_df['prob_2'],          # CHANGE 5: add prob_2
    }).to_csv(SAVE_PREDICTIONS_TWEETS_PATH, index=False)

    # save predictions aggregated per user (mean probs)
    pd.DataFrame({
        'user_id':        user_scores['user_id'],
        'label':          user_scores['final_label'],
        'mean_prob_0':    user_scores['prob_0'],
        'mean_prob_1':    user_scores['prob_1'],
        'mean_prob_2':    user_scores['prob_2'],    # CHANGE 5
        'confidence':     user_scores['confidence'],
        'uncertainty_std': user_uncertainty['uncertainty']
    }).to_csv(SAVE_PREDICTIONS_USERS_PATH, index=False)

    # save predictions aggregated per user (most frequent label)
    pd.DataFrame({
        'user_id':  user_preds['user_id'],
        'label':    user_preds['final_label'],
        'n_tweets': user_preds['n_tweets']
    }).to_csv(SAVE_PREDICTIONS_USERS_MAX_PATH, index=False)


# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluation(class_num, col_pred, TEST_PATH, PRED_PATH):

    y_true = pd.read_csv(TEST_PATH)
    y_pred = pd.read_csv(PRED_PATH)

    print('y_true initial', y_true)

    # CHANGE 2: always use 3-class label map
    label_map = {"lower": 0, "medium": 1, "high": 2}
    y_true["SES_users"] = y_true["SES_users"].map(label_map)

    # filter to 3 classes only
    y_true = y_true[y_true["SES_users"].isin([0, 1, 2])]

    print('y_true filtered', y_true)
    print('y_pred', y_pred)

    y_pred = y_pred.rename(columns={'label': 'SES_users'})

    y_tot = pd.merge(y_true, y_pred, on=col_pred, suffixes=('_true', '_pred'))
    y_tot = y_tot[['SES_users_true', 'SES_users_pred', col_pred]]

    print('y_tot', y_tot)

    # CHANGE 4: always macro F1 for 3 classes
    f1_score_num = f1_score(
        y_tot['SES_users_true'],
        y_tot['SES_users_pred'],
        average='macro'
    )

    # per-class F1
    f1_per_class = f1_score(
        y_tot['SES_users_true'],
        y_tot['SES_users_pred'],
        average=None,
        labels=[0, 1, 2]
    )
    print(f"F1 macro: {f1_score_num:.4f} | F1 lower: {f1_per_class[0]:.4f} | F1 medium: {f1_per_class[1]:.4f} | F1 high: {f1_per_class[2]:.4f}")

    return f1_score_num, PRED_PATH


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    num_runs = 3
    seeds    = [150, 300, 999]
    df_eval  = pd.DataFrame()

    # variable of num classes
    classes = '3' # 2 | 2_2 | 3
    if classes == '2' or classes == '2_2':
        class_num = 2 # 2 | 3
    else:
        class_num = 3


    features_macro = 'only_text' # all_features | corr_features | only_text | neg_plur | len_vocab_neg_plur
    
    masked_location = True

    region        = 'ILE_DE_FRANCE'
    
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

    col_pred_tweet = 'tweet_id'
    col_pred_user  = 'user_id'

    if masked_location == True:
        TRAIN_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_ner_stanza_train_location_3_classes.csv'#tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classestrain_paris_LARGE_LARGE_tot_errors_with_medium_2_classes
        DEV_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_ner_stanza_test_dev_location_3_classes.csv'#tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT
        TEST_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/tweets_ner_stanza_test_test_location_3_classes.csv'#tweets_TEST_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes
    elif region == 'PARIS':
        TRAIN_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium/TRAIN_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv'
        DEV_FINAL_SES_ERRORS_PATH   = 'dev_phase_normal/input with medium/TEST_DEV_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv'
        TEST_FINAL_SES_ERRORS_PATH  = 'dev_phase_normal/input with medium/TEST_TEST_users_mistakes_PARIS_no_spammers_OK_INPUT_with_middle_length_vocab_size.csv'
    elif region == 'ILE_DE_FRANCE':
        TRAIN_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium.csv'
        DEV_FINAL_SES_ERRORS_PATH   = 'dev_phase_normal/input with medium large/test_dev_paris_LARGE_LARGE_tot_errors_with_medium.csv'
        TEST_FINAL_SES_ERRORS_PATH  = 'dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium.csv'

    NORM_FEATUES = []
    if region == 'PARIS':
        if features_macro == 'all_features':
            NORM_FEATURES = ['AGREEMENT', 'CASING', 'CAT_ELISION', 'CAT_GRAMMAIRE', 'CAT_HOMONYMES_PARONYMES', 'CAT_MAJUSCULES', 'CAT_PLEONASMES', 'CAT_REGIONALISMES', 'CAT_REGLES_DE_BASE', 'CAT_TOURS_CRITIQUES', 'CAT_TYPOGRAPHIE', 'MISC', 'MULTITOKEN_SPELLING', 'PONCTUATION_POINT', 'PONCTUATION_VIRGULE', 'PUNCTUATION', 'REPETITIONS_STYLE', 'SEMANTICS', 'STYLE', 'TYPOGRAPHY', 'TYPOS']
        elif features_macro == 'corr_features':
            NORM_FEATURES = ['CAT_GRAMMAIRE', 'TYPOS', 'CAT_HOMONYMES_PARONYMES', 'AGREEMENT', 'PONCTUATION_VIRGULE', 'CAT_ELISION', 'CAT_TYPOGRAPHIE', 'CAT_TOURS_CRITIQUES', 'MISC']
    elif region == 'ILE_DE_FRANCE':
        if features_macro == 'all_features':
            NORM_FEATURES = ['AGREEMENT', 'CASING', 'CAT_ELISION', 'CAT_GRAMMAIRE', 'CAT_HOMONYMES_PARONYMES', 'CAT_MAJUSCULES', 'CAT_PLEONASMES', 'CAT_REGIONALISMES', 'CAT_REGLES_DE_BASE', 'CAT_TOURS_CRITIQUES', 'CAT_TYPOGRAPHIE', 'MISC', 'MULTITOKEN_SPELLING', 'PONCTUATION_POINT', 'PONCTUATION_VIRGULE', 'PUNCTUATION', 'REPETITIONS_STYLE', 'SEMANTICS', 'STYLE', 'TYPOGRAPHY', 'TYPOS', 'CAT_MARQUES_DE_COMMERCE']
        elif features_macro == 'corr_features':
            NORM_FEATURES = ['CAT_GRAMMAIRE', 'TYPOS', 'CAT_HOMONYMES_PARONYMES', 'AGREEMENT', 'PONCTUATION_VIRGULE', 'CAT_ELISION', 'CAT_TYPOGRAPHIE', 'CAT_TOURS_CRITIQUES', 'MISC']

    if features_macro == 'all_features':
        FEATURES = ['all_features_with_length_vocab_negation', 'all_features_with_length_vocab_pluralization', 'all_features_with_length_vocab_negation_pluralization']
        # 'all_features_negation', 'all_features_with_length_negation', 'all_features_with_length_pluralization', 
    elif features_macro == 'corr_features':
        FEATURES = ['corr_features_with_length_vocab_pluralization'] #'corr_features_with_length_vocab_negation',, 'corr_features_with_length_vocab_negation_pluralization' 
        # 'corr_features_negation', 'corr_features_with_length_negation', 'corr_features_with_length_pluralization',      
    elif features_macro == 'neg_plur':
        FEATURES = ['neg_plur']
    elif features_macro == 'len_vocab_neg_plur':
        FEATURES = ['len_vocab_neg_plur']
    else:
        FEATURES = ['only_text']

    for features in FEATURES:

        df_eval  = pd.DataFrame()
        
        if features == 'only_text':
            with_features = False
        else:
            with_features = True

        if masked_location == True:
            SAVE_DIR_EVAL    = f'F1_score_dev_normal_post_ILE_DE_FRANCE_MASKED_LOC/classes_{classes}'   # CHANGE: classes_3
            os.makedirs(SAVE_DIR_EVAL, exist_ok=True)
            OUTPUT_PATH_EVAL = f'{SAVE_DIR_EVAL}/{features}{scaling_name}.csv'
        else:
            SAVE_DIR_EVAL    = f'F1_score_dev_normal_post_ILE_DE_FRANCE/classes_{classes}'   # CHANGE: classes_3
            os.makedirs(SAVE_DIR_EVAL, exist_ok=True)
            OUTPUT_PATH_EVAL = f'{SAVE_DIR_EVAL}/{features}{scaling_name}.csv'

        for i in range(num_runs):

            if num_runs == 1:
                CHECKPOINTS_DIR              = f'./checkpoints_post_ILE_DE_FRANCE/classes_3/{features}{scaling_name}'
                SAVE_MODEL_DIR               = f'./models_post_ILE_DE_FRANCE/classes_3/{features}{scaling_name}'
                SAVE_PREDICTIONS_TWEETS_PATH = f'./predictions_post_ILE_DE_FRANCE/classes_3/{features}{scaling_name}_tweets.csv'
                SAVE_PREDICTIONS_USERS_PATH  = f'./predictions_post_ILE_DE_FRANCE/classes_3/{features}{scaling_name}_users_mean.csv'
                SAVE_PREDICTIONS_USERS_MAX_PATH = f'./predictions_post_ILE_DE_FRANCE/classes_3/{features}{scaling_name}_users_max.csv'
                PRED_PATH_TWEETS             = SAVE_PREDICTIONS_TWEETS_PATH
                PRED_PATH_USERS              = SAVE_PREDICTIONS_USERS_PATH
                PRED_PATH_USERS_MAX          = SAVE_PREDICTIONS_USERS_MAX_PATH
            else:

                if masked_location == True:

                    CHECKPOINTS_DIR              = f'./checkpoints_post_ILE_DE_FRANCE_MASKED_LOC/classes_3/{features}{scaling_name}/run_{i+1}'
                    SAVE_MODEL_DIR               = f'./models_post_ILE_DE_FRANCE_MASKED_LOC/classes_3/{features}{scaling_name}/run_{i+1}'
                    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

                    SAVE_PREDICTIONS_DIR         = f'./predictions_post_ILE_DE_FRANCE_MASKED_LOC/classes_3/{features}{scaling_name}/run_{i+1}'
                    os.makedirs(SAVE_PREDICTIONS_DIR, exist_ok=True)

                    SAVE_PREDICTIONS_TWEETS_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_tweets.csv'
                    SAVE_PREDICTIONS_USERS_PATH  = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_users_mean.csv'
                    SAVE_PREDICTIONS_USERS_MAX_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_users_max.csv'

                    PRED_PATH_TWEETS             = SAVE_PREDICTIONS_TWEETS_PATH
                    PRED_PATH_USERS              = SAVE_PREDICTIONS_USERS_PATH
                    PRED_PATH_USERS_MAX          = SAVE_PREDICTIONS_USERS_MAX_PATH

                else:

                    CHECKPOINTS_DIR              = f'./checkpoints_post_ILE_DE_FRANCE/classes_3/{features}{scaling_name}/run_{i+1}'
                    SAVE_MODEL_DIR               = f'./models_post_ILE_DE_FRANCE/classes_3/{features}{scaling_name}/run_{i+1}'
                    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

                    SAVE_PREDICTIONS_DIR         = f'./predictions_post_ILE_DE_FRANCE/classes_3/{features}{scaling_name}/run_{i+1}'
                    os.makedirs(SAVE_PREDICTIONS_DIR, exist_ok=True)

                    SAVE_PREDICTIONS_TWEETS_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_tweets.csv'
                    SAVE_PREDICTIONS_USERS_PATH  = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_users_mean.csv'
                    SAVE_PREDICTIONS_USERS_MAX_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_users_max.csv'

                    PRED_PATH_TWEETS             = SAVE_PREDICTIONS_TWEETS_PATH
                    PRED_PATH_USERS              = SAVE_PREDICTIONS_USERS_PATH
                    PRED_PATH_USERS_MAX          = SAVE_PREDICTIONS_USERS_MAX_PATH

            # feature columns
            feature_cols = []
            if features in ('all_features', 'corr_features'):
                feature_cols = list(NORM_FEATURES)
            elif features in ('all_features_negation', 'corr_features_negation'):
                feature_cols = NORM_FEATURES + ['has_standard_neg', 'has_nonstandard_neg']
            elif features in ('all_features_with_length', 'corr_features_with_length'):
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS']
            elif features in ('all_features_with_length_negation', 'corr_features_with_length_negation'):
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'has_standard_neg', 'has_nonstandard_neg']
            elif features in ('all_features_with_length_vocab', 'corr_features_with_length_vocab'):
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE']
            elif features in ('all_features_with_length_vocab_negation', 'corr_features_with_length_vocab_negation'):
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE', 'has_standard_neg', 'has_nonstandard_neg']
            elif features in ('all_features_with_length_pluralization', 'corr_features_with_length_pluralization'):
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'standard_plural', 'non_standard_plural']
            elif features in ('all_features_with_length_vocab_pluralization', 'corr_features_with_length_vocab_pluralization'):
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE', 'standard_plural', 'non_standard_plural']
            elif features in ('all_features_with_length_vocab_negation_pluralization', 'corr_features_with_length_vocab_negation_pluralization'):
                feature_cols = NORM_FEATURES + ['TWEET_NUM_WORDS', 'VOCAB_SIZE', 'has_standard_neg', 'has_nonstandard_neg', 'standard_plural', 'non_standard_plural']
            elif features == 'neg_plur':
                feature_cols = ['has_standard_neg', 'has_nonstandard_neg', 'standard_plural', 'non_standard_plural']
            elif features == 'len_vocab_neg_plur':
                feature_cols = ['TWEET_NUM_WORDS', 'VOCAB_SIZE', 'has_standard_neg', 'has_nonstandard_neg', 'standard_plural', 'non_standard_plural']
            elif features == 'only_text':
                print('Only text')
            else:
                raise ValueError(f"Invalid value for 'features': {features}")
            
            print('features_macro', features_macro)
            print('features', features)
            print('with_features', with_features)
            print('classes', classes)
            print('train path', TRAIN_FINAL_SES_ERRORS_PATH)

            training_args = TrainingArguments(
                output_dir=CHECKPOINTS_DIR,
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=5,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",   # CHANGE 4
                greater_is_better=True,
                logging_dir="./logs",
                weight_decay=0.01,
                save_total_limit=2,
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

            # 1) prepare data
            train_dataset, dev_dataset, tokenizer = prepare_data(
                with_features, feature_cols,
                TRAIN_FINAL_SES_ERRORS_PATH, DEV_FINAL_SES_ERRORS_PATH
            )

            # 2) train
            train(with_features, training_args, train_dataset, dev_dataset,
                  tokenizer, SAVE_MODEL_DIR)

            # 3) test
            test(with_features, test_args, feature_cols,
                 TEST_FINAL_SES_ERRORS_PATH, SAVE_MODEL_DIR,
                 SAVE_PREDICTIONS_TWEETS_PATH,
                 SAVE_PREDICTIONS_USERS_PATH,
                 SAVE_PREDICTIONS_USERS_MAX_PATH)

            # 4) evaluation
            f1_tweets,    pred_path_tweets    = evaluation(class_num, col_pred_tweet, TEST_FINAL_SES_ERRORS_PATH, PRED_PATH_TWEETS)
            f1_users_avg, pred_path_users_avg = evaluation(class_num, col_pred_user,  TEST_FINAL_SES_ERRORS_PATH, PRED_PATH_USERS)
            f1_users_max, pred_path_users_max = evaluation(class_num, col_pred_user,  TEST_FINAL_SES_ERRORS_PATH, PRED_PATH_USERS_MAX)

            df_eval = pd.concat([df_eval, pd.DataFrame({
                'run_num':          [i+1, i+1, i+1],
                'F1_score':         [f1_tweets, f1_users_avg, f1_users_max],
                'Aggregation_type': ['Single tweets', 'User avg', 'User max'],
                'Prediction_path':  [pred_path_tweets, pred_path_users_avg, pred_path_users_max]
            })])

        df_eval.to_csv(OUTPUT_PATH_EVAL, index=False)