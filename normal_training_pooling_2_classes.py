import pandas as pd
from collections import defaultdict
import numpy as np
import random
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os


MAX_TWEETS = 30

class CamembertMeanPoolingFeatures(nn.Module):

    def __init__(self, n_features, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained("camembert-base")
        hidden = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.1)

        self.layer_norm = nn.LayerNorm(hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden + n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, input_ids, attention_mask, features, labels=None):

        B, T, L = input_ids.shape

        # build tweet mask BEFORE reshape
        tweet_mask = (attention_mask.sum(dim=2) > 0).float()

        # flatten
        input_ids = input_ids.reshape(B*T, L)
        attention_mask = attention_mask.reshape(B*T, L)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        
        # cls embeddings
        #cls = outputs.last_hidden_state[:, 0]
        # restore shape
        #cls = cls.reshape(B, T, -1)
        #cls = self.dropout(cls)

        # mean token pooling
        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)

        tweet_emb = (
            (token_embeddings * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1e-9)
        )

        # restore user structure
        tweet_emb = tweet_emb.reshape(B, T, -1)

        tweet_emb = self.dropout(tweet_emb)

        # masked mean pooling
        tweet_mask = tweet_mask.unsqueeze(-1)

        user_emb = (
            (tweet_emb * tweet_mask).sum(dim=1)
            / tweet_mask.sum(dim=1).clamp(min=1e-9)
        )

        user_emb = self.layer_norm(user_emb)

        combined = torch.cat([user_emb, features.float()], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
    
    '''def forward_old(self, input_ids, attention_mask, features, labels=None):
        
        B, T, L = input_ids.shape

        input_ids = input_ids.view(B*T, L)
        attention_mask = attention_mask.view(B*T, L)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]

        cls = cls.view(B, T, -1)

        # ✅ MEAN POOLING
        user_emb = cls.mean(dim=1)
        #mask = (attention_mask.sum(dim=2) > 0).float()  # (B, T)
        #mask = mask.unsqueeze(-1)  # (B, T, 1)
        user_emb = (cls * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


        combined = torch.cat([user_emb, features], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)'''

class CamembertMeanPooling(nn.Module):

    def __init__(self, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained("camembert-base")

        hidden = self.bert.config.hidden_size

        # freeze camembert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.1)

        self.layer_norm = nn.LayerNorm(hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, input_ids, attention_mask, labels=None):

        B, T, L = input_ids.shape

        # build tweet mask BEFORE reshape
        tweet_mask = (attention_mask.sum(dim=2) > 0).float()

        # flatten
        input_ids = input_ids.reshape(B*T, L)
        attention_mask = attention_mask.reshape(B*T, L)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # cls embeddings
        #cls = outputs.last_hidden_state[:, 0]
        # restore shape
        #cls = cls.reshape(B, T, -1)
        #cls = self.dropout(cls)
        
        # mean token pooling
        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)

        tweet_emb = (
            (token_embeddings * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1e-9)
        )

        # restore user structure
        tweet_emb = tweet_emb.reshape(B, T, -1)

        tweet_emb = self.dropout(tweet_emb)
        
        # masked mean pooling
        tweet_mask = tweet_mask.unsqueeze(-1)

        user_emb = (
            (tweet_emb * tweet_mask).sum(dim=1)
            / tweet_mask.sum(dim=1).clamp(min=1e-9)
        )

        user_emb = self.layer_norm(user_emb)

        logits = self.classifier(user_emb)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
    

class CamembertAttentionPoolingFeatures(nn.Module):
    def __init__(self, n_features, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained("camembert-base")
        hidden = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.layer_norm = nn.LayerNorm(hidden)

        # freeze camembert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # attention scorer
        self.attention = nn.Linear(hidden, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden + n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, features, labels=None):

        B, T, L = input_ids.shape

        # build tweet mask BEFORE reshape
        tweet_mask = (attention_mask.sum(dim=2) > 0).float()

        # flatten
        input_ids = input_ids.reshape(B*T, L)
        attention_mask = attention_mask.reshape(B*T, L)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        #cls = outputs.last_hidden_state[:, 0]
        # restore shape
        #cls = cls.reshape(B, T, -1)
        #cls = self.dropout(cls)

        # mean token pooling
        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)

        tweet_emb = (
            (token_embeddings * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1e-9)
        )

        # restore user structure
        tweet_emb = tweet_emb.reshape(B, T, -1)

        tweet_emb = self.dropout(tweet_emb)

        # attention scores
        scores = self.attention(tweet_emb).squeeze(-1)

        # mask padded tweets
        scores = scores.masked_fill(tweet_mask == 0, -1e9)
        
        #tweet_mask = (attention_mask.sum(dim=2) > 0).float()
        # normalize
        weights = torch.softmax(scores, dim=1)
        # expand dims
        weights = weights.unsqueeze(-1)

        # weighted pooling
        user_emb = (tweet_emb * weights).sum(dim=1)

        user_emb = self.layer_norm(user_emb)

        # concatenate features
        combined = torch.cat([user_emb, features.float()], dim=1)

        logits = self.classifier(combined)

        loss = None

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
    
    
    '''def forward_old(self, input_ids, attention_mask, features, labels=None):
        B, T, L = input_ids.shape

        input_ids = input_ids.view(B*T, L)
        attention_mask = attention_mask.view(B*T, L)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]

        cls = cls.view(B, T, -1)

        # 🔥 ATTENTION
        scores = self.attention(cls)          # (B, T, 1)
        weights = torch.softmax(scores, dim=1)

        user_emb = (cls * weights).sum(dim=1)

        combined = torch.cat([user_emb, features], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
    '''

class CamembertAttentionPooling(nn.Module):
    def __init__(self, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained("camembert-base")
        hidden = self.bert.config.hidden_size

        # freeze camembert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.1)

        self.layer_norm = nn.LayerNorm(hidden)

        # attention scorer
        self.attention = nn.Linear(hidden, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, labels=None):

        B, T, L = input_ids.shape

        # build tweet mask BEFORE reshape
        tweet_mask = (attention_mask.sum(dim=2) > 0).float()

        # flatten
        input_ids = input_ids.reshape(B*T, L)
        attention_mask = attention_mask.reshape(B*T, L)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        #cls = outputs.last_hidden_state[:, 0]
        # restore shape
        #cls = cls.reshape(B, T, -1)
        #cls = self.dropout(cls)

        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)

        tweet_emb = (
            (token_embeddings * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1e-9)
        )

        # restore user structure
        tweet_emb = tweet_emb.reshape(B, T, -1)

        tweet_emb = self.dropout(tweet_emb)

        # attention scores
        scores = self.attention(tweet_emb).squeeze(-1)

        # mask padded tweets
        scores = scores.masked_fill(tweet_mask == 0, -1e9)
        
        #tweet_mask = (attention_mask.sum(dim=2) > 0).float()
        # normalize
        weights = torch.softmax(scores, dim=1)
        # expand dims
        weights = weights.unsqueeze(-1)

        # weighted pooling
        user_emb = (tweet_emb * weights).sum(dim=1)

        user_emb = self.layer_norm(user_emb)

        logits = self.classifier(user_emb)

        loss = None

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
    

'''class CamembertWithFeaturesOrdinal(nn.Module):
    """
    CORAL-style ordinal regression:
    K classes → K-1 binary ordinal outputs
    """

    def __init__(self, n_features, num_classes=3):
        super().__init__()

        self.num_classes = num_classes
        self.bert = AutoModel.from_pretrained("camembert-base")

        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes - 1)  # IMPORTANT
        )

    def forward(self, input_ids, attention_mask, features, labels=None):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]
        combined = torch.cat([cls, features.float()], dim=1)

        logits = self.classifier(combined)

        loss = None

        if labels is not None:
            # CORAL loss: K-1 binary tasks
            labels = labels.long()

            # convert to ordinal targets
            ordinal_targets = torch.stack([
                (labels > i).float()
                for i in range(self.num_classes - 1)
            ], dim=1)

            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, ordinal_targets)

        return SequenceClassifierOutput(loss=loss, logits=logits)'''
    

'''def build_user_dataset_feat(df, feature_cols, MAX_TWEETS=5):
    
    #users = defaultdict(lambda: {"texts": [], "label": None})

    for _, row in df.iterrows():
        uid = row["user_id"]

        #tweets = users[uid]['texts']

        if len(tweets) > MAX_TWEETS:
            tweets = random.sample(tweets, MAX_TWEETS)

        users[uid]["texts"].append(row["text"])
        users[uid]["label"] = row["labels"]

    dataset = []
    for uid, data in users.items():
        print('uid', uid)
        print('data', data)
        dataset.append({
            "texts": data["texts"],
            "labels": data["label"]
        })
        print({
            "texts": data["texts"],
            "labels": data["label"]
        })
        print('\n')

    return dataset

    dataset = []

    # group tweets by user
    grouped = df.groupby("user_id")

    for uid, group in grouped:

        # random sample of tweets
        sampled = group.sample(
            n=min(MAX_TWEETS, len(group)),
            random_state=42
        )

        texts = sampled["text"].tolist()

        # aggregate handcrafted features
        features = (
            sampled[feature_cols]
            .mean(axis=0)
            .values
            .astype(np.float32)
        )

        # user label
        label = sampled["labels"].iloc[0]

        dataset.append({
            "texts": texts,
            "features": features,
            "labels": label
        })

    return dataset'''


def build_user_dataset_feat(df, feature_cols, MAX_TWEETS=MAX_TWEETS):

    dataset = []

    grouped = df.groupby("user_id")

    for uid, group in grouped:

        # deterministic shuffle
        #group = group.sample(frac=1, random_state=42)
        #sampled = group.head(MAX_TWEETS)
        
        # random sampling every call
        sampled = group.sample(
            n=min(MAX_TWEETS, len(group)),
            replace=False
        )

        texts = sampled["text"].tolist()

        # feature aggregation
        feat_mean = sampled[feature_cols].mean(axis=0)
        feat_std  = sampled[feature_cols].std(axis=0).fillna(0)

        features = np.concatenate([
            feat_mean.values,
            feat_std.values,
            np.array([np.log1p(len(group))])
        ]).astype(np.float32)

        label = sampled["labels"].iloc[0]

        dataset.append({
            "user_id": uid,
            "texts": texts,
            "features": features,
            "labels": label
        })

    return dataset

def build_user_dataset_no_feat(df, MAX_TWEETS=MAX_TWEETS):

    dataset = []

    grouped = df.groupby("user_id")

    for uid, group in grouped:

        # deterministic shuffle
        #group = group.sample(frac=1, random_state=42)
        # keep at most MAX_TWEETS
        #sampled = group.head(MAX_TWEETS)

        # random sampling every call
        sampled = group.sample(
            n=min(MAX_TWEETS, len(group)),
            replace=False
        )

        texts = sampled["text"].tolist()

        label = sampled["labels"].iloc[0]

        dataset.append({
            "user_id": uid,
            "texts": texts,
            "labels": label
        })

    return dataset

'''def build_user_dataset_feat(df, MAX_TWEETS=5):'''
'''users = defaultdict(lambda: {"texts": [], "features": [], "label": None})

for _, row in df.iterrows():
    uid = row["user_id"]

    tweets = users[uid]['texts']

    if len(tweets) > MAX_TWEETS:
        tweets = random.sample(tweets, MAX_TWEETS)

    users[uid]["texts"].append(row["text"])
    users[uid]["features"].append([row[col] for col in feature_cols])
    users[uid]["label"] = row["labels"]

dataset = []
for uid, data in users.items():
    print('uid', uid)
    print('data', data)
    dataset.append({
        "texts": data["texts"],
        "features": np.mean(data["features"], axis=0),  # aggregate features
        "labels": data["label"]
    })
    print({
        "texts": data["texts"],
        "features": np.mean(data["features"], axis=0),  # aggregate features
        "labels": data["label"]
    })
    print('\n')

    print(dataset)

return dataset

def build_user_dataset_feat(df, feature_cols, MAX_TWEETS=5):'''

'''dataset = []

# group all tweets by user
grouped = df.groupby("user_id")

for uid, group in grouped:

    # random sample of tweets per user
    sampled = group.sample(
        n=min(MAX_TWEETS, len(group)),
        random_state=42
    )

    texts = sampled["text"].tolist()

    # all tweets of a user have same label
    label = sampled["labels"].iloc[0]

    dataset.append({
        "texts": texts,
        "labels": label
    })

return dataset'''

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average='macro')
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


# 1) PREPARE DATA

def prepare_data(with_features, feature_cols, NORM_FEATURES, TRAIN_FINAL_SES_ERRORS_PATH, DEV_FINAL_SES_ERRORS_PATH):

    # 1) prepare dataset

    print('Reading files')

    train_df = pd.read_csv(TRAIN_FINAL_SES_ERRORS_PATH)#, sep='\t'
    dev_df = pd.read_csv(DEV_FINAL_SES_ERRORS_PATH)#, sep='\t'


    if with_features == False:
        # keep only useful columns
        train_df = train_df[['tweet_id', 'user_id', 'text', 'SES_users']]
        dev_df = dev_df[['tweet_id', 'user_id', 'text', 'SES_users']]
    
    # rename column ses labels for right format for training
    train_df = train_df.rename(columns={'SES_users': 'labels'})
    dev_df = dev_df.rename(columns={'SES_users': 'labels'})

    # keep only low and high class
    train_df = train_df[train_df['labels'].isin(['lower', 'high'])]
    dev_df = dev_df[dev_df['labels'].isin(['lower', 'high'])]

    label_map = {"lower": 0, "high": 1}

    train_df["labels"] = train_df["labels"].map(label_map)
    dev_df["labels"]   = dev_df["labels"].map(label_map)

    
    if with_features == True and scaling == True:

        print("Standardizing features")

        # normalize error features
        # log of vocabulary size
        for df in [train_df, dev_df]:

            # avoid division by zero
            #length = df["TWEET_NUM_WORDS"].clip(lower=1)

            for col in NORM_FEATURES:
                df[col] = df[col] / max(df['TWEET_NUM_WORDS'], 1)

            df['VOCAB_SIZE'] = np.log1p(df['VOCAB_SIZE'])

        # extract feature matrix
        X_train = train_df[feature_cols].values
        X_dev   = dev_df[feature_cols].values
        #X_test  = test_df[feature_cols].values

        # fit ONLY on train
        scaler = StandardScaler()
        scaler.fit(X_train)

        # transform all
        train_df[feature_cols] = scaler.transform(X_train)
        dev_df[feature_cols]   = scaler.transform(X_dev)
        #test_df[feature_cols]  = scaler.transform(X_test)

        # save scaler
        joblib.dump(scaler, f"{SAVE_MODEL_DIR}/scaler.pkl")
    
    if with_features == True:
        print('Building user dataset with features')
        train_users = build_user_dataset_feat(train_df, feature_cols)
        dev_users   = build_user_dataset_feat(dev_df, feature_cols)
    else:
        train_users = build_user_dataset_no_feat(train_df)
        dev_users = build_user_dataset_no_feat(dev_df)


    # 2) tokenization

    print('Preparing data')

    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    def tokenize_user(example):
        
        enc = tokenizer(
            example["texts"],
            padding="max_length",
            truncation=True,
            max_length=64 # previous parameter -> 128
        )

        result = {
            "input_ids": enc["input_ids"],              # (T, L)
            "attention_mask": enc["attention_mask"],    # (T, L)
            "labels": example["labels"]
        }

        if'features' in example:
            result['features'] = example['features']
        
        return result
        
    train_dataset = Dataset.from_list(train_users)
    dev_dataset   = Dataset.from_list(dev_users)

    train_dataset = train_dataset.map(tokenize_user)
    dev_dataset   = dev_dataset.map(tokenize_user)

    print('train df', train_dataset)

    return train_dataset, dev_dataset, tokenizer


def collate_fn(batch):
    max_tweets = max(len(x["input_ids"]) for x in batch)
    seq_len = len(batch[0]["input_ids"][0])

    input_ids, attention_masks = [], []

    for x in batch:
        n = len(x["input_ids"])

        pad_tweets = max_tweets - n

        input_ids.append(
            x["input_ids"] + [[0]*seq_len]*pad_tweets
        )
        attention_masks.append(
            x["attention_mask"] + [[0]*seq_len]*pad_tweets
        )

    batch_dict = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    }

    if "features" in batch[0]:
        batch_dict["features"] = torch.tensor(
            [x["features"] for x in batch],
            dtype=torch.float
        )

    return batch_dict


# 1) TRAIN

def train(with_features, avg_type, n_features, training_args, train_dataset, dev_dataset, tokenizer, SAVE_MODEL_DIR):
    #model = CamembertMeanPooling(n_features=len(feature_cols))
    # or:

    if with_features == True and avg_type == 'mean':
        model = CamembertMeanPoolingFeatures(n_features)#=len(feature_cols)
    elif with_features == False and avg_type == 'mean':
        model = CamembertMeanPooling()
    elif with_features == True and avg_type == 'attention':
        model = CamembertAttentionPoolingFeatures(n_features)#=len(feature_cols)
    elif with_features == False and avg_type == 'attention':
        model = CamembertAttentionPooling()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    print('TRAIN: Training the model')
    trainer.train()

    try:
        # save model weights
        torch.save(model.state_dict(), f"{SAVE_MODEL_DIR}/model.pt")
        # save tokenizer
        tokenizer.save_pretrained(SAVE_MODEL_DIR)

        # save config
        config = {
            "with_features": with_features,
            "avg_type": avg_type
        }

        if with_features:
            config["n_features"] = n_features #len(feature_cols)

        with open(f"{SAVE_MODEL_DIR}/config.json", "w") as f:
            json.dump(config, f)

    except Exception as e:
        print('Not able to save model:', e)


# 2) TEST

def test(with_features, avg_type, feature_cols, n_features, NORM_FEATURES, training_args, TEST_FINAL_SES_ERRORS_PATH, SAVE_MODEL_DIR, SAVE_PREDICTIONS_PATH, num_tweets=30):

    test_df = pd.read_csv(TEST_FINAL_SES_ERRORS_PATH)

    test_df = (
    test_df.groupby('user_id', group_keys=False)
    .apply(lambda x: x.sample(n=num_tweets, random_state=42))
    .reset_index(drop=True)
    )

    print(test_df)

    print(len(test_df[test_df['SES_users'] == 'lower']))
    print(len(test_df[test_df['SES_users'] == 'medium']))
    print(len(test_df[test_df['SES_users'] == 'high']))

    if with_features == False:
        test_df = test_df[['tweet_id', 'user_id', 'text', 'SES_users']]

    test_df = test_df.rename(columns={'SES_users': 'labels'})
    test_df = test_df[test_df['labels'].isin(['lower', 'high'])]

    label_map = {"lower": 0, "high": 1}
    test_df["labels"]  = test_df["labels"].map(label_map)

    test_df_grouped = (
    test_df.groupby('user_id')
    .first()
    .reset_index()
    )

    print(test_df_grouped)


    # reload config
    with open(f"{SAVE_MODEL_DIR}/config.json") as f:
        config = json.load(f)

    # rebuild model architecture
    if with_features == True and avg_type == 'mean':
        model = CamembertMeanPoolingFeatures(n_features)#=len(feature_cols)
    elif with_features == False and avg_type == 'mean':
        model = CamembertMeanPooling()
    elif with_features == True and avg_type == 'attention':
        model = CamembertAttentionPoolingFeatures(n_features)#=len(feature_cols)
    elif with_features == False and avg_type == 'attention':
        model = CamembertAttentionPooling()

    # load weights
    model.load_state_dict(torch.load(f"{SAVE_MODEL_DIR}/model.pt"))

    if with_features == True and scaling == True:

        # normalize error features
        # log of vocabulary size
        # avoid division by zero
        #length = df["TWEET_NUM_WORDS"].clip(lower=1)
        for col in NORM_FEATURES:
            test_df[col] = test_df[col] / max(test_df['TWEET_NUM_WORDS'], 1)
        
        test_df['VOCAB_SIZE'] = np.log1p(test_df['VOCAB_SIZE'])

        # load scaler
        scaler = joblib.load(f"{SAVE_MODEL_DIR}/scaler.pkl")
        # apply loader
        X_test = test_df[feature_cols].values
        test_df[feature_cols] = scaler.transform(X_test)

    model.eval()
    
    if with_features == True:
        test_users = build_user_dataset_feat(test_df, feature_cols)
    else:
        test_users = build_user_dataset_no_feat(test_df)

    test_dataset = Dataset.from_list(test_users)

    print(test_df)
    print(test_dataset)

    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    def tokenize_user(example):
        
        enc = tokenizer(
            example["texts"],
            padding="max_length",
            truncation=True,
            max_length=64
        )

        result = {
            "user_id": example["user_id"],
            "input_ids": enc["input_ids"],              # (T, L)
            "attention_mask": enc["attention_mask"],    # (T, L)
            "labels": example["labels"]
        }

        if 'features' in example:
            result['features'] = example['features']
        
        return result

    test_dataset = test_dataset.map(tokenize_user)

    print(test_dataset)

    if with_features == True:
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
    preds = logits.argmax(axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)

    test_df_grouped["label"] = preds
    test_df_grouped["prob_0"] = probs[:, 0]
    test_df_grouped["prob_1"] = probs[:, 1]

    print('test df', test_df_grouped)
    
    test_evaluation = trainer.evaluate(test_dataset)
    print(test_evaluation)

    test_df_grouped[['user_id', 'label', 'prob_0', 'prob_1']].to_csv(SAVE_PREDICTIONS_PATH, index=False)


# 4) EVALUATION

def evaluation(class_num, col_pred, TEST_PATH, PRED_PATH):

    # set prediction and true labels vectors
    y_true = pd.read_csv(TEST_PATH)#, sep='\t'
    y_pred = pd.read_csv(PRED_PATH)

    # merge true and pred to have labels on each tweet
    y_tot = pd.merge(y_true, y_pred, on='user_id')

    print(y_true)
    # set columns predictions df
    #y_pred.columns = COLUMNS_TOT_ID_TEXT

    if col_pred == 'user_id':
        # keep only one example per user since we check the user's label
        #y_true = y_true.drop_duplicates(col_pred, keep='first')
        # change labels
        if class_num == 2:
            label_map = {"lower": 0, "high": 1}
        elif class_num == 3:
            label_map = {"lower": 0, "medium": 1, "high": 2}
        y_true["SES_users"] = y_true["SES_users"].map(label_map)

    else:
        # keep only one example per user since we check the user's label
        #y_true = y_true.drop_duplicates('user_id', keep='first')
        # change labels
        if class_num == 2:
            label_map = {"lower": 0, "high": 1}
        elif class_num == 3:
            label_map = {"lower": 0, "medium": 1, "high": 2}
        y_true["SES_users"] = y_true["SES_users"].map(label_map)
        

    print('true', y_true)
    print('pred', y_pred)

    y_pred = y_pred.rename(columns={'label': 'SES_users'})
    print(y_pred['SES_users'].unique)

    y_tot = pd.merge(y_true, y_pred, on=col_pred, suffixes=('_true', '_pred'))
    y_tot = y_tot[['SES_users_true', 'SES_users_pred', col_pred, 'tweet_id']]

    print(y_tot)

    # sort by tweet_id
    '''y_true = y_true.sort_values(by='tweet_id')
    y_pred = y_pred.sort_values(by='tweet_id')'''

    # compute F1 score
    if class_num == 2:
        f1_score_num = f1_score(y_tot['SES_users_true'], y_tot['SES_users_pred'], average='macro')
    elif class_num == 3:
        f1_score_num = f1_score(y_tot['SES_users_true'], y_tot['SES_users_pred'], average='macro')

    return f1_score_num, PRED_PATH

if __name__ == "__main__":

    num_runs = 3

    seeds = [150, 300, 999]

    NUM_TWEETS_LIST = [30, 15, 5]#

    df_eval = pd.DataFrame()

    # variable of num classes
    classes = '2_2' # 2 | 2_2 | 3
    if classes == '2' or classes == '2_2':
        class_num = 2 # 2 | 3
    else:
        class_num = 3
        
    # avg typ -> mean | attention
    avg_type = 'attention'

    masked_location = True

    features_macro = 'only_text' # all_features | corr_features | only_text | neg_plur | len_vocab_neg_plur

    tweets_selection = '' # random -> '' | chosen -> '_chosen'
    
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
    else:
        if classes == '2_2':
            TRAIN_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv'#tweets_TRAIN_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classestrain_paris_LARGE_LARGE_tot_errors_with_medium_2_classes
            DEV_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/test_dev_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv'#tweets_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT
            TEST_FINAL_SES_ERRORS_PATH = 'dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv'#tweets_TEST_TEST_users_PARIS_LARGE_LARGE_no_spammers_with_middle_30_tweets_CHOSEN_CORRECT_2_classes
        else:
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
            # with chosen tweets
            #NORM_FEATURES = ['AGREEMENT', 'CASING', 'CAT_ELISION', 'CAT_GRAMMAIRE', 'CAT_HOMONYMES_PARONYMES', 'CAT_MAJUSCULES', 'CAT_PLEONASMES', 'CAT_REGIONALISMES', 'CAT_REGLES_DE_BASE', 'CAT_TOURS_CRITIQUES', 'CAT_TYPOGRAPHIE', 'MISC', 'MULTITOKEN_SPELLING', 'PONCTUATION_POINT', 'PONCTUATION_VIRGULE', 'PUNCTUATION', 'REPETITIONS_STYLE', 'SEMANTICS', 'STYLE', 'TYPOGRAPHY', 'TYPOS']#, 'CAT_MARQUES_DE_COMMERCE'
        elif features_macro == 'corr_features':
            #NORM_FEATURES = ['CAT_GRAMMAIRE', 'TYPOS', 'CAT_HOMONYMES_PARONYMES', 'AGREEMENT', 'PONCTUATION_VIRGULE', 'CAT_ELISION', 'CAT_TYPOGRAPHIE', 'CAT_TOURS_CRITIQUES', 'MISC']
            NORM_FEATURES = ['CAT_GRAMMAIRE', 'TYPOS', 'CAT_HOMONYMES_PARONYMES', 'AGREEMENT', 'PONCTUATION_VIRGULE', 'CAT_ELISION', 'CAT_TYPOGRAPHIE', 'CAT_TOURS_CRITIQUES', 'MISC']
    if features_macro == 'all_features':
        FEATURES = ['all_features_with_length_vocab_negation', 'all_features_with_length_vocab_pluralization', 'all_features_with_length_vocab_negation_pluralization']
        # 'all_features_negation', 'all_features_with_length_negation', 'all_features_with_length_pluralization', 
    elif features_macro == 'corr_features':
        FEATURES = ['corr_features_with_length_vocab_negation', 'corr_features_with_length_vocab_pluralization', 'corr_features_with_length_vocab_negation_pluralization']
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
        
        for i in range(num_runs):

            if num_runs == 1:

                # output paths
                CHECKPOINTS_DIR = './checkpoints_pre_avg_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_' + avg_type
                SAVE_MODEL_DIR = './models_pre_avg_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_' + avg_type
                SAVE_PREDICTIONS_PATH = './predictions_pre_avg_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_' + avg_type + '.csv'
                
                # evaluation paths
                PRED_PATH = 'predictions_pre_avg_ILE_DE_FRANCE/classes_2_2/' + features + scaling_name + '_' + avg_type + '.csv'#predictions_normal_training_features_corr_NEW.csv
                OUTPUT_PATH = 'F1_score_dev_normal_pre_avg_ILE_DE_FRANCE/classes_2/' + features + scaling_name + '_' + avg_type + '_users.csv'#predictions_normal_training_features_corr_NEW.tsv
            
            else:

                # output paths
                
                if masked_location == True:

                    CHECKPOINTS_DIR = f'./checkpoints_pre_avg_v2_ILE_DE_FRANCE_MASKED_LOC/classes_{classes}{tweets_selection}/{features}{scaling_name}_{avg_type}/run_{i+1}'
                    
                    SAVE_MODEL_DIR = f'./models_pre_avg_v2_ILE_DE_FRANCE_MASKED_LOC/classes_{classes}{tweets_selection}/{features}{scaling_name}_{avg_type}/run_{i+1}'
                    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

                
                else:

                    CHECKPOINTS_DIR = f'./checkpoints_pre_avg_v2_ILE_DE_FRANCE/classes_{classes}{tweets_selection}/{features}{scaling_name}_{avg_type}/run_{i+1}'
                    
                    SAVE_MODEL_DIR = f'./models_pre_avg_v2_ILE_DE_FRANCE/classes_{classes}{tweets_selection}/{features}{scaling_name}_{avg_type}/run_{i+1}'
                    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)


            # variables
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

            # variables
            if features == 'only_text':
                N_FEATURES = 0
            else:
                N_FEATURES = 2 * len(feature_cols) + 1

            print('features_macro', features_macro)
            print('features', features)
            print('with_features', with_features)
            print('classes', classes)
            print('train path', TRAIN_FINAL_SES_ERRORS_PATH)

            print("N_FEATURES:", N_FEATURES)
            print("len(feature_cols):", len(feature_cols))
            print("len(NORM_FEATURES):", len(NORM_FEATURES))
            print(feature_cols)

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
                evaluation_strategy="epoch",
                save_strategy="epoch",

                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                greater_is_better=True,

                # logging
                logging_dir="./logs",
                #logging_steps=50,

                # regularization
                weight_decay=0.01,

                # new parameters added
                warmup_ratio=0.1,
                lr_scheduler_type="linear",
                max_grad_norm=1.0,

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



            # 1) read, prepare and tokenize data
            train_dataset, dev_dataset, tokenizer = prepare_data(with_features, feature_cols, NORM_FEATURES, TRAIN_FINAL_SES_ERRORS_PATH, DEV_FINAL_SES_ERRORS_PATH)

            # 2) TRAIN
            train(with_features, avg_type, N_FEATURES, training_args, train_dataset, dev_dataset, tokenizer, SAVE_MODEL_DIR)
            
            for NUM_TWEETS in NUM_TWEETS_LIST:

                # output eval

                if masked_location == True:

                    SAVE_DIR_EVAL = f'F1_score_dev_normal_pre_avg_v2_ILE_DE_FRANCE_MASKED_LOC/classes_{classes}{tweets_selection}_{NUM_TWEETS}'
                    os.makedirs(SAVE_DIR_EVAL, exist_ok=True)
                    OUTPUT_PATH_EVAL = f'{SAVE_DIR_EVAL }/{features}{scaling_name}_{avg_type}.csv'

                    SAVE_PREDICTIONS_DIR = f'./predictions_pre_avg_v2_ILE_DE_FRANCE_MASKED_LOC/classes_{classes}{tweets_selection}_{NUM_TWEETS}/{features}{scaling_name}_{avg_type}/run_{i+1}'
                    os.makedirs(SAVE_PREDICTIONS_DIR, exist_ok=True)
                    SAVE_PREDICTIONS_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_{avg_type}.csv'

                else:

                    SAVE_DIR_EVAL = f'F1_score_dev_normal_pre_avg_v2_ILE_DE_FRANCE/classes_{classes}{tweets_selection}_{NUM_TWEETS}'
                    os.makedirs(SAVE_DIR_EVAL, exist_ok=True)
                    OUTPUT_PATH_EVAL = f'{SAVE_DIR_EVAL }/{features}{scaling_name}_{avg_type}.csv'

                    SAVE_PREDICTIONS_DIR = f'./predictions_pre_avg_v2_ILE_DE_FRANCE/classes_{classes}{tweets_selection}_{NUM_TWEETS}/{features}{scaling_name}_{avg_type}/run_{i+1}'
                    os.makedirs(SAVE_PREDICTIONS_DIR, exist_ok=True)
                    SAVE_PREDICTIONS_PATH = f'{SAVE_PREDICTIONS_DIR}/{features}{scaling_name}_{avg_type}.csv'
                
                # evaluation paths
                # single tweets
                PRED_PATH = SAVE_PREDICTIONS_PATH
                
                # 3) TEST
                test(with_features, avg_type, feature_cols, N_FEATURES, NORM_FEATURES, test_args, TEST_FINAL_SES_ERRORS_PATH, SAVE_MODEL_DIR, SAVE_PREDICTIONS_PATH, NUM_TWEETS)
                
                # 4) EVALUATION
                f1_score_num, pred_path = evaluation(class_num, col_pred_user, TEST_FINAL_SES_ERRORS_PATH, PRED_PATH)

                df_eval = pd.concat([df_eval, pd.DataFrame({'run_num': [i+1], 'F1_score': [f1_score_num], 'avg_type': [avg_type], 'Prediction_path': [pred_path]})])

        df_eval.to_csv(OUTPUT_PATH_EVAL, index=False)