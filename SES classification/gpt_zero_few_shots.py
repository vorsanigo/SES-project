"""
Zero-/few-shot socioeconomic status classification of Twitter users with GPT
(OpenAI API).

Mirrors the Llama script, but since GPT is API-only there is no access to the
full next-token logit vector. Instead we constrain the model to emit exactly
one label token and read the per-token `logprobs` the API returns, giving a
probability distribution over the classes — the API analogue of the logit trick.

Requires:  pip install openai
Set your key:  export OPENAI_API_KEY=sk-...
"""

import os
import math
import time
import random
from collections import Counter

import polars as pl
from openai import OpenAI


api_key = os.environ.get("OPENAI_API_KEY") 
organization = 'org-mjd4IFp6ilSsUlr4fnK48T83'




# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "gpt-4o"     # "gpt-4o" | "gpt-4o-mini" | "gpt-4-turbo" | ...

CLASSES     = '3'            # '2' / '2_2' -> low/high, '3' -> low/medium/high

prompt_types = ['simple']# 'complex', 'complex_with_direction', ,'specific_all_features'

for prompt_type in prompt_types:

    PROMPT_TYPE = prompt_type      # simple | complex | complex_with_direction | specific_all_features
    if prompt_type == 'specific_all_features':
        FEATURES = 'all'
    else:
        FEATURES    = "only_text"   # all | corr | only_text
    '''PROMPT_TYPE = 'simple'      # simple | complex | complex_with_direction | specific_all_features
    FEATURES    = "only_text"   # all | corr | only_text'''

    FEW_SHOTS   = False
    N_USERS     = None          # None = all users; int to debug on a subset
    SHUFFLE_LABELS = True
    SEED        = 42

    # rate-limit / robustness
    MAX_RETRIES = 5
    RETRY_WAIT  = 5             # seconds, exponential backoff base
    REQUEST_PAUSE = 0.0        # seconds between calls (raise if you hit rate limits)

    LABELS = ["low", "medium", "high"] if CLASSES == '3' else ["low", "high"]

    random.seed(SEED)
    #client = OpenAI()          # reads OPENAI_API_KEY from env

    # client
    client = OpenAI(
        organization=organization,
        #project='Default project',
        api_key=api_key,
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Input / output paths
    # ─────────────────────────────────────────────────────────────────────────────

    if FEATURES == 'only_text':
        if CLASSES == '2_2':
            INPUT_TEST_PATH = "dev_phase_normal/input with medium large/LLAMA_test_paris_LARGE_LARGE_only_text_with_medium_2_classes.jsonl"
        else:
            INPUT_TEST_PATH = "dev_phase_normal/input with medium large/LLAMA_test_paris_LARGE_LARGE_only_text_with_medium.jsonl"
    elif FEATURES == 'all':
        if CLASSES == '2_2':
            INPUT_TEST_PATH = "dev_phase_normal/input with medium large/LLAMA_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.jsonl"
        else:
            INPUT_TEST_PATH = "dev_phase_normal/input with medium large/LLAMA_test_paris_LARGE_LARGE_tot_errors_with_medium.jsonl"

    print(CLASSES, PROMPT_TYPE, FEATURES, FEW_SHOTS)
    print(INPUT_TEST_PATH)

    zero_few = "few" if FEW_SHOTS else "zero"
    PREDICTIONS_OUTPUT_PATH = (
        f"predictions_gpt/classes_{CLASSES}/gpt_{MODEL_ID}_{zero_few}_"
        f"predictions_{FEATURES}_prompt_{PROMPT_TYPE}_{CLASSES}_classes.csv"
    )
    os.makedirs(os.path.dirname(PREDICTIONS_OUTPUT_PATH), exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # Prompts  (identical to the Llama version)
    # ─────────────────────────────────────────────────────────────────────────────

    if PROMPT_TYPE == 'simple':
        SYSTEM_PROMPT_TEMPLATE = (
            "You are a sociolinguistics expert specializing in French. "
            "You will receive a sample of tweets from a user. "
            "Socioeconomic status reflects income level: "
            "{definitions}. "
            "Analyze the writing style and linguistic errors in the tweets "
            "to infer the user's socioeconomic status. "
            "Answer with exactly one word: {options}."
        )
    elif PROMPT_TYPE == 'complex':
        SYSTEM_PROMPT_TEMPLATE = (
            "You are a sociolinguistics expert specializing in French. "
            "You will receive a sample of tweets from a user. "
            "Socioeconomic status reflects income level: "
            "{definitions}. "
            "Analyze the writing style and linguistic errors in the tweets "
            "to infer the user's socioeconomic status. "
            "In particular, focus on grammatical errors, typos, "
            "homonym and paronym confusion, agreement errors, punctuation, "
            "and the standard or non-standard use of negation and pluralization. "
            "Answer with exactly one word: {options}."
        )
    elif PROMPT_TYPE == 'complex_with_direction':
        SYSTEM_PROMPT_TEMPLATE = (
            "You are a sociolinguistics expert specializing in French. "
            "You will receive a sample of tweets from a user. "
            "Socioeconomic status reflects income level: "
            "{definitions}. "
            "Analyze the writing style and linguistic errors in the tweets "
            "to infer the user's socioeconomic status. "
            "In particular, focus on grammatical errors, typos, "
            "homonym and paronym confusion, agreement errors, punctuation, "
            "and the standard or non-standard use of negation and pluralization. "
            "A higher frequency of such errors and of non-standard forms "
            "tends to indicate a lower socioeconomic status, "
            "while consistently correct and standard language tends to indicate a higher one. "
            "Answer with exactly one word: {options}."
        )
    elif PROMPT_TYPE == 'specific_all_features':
        SYSTEM_PROMPT_TEMPLATE = (
            "You are a sociolinguistics expert specializing in French. "
            "You will receive the linguistic profile of a user together with a sample of their tweets. "
            "Socioeconomic status reflects income level: "
            "{definitions}. "
            "Use both the quantified error statistics and the writing style of the "
            "tweets to infer the user's socioeconomic status. "
            "Answer with exactly one word: {options}."
        )

    LABEL_DEFINITIONS = {
        "low":    "low = lower income",
        "medium": "medium = middle income",
        "high":   "high = upper income",
    }

    FEW_SHOT_EXAMPLES = """### Example 1
    1. jai pa eu le tps de manger lol
    2. chuis trop fatigué jsp
    3. c pas faux jcrois tjs
    Answer: low

    ### Example 2
    1. La conférence d'hier était particulièrement instructive
    2. Je recommande vivement cette lecture à quiconque s'intéresse à l'économie
    3. Un excellent article sur les enjeux climatiques dans Les Echos
    Answer: high

    """


    def build_messages(user_row, few_shots, prompt_labels):
        definitions = ", ".join(LABEL_DEFINITIONS[l] for l in prompt_labels)
        options     = ", ".join(prompt_labels[:-1]) + f", or {prompt_labels[-1]}"
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(definitions=definitions, options=options)

        examples   = FEW_SHOT_EXAMPLES if few_shots else ""
        tweet_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(user_row["text"]))

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"{examples}{tweet_list}\n\nAnswer:"},
        ]


    def build_messages_with_features(user_row, few_shots, prompt_labels):
        definitions = ", ".join(LABEL_DEFINITIONS[l] for l in prompt_labels)
        options     = ", ".join(prompt_labels[:-1]) + f", or {prompt_labels[-1]}"
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(definitions=definitions, options=options)

        examples = FEW_SHOT_EXAMPLES if few_shots else ""
        profile  = user_row["llm_prompt"]
        tweets   = "\n".join(f"{i+1}. {t}" for i, t in enumerate(user_row["text"]))
        content  = f"{examples}{profile}\n\nSample tweets:\n{tweets}\n\nAnswer:"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ]


    # ─────────────────────────────────────────────────────────────────────────────
    # API call with logprobs → probability over the label tokens
    # ─────────────────────────────────────────────────────────────────────────────

    def classify_user(messages):
        """
        Returns (pred_label, {label: prob}) using the top logprobs of the first
        generated token. We request several top logprobs and keep the mass that
        lands on our label tokens, renormalising over them.
        """
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages,
                    max_tokens=1,            # one label token
                    temperature=0,           # deterministic
                    logprobs=True,
                    top_logprobs=20,         # inspect the 20 most likely first tokens
                )
                break
            except Exception as e:
                wait = RETRY_WAIT * (2 ** attempt)
                print(f"  API error ({e}); retry {attempt+1}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError("max retries exceeded")

        top = resp.choices[0].logprobs.content[0].top_logprobs

        # collect probability mass on each label (match case-insensitively, first token)
        label_prob = {l: 0.0 for l in LABELS}
        for cand in top:
            tok = cand.token.strip().lower()
            for l in LABELS:
                if tok == l or tok == l[:3]:      # 'med' also counts toward 'medium'
                    label_prob[l] += math.exp(cand.logprob)

        total = sum(label_prob.values())
        if total > 0:
            label_prob = {l: p / total for l, p in label_prob.items()}
            pred = max(label_prob, key=label_prob.get)
        else:
            # none of the top logprobs were labels — fall back to the raw text
            raw = resp.choices[0].message.content.strip().lower()
            pred = next((l for l in LABELS if l in raw), "unknown")
            label_prob = {l: (1.0 if l == pred else 0.0) for l in LABELS}

        return pred, label_prob


    # ─────────────────────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────────────────────

    test_users = pl.read_ndjson(INPUT_TEST_PATH)
    if N_USERS is not None:
        test_users = test_users.head(N_USERS)

    test_users = test_users.with_columns(
        pl.col("SES_users").replace({"lower": "low", "middle": "medium"})
    )
    if CLASSES != '3':
        test_users = test_users.filter(pl.col("SES_users") != "medium")

    print(f"\n{len(test_users)} users | true class distribution:")
    print(test_users["SES_users"].value_counts())

    # ─────────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────────

    results = []

    for i, row in enumerate(test_users.iter_rows(named=True), 1):
        try:
            prompt_labels = random.sample(LABELS, k=len(LABELS)) if SHUFFLE_LABELS else LABELS

            if FEATURES == 'only_text':
                messages = build_messages(row, FEW_SHOTS, prompt_labels)
            else:
                messages = build_messages_with_features(row, FEW_SHOTS, prompt_labels)

            #print(messages)

            pred, probs = classify_user(messages)

            rec = {
                "user_id":    row["user_id"],
                "true":       row["SES_users"],
                "pred":       pred,
                "confidence": max(probs.values()),
                "n_tweets":   len(row["text"]),
            }
            rec.update({f"prob_{l}": probs[l] for l in LABELS})
            results.append(rec)

            mark = "✓" if pred == row["SES_users"] else "✗"
            print(f"[{i}/{len(test_users)}] {mark} user {row['user_id']} → "
                f"{pred} (conf {max(probs.values()):.2f}) | true: {row['SES_users']}")

            if REQUEST_PAUSE:
                time.sleep(REQUEST_PAUSE)

        except Exception as e:
            print(f"[{i}/{len(test_users)}] ERROR user {row.get('user_id')}: {e}")
            continue

    # ─────────────────────────────────────────────────────────────────────────────
    # Save + summary
    # ─────────────────────────────────────────────────────────────────────────────

    results_df = pl.DataFrame(results)
    results_df.write_csv(PREDICTIONS_OUTPUT_PATH)
    print(f"\nSaved → {PREDICTIONS_OUTPUT_PATH}")

    try:
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        y_true = results_df["true"].to_list()
        y_pred = results_df["pred"].to_list()
        print(f"\nAccuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"F1 macro : {f1_score(y_true, y_pred, average='macro'):.4f}")
        print(f"Random baseline (balanced): {1/len(LABELS):.4f}")
        print("\n" + classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    except ImportError:
        pass