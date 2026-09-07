"""
Zero-/few-shot socioeconomic status classification of Twitter users with Llama.

Works with both Meta-Llama-3.1-8B-Instruct and Meta-Llama-3.1-70B-Instruct:
set MODEL_SIZE below and the GPU/quantization settings adapt automatically.

Predictions are read from the next-token logits over the label tokens rather
than from generated text, so every user yields a calibrated probability
distribution over the classes and no output parsing is needed.
"""

import os


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — must be set BEFORE importing torch
# ─────────────────────────────────────────────────────────────────────────────

MODEL_SIZE = "70B"          # "8B" | "70B"if GPUS is not None:
GPUS       = None           # None = use all GPUs the container exposes
                            # "0"   = first one only
                            # "0,1" = first two

if GPUS is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUS

os.environ.setdefault("HF_HOME", "/source/hf_cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
from collections import Counter

import polars as pl
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ─────────────────────────────────────────────────────────────────────────────
# Model configuration per size
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "8B": {
        "model_id":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "quantize":   False,        # ~16 GB in bf16 — fits one A40 comfortably
        "per_gpu_gb": 40,
    },
    "70B": {
        "model_id":   "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "quantize":   True,         # ~40 GB in 4-bit; 140 GB in bf16
        "per_gpu_gb": 40,
    },
}

CFG      = MODEL_CONFIGS[MODEL_SIZE]
MODEL_ID = CFG["model_id"]

# ─────────────────────────────────────────────────────────────────────────────
# Experiment settings
# ─────────────────────────────────────────────────────────────────────────────

CLASSES     = '2_2'             # 2 / 2_2 -> low/high, 3 -> low/medium/high

prompt_types = ['complex_with_direction']#'simple', 'complex', , 'specific_all_features'

for prompt_type in prompt_types:

    PROMPT_TYPE = prompt_type      # simple | complex | complex_with_direction | specific_all_features
    if prompt_type == 'specific_all_features':
        FEATURES = 'all'
    else:
        FEATURES    = "only_text"   # all | corr | only_text

    FEW_SHOTS   = False
    N_USERS     = None          # None = all users; set an int to debug on a subset
    SHUFFLE_LABELS = True       # randomise label order in the prompt (position bias)
    SEED        = 42

    LABELS = ["low", "medium", "high"] if CLASSES == '3' else ["low", "high"]


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

    print(MODEL_ID, CLASSES, PROMPT_TYPE, FEATURES, FEW_SHOTS)
    print(INPUT_TEST_PATH)

    zero_few = "few" if FEW_SHOTS else "zero"
    PREDICTIONS_OUTPUT_PATH = (
        f"predictions_llama/classes_{CLASSES}/llama_{MODEL_SIZE}_{zero_few}_"
        f"predictions_{FEATURES}_prompt_{PROMPT_TYPE}_{CLASSES}_classes.csv"
    )

    os.makedirs("predictions_llama", exist_ok=True)
    random.seed(SEED)

    # ─────────────────────────────────────────────────────────────────────────────
    # Prompts
    # ─────────────────────────────────────────────────────────────────────────────

    if  PROMPT_TYPE == 'simple':
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
        """prompt_labels: label order as presented to the model (may be shuffled)."""
        definitions = ", ".join(LABEL_DEFINITIONS[l] for l in prompt_labels)
        options     = ", ".join(prompt_labels[:-1]) + f", or {prompt_labels[-1]}"

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            definitions=definitions, options=options
        )

        examples   = FEW_SHOT_EXAMPLES if few_shots else ""
        tweet_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(user_row["text"]))

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"{examples}{tweet_list}\n\nAnswer:"},
        ]

    def build_messages_with_features(user_row, few_shots, prompt_labels):
        """prompt_labels: label order as presented to the model (may be shuffled)."""
        definitions = ", ".join(LABEL_DEFINITIONS[l] for l in prompt_labels)
        options     = ", ".join(prompt_labels[:-1]) + f", or {prompt_labels[-1]}"

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            definitions=definitions, options=options
        )

        examples   = FEW_SHOT_EXAMPLES if few_shots else ""
        profile = user_row["llm_prompt"]
        tweets = "\n".join(f"{i+1}, {t}" for i, t in enumerate(user_row["text"]))

        content = f"{examples}{profile}\n\nSample tweets:\n{tweets}\n\nAnswer:"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ]


    def messages_to_prompt(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


    # ─────────────────────────────────────────────────────────────────────────────
    # Load model
    # ─────────────────────────────────────────────────────────────────────────────

    print(f"Loading {MODEL_ID}  (quantized={CFG['quantize']}, GPUs={GPUS})")

    n_gpus     = torch.cuda.device_count()
    max_memory = {i: f"{CFG['per_gpu_gb']}GiB" for i in range(n_gpus)}

    load_kwargs = dict(device_map="auto", max_memory=max_memory)

    if CFG["quantize"]:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()

    # ── verify nothing was offloaded ─────────────────────────────────────────────
    placement = Counter(str(d) for d in model.hf_device_map.values())
    print("Device placement:", dict(placement))

    offloaded = sum(v for k, v in placement.items() if k in ("cpu", "disk"))
    if offloaded:
        raise RuntimeError(
            f"{offloaded} modules offloaded to cpu/disk — the model does not fit.\n"
            f"Use more GPUs (GPUS='0,1,2'), enable quantization, "
            f"or pick GPUs that are not in use by other processes."
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Label token ids — computed once, verified to be single tokens
    # ─────────────────────────────────────────────────────────────────────────────

    label_token_ids = {}
    for lab in LABELS:
        ids = tokenizer.encode(lab, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Label '{lab}' tokenizes to {len(ids)} tokens ({ids}). "
                f"The logit-based method needs single-token labels — "
                f"pick a different surface form (e.g. 'mid' instead of 'medium')."
            )
        label_token_ids[lab] = ids[0]

    print("Label token ids:", label_token_ids)

    LABEL_IDS = torch.tensor([label_token_ids[l] for l in LABELS])

    # ─────────────────────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────────────────────
    print('INPUT TEST PATH', INPUT_TEST_PATH)
    test_users = pl.read_ndjson(INPUT_TEST_PATH)
    print(test_users)
    if N_USERS is not None:
        test_users = test_users.head(N_USERS)

    test_users = test_users.with_columns(
        pl.col("SES_users").replace({"lower": "low", "middle": "medium"})
    )

    # extract only users with labels SES to use
    if CLASSES != '3':
        test_users = test_users.filter(pl.col("SES_users") != "medium")

    print(test_users)
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
            prompt   = messages_to_prompt(messages)
            print(prompt)
            inputs   = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                logits            = model(**inputs).logits
                next_token_logits = logits[0, -1, :].float()

            # NOTE: LABEL_IDS follows LABELS order, not the shuffled prompt order,
            # so probs[k] always corresponds to LABELS[k].
            label_logits = next_token_logits[LABEL_IDS.to(next_token_logits.device)]
            probs        = torch.softmax(label_logits, dim=0).cpu()
            pred         = LABELS[int(probs.argmax())]

            rec = {
                "user_id":    row["user_id"],
                "true":       row["SES_users"],
                "pred":       pred,
                "confidence": float(probs.max()),
                "n_tweets":   len(row["text"]),
            }
            rec.update({f"prob_{l}": float(p) for l, p in zip(LABELS, probs)})
            results.append(rec)

            mark = "✓" if pred == row["SES_users"] else "✗"
            print(f"[{i}/{len(test_users)}] {mark} user {row['user_id']} → "
                f"{pred} (conf {float(probs.max()):.2f}) | true: {row['SES_users']}")

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