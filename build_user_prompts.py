"""
Aggregate tweet-level LanguageTool features to user level and build an
llm_prompt per user that explicitly reports error counts.

Output: jsonl with user_id, SES_users, llm_prompt  (+ raw feature columns)
ready to feed the Llama classification script.
"""

import polars as pl

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV   = "dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium.csv"#_2_classes
OUTPUT_JSONL = "dev_phase_normal/input with medium large/LLAMA_test_paris_LARGE_LARGE_tot_errors_with_medium.jsonl"#_2_classes

# error columns present in the CSV
ERROR_COLS = [
    "AGREEMENT", "CASING", "CAT_ELISION", "CAT_GRAMMAIRE",
    "CAT_HOMONYMES_PARONYMES", "CAT_MAJUSCULES", "CAT_PLEONASMES",
    "CAT_REGLES_DE_BASE", "CAT_TOURS_CRITIQUES", "CAT_TYPOGRAPHIE",
    "MISC", "MULTITOKEN_SPELLING", "PONCTUATION_POINT", "PONCTUATION_VIRGULE",
    "PUNCTUATION", "REPETITIONS_STYLE", "SEMANTICS", "STYLE",
    "TYPOGRAPHY", "TYPOS", "CAT_MARQUES_DE_COMMERCE", "CAT_REGIONALISMES",
]

# only these are surfaced in the prompt text (discriminative, human-readable).
# the rest are still aggregated and kept in the file, just not written into the
# prompt — dumping 22 mostly-zero categories adds noise.
PROMPT_ERROR_LABELS = {
    "TYPOS":                   "Typos",
    "CAT_GRAMMAIRE":           "Grammar",
    "AGREEMENT":               "Agreement",
    "CAT_HOMONYMES_PARONYMES": "Homonyms/paronyms",
    "CAT_ELISION":             "Elision",
    "CAT_TYPOGRAPHIE":         "Typography",
    "PONCTUATION_VIRGULE":     "Commas",
    "PONCTUATION_POINT":       "Periods",
    "CAT_TOURS_CRITIQUES":     "Critical turns",
    "STYLE":                   "Style",
}

N_SAMPLE_TWEETS = 0      # raw tweets shown in the prompt (0 to disable)

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

df = pl.read_csv(INPUT_CSV, infer_schema_length=5000)

# normalise label spelling if needed
df = df.with_columns(
    pl.col("SES_users").replace({"lower": "low", "middle": "medium"})
)

# ─────────────────────────────────────────────────────────────────────────────
# Aggregate to user level
# ─────────────────────────────────────────────────────────────────────────────

agg_exprs = []

# error columns: total and mean-per-tweet
for col in ERROR_COLS:
    agg_exprs.append(pl.col(col).sum().alias(f"{col}_total"))
    agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))

# tweet-level stats
agg_exprs += [
    pl.len().alias("n_tweets"),
    pl.col("TWEET_NUM_WORDS").sum().alias("total_words"),
    pl.col("TWEET_NUM_WORDS").mean().alias("avg_words"),
    pl.col("VOCAB_SIZE").first().alias("vocab_size"),   # already user-level
    pl.col("SES_users").first().alias("SES_users"),
    # negations / plurals
    pl.col("has_standard_neg").sum().alias("standard_neg"),
    pl.col("has_nonstandard_neg").sum().alias("nonstandard_neg"),
    pl.col("standard_plural").sum().alias("standard_plural"),
    pl.col("non_standard_plural").sum().alias("nonstandard_plural"),
    # keep all tweets for sampling
    pl.col("text").alias("all_tweets"),
]

user_df = df.group_by("user_id").agg(agg_exprs).sort("user_id")

# ─────────────────────────────────────────────────────────────────────────────
# Derived features
# ─────────────────────────────────────────────────────────────────────────────

user_df = user_df.with_columns(
    # total errors across all categories
    pl.sum_horizontal([pl.col(f"{c}_total") for c in ERROR_COLS])
      .alias("total_errors"),
).with_columns(
    # errors per word — comparable across users regardless of tweet volume
    (pl.col("total_errors") / (pl.col("total_words") + 1e-9))
      .alias("errors_per_word"),
    # negation / plural ratios
    (pl.col("nonstandard_neg") /
     (pl.col("standard_neg") + pl.col("nonstandard_neg") + 1e-9))
      .alias("nonstandard_neg_ratio"),
    (pl.col("nonstandard_plural") /
     (pl.col("standard_plural") + pl.col("nonstandard_plural") + 1e-9))
      .alias("nonstandard_plural_ratio"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(row: dict) -> str:
    # error lines — show total and per-tweet average, skip all-zero categories
    error_lines = []
    for col, label in PROMPT_ERROR_LABELS.items():
        total = row[f"{col}_total"]
        if total > 0:
            per_tweet = row[f"{col}_mean"]
            error_lines.append(f"- {label}: {int(total)} total ({per_tweet:.2f}/tweet)")
    errors_block = "\n".join(error_lines) if error_lines else "- No significant errors detected"

    # sample tweets
    sample_block = ""
    if N_SAMPLE_TWEETS > 0 and row["all_tweets"]:
        sample = row["all_tweets"][:N_SAMPLE_TWEETS]
        sample_block = "\n\nSample tweets:\n" + "\n".join(
            f"{i+1}. {t}" for i, t in enumerate(sample)
        )

    return (
        f"User linguistic profile ({row['n_tweets']} tweets):\n"
        f"- Avg tweet length: {row['avg_words']:.1f} words\n"
        f"- Vocabulary richness: {row['vocab_size']:.2f}\n"
        f"- Total errors: {int(row['total_errors'])} "
        f"({row['errors_per_word']:.3f} per word)\n"
        f"\nErrors by category:\n{errors_block}\n"
        f"\nNegations: {row['standard_neg']} standard, "
        f"{row['nonstandard_neg']} non-standard "
        f"({row['nonstandard_neg_ratio']:.0%} non-standard)\n"
        f"Pluralization: {row['standard_plural']} standard, "
        f"{row['nonstandard_plural']} non-standard "
        f"({row['nonstandard_plural_ratio']:.0%} non-standard)"
        f"{sample_block}"
    )


user_df = user_df.with_columns(
    pl.struct(pl.all()).map_elements(build_prompt, return_dtype=pl.String)
      .alias("llm_prompt")
)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

# full feature matrix (for CamemBERT head / analysis) — drop the list column
#user_df.drop("all_tweets").write_csv("user_features.csv")

# prompt file for Llama — text column is a list, needed by the text-only prompt
out = user_df.select([
    "user_id",
    "SES_users",
    "llm_prompt",
    pl.col("all_tweets").alias("text"),   # keep raw tweets too
])
out.write_ndjson(OUTPUT_JSONL)

print(f"{len(user_df)} users written to {OUTPUT_JSONL}")
print("\nClass distribution:")
print(user_df["SES_users"].value_counts())
print("\nSample prompt:\n")
print(user_df["llm_prompt"][0])
