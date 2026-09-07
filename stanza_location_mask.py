"""
Mask location (and optionally organization) mentions in tweets with placeholders,
using Stanza's French NER. Parallel to the GLiNER version.

Stanza uses a FIXED label set for French NER (LOC, ORG, PER, MISC), so instead of
requesting custom entity types we filter to the labels we want.

Install:
    pip install stanza polars
    # first run downloads the French models:
    #   python -c "import stanza; stanza.download('fr')"
"""

import re
import polars as pl
import stanza

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

INPUT_PATH  = "dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv"        # must have a `text` column
TEXT_COL    = "text"

MASK_MODE   = "location"              # "location" | "organization" | "both"
OUTPUT_PATH = f"tweets_ner_stanza_test_test_{MASK_MODE}.csv"

# Stanza French NER entity types -> placeholder.
# French NER labels: LOC (location), ORG (organization), PER (person), MISC.
TYPE_TO_PLACEHOLDER = {
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
}

if MASK_MODE == "location":
    ACTIVE_TYPES = {"LOC"}
elif MASK_MODE == "organization":
    ACTIVE_TYPES = {"ORG"}
elif MASK_MODE == "both":
    ACTIVE_TYPES = {"LOC", "ORG"}
else:
    raise ValueError(f"unknown MASK_MODE: {MASK_MODE}")

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

print("Loading Stanza French NER pipeline")
# download once if not present
try:
    nlp = stanza.Pipeline(lang="fr", processors="tokenize,ner", use_gpu=True,
                          tokenize_no_ssplit=True, verbose=False)
except Exception:
    stanza.download("fr")
    nlp = stanza.Pipeline(lang="fr", processors="tokenize,ner", use_gpu=True,
                          tokenize_no_ssplit=True, verbose=False)

# ─────────────────────────────────────────────────────────────────────────────
# Masking
# ─────────────────────────────────────────────────────────────────────────────

def mask_text(text: str):
    """Replace each detected entity of an active type with its placeholder.
    Returns (masked_text, list_of_(surface, placeholder))."""
    if not text or not text.strip():
        return text, []

    doc = nlp(text)

    # collect entities of the types we care about, with char offsets
    ents = [
        (e.start_char, e.end_char, TYPE_TO_PLACEHOLDER[e.type])
        for e in doc.ents
        if e.type in ACTIVE_TYPES
    ]
    if not ents:
        return text, []

    found = [(text[s:e], ph) for (s, e, ph) in ents]

    # replace right-to-left so char offsets stay valid
    ents = sorted(ents, key=lambda x: x[0], reverse=True)
    chars = list(text)
    for s, e, ph in ents:
        chars[s:e] = list(ph)
    masked = "".join(chars)

    # collapse consecutive identical placeholders
    for ph in set(TYPE_TO_PLACEHOLDER.values()):
        masked = re.sub(rf"({re.escape(ph)}\s*){{2,}}", ph + " ", masked)
    return masked.strip(), found

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

df = pl.read_csv(INPUT_PATH)
#df = df.head(100)
print(f"{len(df)} rows")

texts   = df[TEXT_COL].to_list()
masked  = []
found_c = []

for i, t in enumerate(texts):
    try:
        m, found = mask_text(t)
    except Exception as ex:
        print(f"  row {i} failed: {ex}")
        m, found = t, []
    masked.append(m)
    found_c.append("; ".join(f"{s}->{p}" for s, p in found))

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(texts)}")

df = df.with_columns([
    pl.Series("text_ner", masked),
    pl.Series("masked_entities", found_c),
])

df = df.rename({'text': 'text old', 'text_ner': 'text'})

df.write_csv(OUTPUT_PATH)
print(f"Saved -> {OUTPUT_PATH}")

# examples
print("\nExamples where something changed:")
shown = 0
for o, m in zip(texts, masked):
    if o != m:
        print(f"  orig: {o}")
        print(f"  mask: {m}\n")
        shown += 1
        if shown >= 8:
            break
