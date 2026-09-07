import pandas as pd
import polars as pl
from sklearn.metrics import confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


# read train, dev, test data, put them togehter and compute statistics on them

INPUT_TRAIN_PATH = "data statistics/train_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION.csv"
INPUT_DEV_PATH = "data statistics/test_dev_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION.csv"
INPUT_TEST_PATH = "data statistics/test_test_paris_LARGE_LARGE_tot_errors_with_medium_REGRESSION.csv"

df_train = pd.read_csv(INPUT_TRAIN_PATH)
df_dev = pd.read_csv(INPUT_DEV_PATH)
df_test = pd.read_csv(INPUT_TEST_PATH)

df_train = df_train.drop(columns=['PONCTUATION_STYLE_OS'])

df_tot = pd.concat([df_train, df_dev, df_test], ignore_index=True)


# --------------------------
# FUNCTIONS
# --------------------------


# ASSOCIATION SES - LINGUISTIC VARIABLES

def analyze_ses_associations(df, ses_col='SES_users', 
                              continuous_vars=None, binary_vars=None):
    """
    Analyze associations between SES (ordinal) and linguistic variables.
    
    - continuous vars (vocab size, tweet length): Spearman + Kruskal-Wallis + descriptives
    - binary vars (negation, pluralization):      Spearman + Mann-Whitney per SES pair 
                                                  + rank-biserial + descriptives
    """
    if continuous_vars is None:
        continuous_vars = ['VOCAB_SIZE', 'TWEET_NUM_WORDS']
    if binary_vars is None:
        binary_vars = ['has_nonstandard_neg', 'non_standard_plural']

    ses_groups = sorted(df[ses_col].unique())

    # ------------------------------------------------------------------ #
    # continuous variables                                                  #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("CONTINUOUS VARIABLES vs SES")
    print("=" * 60)

    for var in continuous_vars:
        print(f"\n--- {var.upper()} ---")

        # Spearman
        r, p = stats.spearmanr(df[var], df[ses_col])
        print(f"Spearman r={r:.4f}, p={p:.4f}")

        # Kruskal-Wallis across all SES groups
        groups = [df.loc[df[ses_col] == g, var].dropna() for g in ses_groups]
        h, p_kw = stats.kruskal(*groups)
        print(f"Kruskal-Wallis H={h:.2f}, p={p_kw:.4f}")

        # descriptives per SES group
        print(df.groupby(ses_col)[var].agg(['mean', 'median', 'std', 'count']).round(3))

    # ------------------------------------------------------------------ #
    # binary variables                                                      #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("BINARY VARIABLES vs SES")
    print("=" * 60)

    for var in binary_vars:
        print(f"\n--- {var.upper()} ---")

        # Spearman (works fine with binary vs ordinal)
        r, p = stats.spearmanr(df[var], df[ses_col])
        print(f"Spearman r={r:.4f}, p={p:.4f}")

        # prevalence of the binary variable per SES group
        prev = df.groupby(ses_col)[var].agg(['mean', 'sum', 'count'])
        prev.columns = ['prevalence', 'n_positive', 'n_total']
        prev['prevalence_pct'] = (prev['prevalence'] * 100).round(2)
        print(prev[['n_total', 'n_positive', 'prevalence_pct']])

        # Kruskal-Wallis across SES groups
        groups = [df.loc[df[ses_col] == g, var].dropna() for g in ses_groups]
        h, p_kw = stats.kruskal(*groups)
        print(f"Kruskal-Wallis H={h:.2f}, p={p_kw:.4f}")

        # pairwise Mann-Whitney between consecutive SES groups + rank-biserial
        print("Pairwise Mann-Whitney (consecutive SES groups):")
        for g1, g2 in zip(ses_groups, ses_groups[1:]):
            a = df.loc[df[ses_col] == g1, var].dropna()
            b = df.loc[df[ses_col] == g2, var].dropna()
            u, p_mw = stats.mannwhitneyu(a, b, alternative='two-sided')
            rb = 1 - (2 * u) / (len(a) * len(b))
            print(f"  {g1} vs {g2}: U={u:.1f}, p={p_mw:.4f}, rank-biserial={rb:.4f}")



# function to create df in right format to plot categories or compute statistics

def create_categories_df(df, errors_categories, groupby_col, columns_columns):
    
    # 1. Melt category columns into rows
    cat_cols = [c for c in errors_categories]
    
    melted = df.melt(
        id_vars=["user_id", "SES_users"],
        value_vars=cat_cols,
        var_name="category",
        value_name="count"
    )
    
    # 2. Normalize category names
    melted["category"] = melted["category"].str.upper()
    
    # 3. Group by category and ses_users, sum counts
    result = (
        melted
        .groupby(["category", groupby_col])["count"]
        .sum()
        .unstack(columns_columns)  # ses_users values become columns
    )

    return result


# CORRELATION FEATURES - SES/INCOME -> LINGUISTIC ERRORS FROM LANGTOOL AGGREAGTED

def aggregate_and_correlate(
    df,
    user_col       = 'user_id',
    tweet_col      = 'tweet_id',
    error_cols     = None,       # list of error count columns
    binary_cols    = None,       # list of binary linguistic feature columns
    continuous_cols= None,       # list of continuous linguistic feature columns
    target_cols    = None,       # list of targets, e.g. ['income', 'ses_class']
    first_cols     = None,       # columns that are user-level (take first)
    drop_threshold = 0.02,       # for binarizing proportions if needed
):
    """
    1. Aggregates df from tweet-level to user-level
    2. Computes error rates
    3. Runs Spearman correlations of all features vs each target
    """

    if error_cols      is None: error_cols      = []
    if binary_cols     is None: binary_cols     = []
    if continuous_cols is None: continuous_cols = []
    if target_cols     is None: target_cols     = []
    if first_cols      is None: first_cols      = []

    # ------------------------------------------------------------------ #
    # 1. build aggregation dict                                            #
    # ------------------------------------------------------------------ #
    agg_dict = {}

    # tweet count
    agg_dict['n_tweets'] = (tweet_col, 'count')

    # continuous features → mean
    for col in continuous_cols:
        agg_dict[col] = (col, 'mean')

    # binary features → proportion of tweets with that feature , we have 0/1 for each tweet, if we compute the mean it corresponds to te rate 
    # (1+0+1+1+0.. / N_samples -> num_tweets_with_non-st/num_tot_tweets)
    for col in binary_cols:
        print('ciao')
        print(col)
        agg_dict[col] = (col, 'mean')

    # error counts → sum (we'll compute rate after)
    for col in error_cols:
        agg_dict[f'{col}_sum'] = (col, 'sum')
        agg_dict[f'{col}'] = (col, 'mean')

    # user-level columns → first
    for col in first_cols + target_cols:
        agg_dict[col] = (col, 'first')

    # ------------------------------------------------------------------ #
    # 2. aggregate                                                         #
    # ------------------------------------------------------------------ #
    agg_df = df.groupby(user_col).agg(**agg_dict).reset_index()

    # compute error rates
    for col in error_cols:
        agg_df[f'{col}_rate'] = agg_df[f'{col}_sum'] / agg_df['n_tweets']

    print(f"Aggregated: {len(agg_df)} users")
    print(f"Columns: {list(agg_df.columns)}\n")

    # ------------------------------------------------------------------ #
    # 3. define feature sets for correlation                               #
    # ------------------------------------------------------------------ #
    rate_cols    = [f'{col}_rate' for col in error_cols]
    sum_cols  = [f'{col}_sum'  for col in error_cols]
    mean_cols = [f'{col}' for col in error_cols]
    feature_cols = continuous_cols + binary_cols + rate_cols + sum_cols + mean_cols

    print(f"Features for correlation: {feature_cols}")

    # ------------------------------------------------------------------ #
    # 4. run correlations vs each target                                   #
    # ------------------------------------------------------------------ #
    all_results = {}

    for target in target_cols:
        print("=" * 60)
        print(f"CORRELATIONS vs {target.upper()}")
        print("=" * 60)

        results = []
        for col in feature_cols:
            valid = agg_df[[col, target]].dropna()
            r, p  = stats.spearmanr(valid[col], valid[target])
            results.append({
                'variable':   col,
                'spearman_r': round(r, 4),
                'p_value':    p,
                'n':          len(valid),
                'significant': p < 0.05,
            })

        results_df = (
            pd.DataFrame(results)
            .sort_values('spearman_r', key=abs, ascending=False)
            .reset_index(drop=True)
        )
        print(results_df.to_string(index=False))
        print()
        all_results[target] = results_df
    
    print("Aggregated DataFrame:")
    print(all_results.keys())
    print(agg_df.columns)

    return agg_df, all_results


# PLOT CORRELATION FEATURES - SES/INCOME -> LINGUISTIC ERRORS FROM LANGTOOL AGGREAGTED

def plot_correlations(results, target, feature_subset=None, title=None, type_feat=None, type_df=None, output_dir='correlations/correlations_no_p_no_annotation_ok/'):
    """
    Bar plot of Spearman correlations for a subset of features vs a target.
    
    results       : dict output from aggregate_and_correlate
    target        : string, key in results dict (e.g. 'income')
    feature_subset: list of column names to plot (e.g. rate_cols). If None, plots all.
    """
    df_res = results[target].copy()

    if feature_subset is not None:
        df_res = df_res[df_res['variable'].isin(feature_subset)]

    df_res = df_res.sort_values('spearman_r')

    colors = ['#2a78d6' if r >= 0 else '#eb6834' for r in df_res['spearman_r']]
    # hatching for non-significant
    #hatches = ['' if s else '///' for s in df_res['significant']]

    fig, ax = plt.subplots(figsize=(12, len(df_res) * 0.5 + 1.5))

    bars = ax.barh(df_res['variable'], df_res['spearman_r'],
                   color=colors, edgecolor='white', height=0.6)

    '''for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    '''
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Spearman r', fontsize=25)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    #ax.set_title(title or f'Correlations vs {target}')

    # annotate r values
    '''for bar, (_, row) in zip(bars, df_res.iterrows()):
        x = row['spearman_r']
        offset = 0.002 if x >= 0 else -0.002
        ha = 'left' if x >= 0 else 'right'
        #sig_marker = '' if row['significant'] else ' (n.s.)'
        sig_marker = ''
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                f"{x:.3f}{sig_marker}", va='center', ha=ha, fontsize=9)'''

    plt.tight_layout()
    plt.savefig(output_dir + f'correlations_{type_df}_{target}_{type_feat}.png', dpi=150)
    plt.show()


def plot_two_correlations(results, target, subset_top, subset_bottom, output_path):
    df1 = results[target][results[target]['variable'].isin(subset_top)].sort_values('spearman_r')
    df2 = results[target][results[target]['variable'].isin(subset_bottom)].sort_values('spearman_r')

    n1, n2 = len(df1), len(df2)
    # height ratio proportional to bar counts -> identical bar thickness
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(16, (n1 + n2) * 0.5 + 2),
        gridspec_kw={'height_ratios': [n1, n2]},
        sharex=True                       # same x-axis scale for both
    )

    # panel labels (a) and (b)
    ax1.text(-0.7, 1.05, 'A', transform=ax1.transAxes,
         fontsize=40, va='bottom', ha='left')#fontweight='bold', 
    ax2.text(-0.7, 1.05, 'B', transform=ax2.transAxes,
            fontsize=40, va='bottom', ha='left')#fontweight='bold', 

    for ax, df in [(ax1, df1), (ax2, df2)]:
        colors = ['#2a78d6' if r >= 0 else '#eb6834' for r in df['spearman_r']]
        ax.barh(df['variable'], df['spearman_r'], color=colors,
                edgecolor='white', height=0.6)
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.tick_params(axis='y', labelsize=20)

    ax1.tick_params(axis='x', labelbottom=True, labelsize=20)
    ax2.set_xlabel('Spearman r', fontsize=25)
    ax2.tick_params(axis='x', labelsize=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

# ----------------------------------------
# 1) LINGUISTIC VARIABLES VS INCOME/SES
# ----------------------------------------

'''# SES
analyze_ses_associations(
    df_train,
    ses_col='SES_users',
    continuous_vars=['VOCAB_SIZE', 'TWEET_NUM_WORDS'],
    binary_vars=['has_nonstandard_neg', 'non_standard_plural']
)'''


# ------------------------------------------------
# COMPUTE CORRELATION LANGTOOL ERRORS - INCOME
# ------------------------------------------------

'''ERRORS_CATEGORIES = ['AGREEMENT', 'CASING',
       'CAT_ELISION', 'CAT_GRAMMAIRE', 'CAT_HOMONYMES_PARONYMES',
       'CAT_MAJUSCULES', 'CAT_PLEONASMES', 'CAT_REGLES_DE_BASE',
       'CAT_TOURS_CRITIQUES', 'CAT_TYPOGRAPHIE', 'MISC', 'MULTITOKEN_SPELLING',
       'PONCTUATION_POINT', 'PONCTUATION_VIRGULE', 'PUNCTUATION',
       'REPETITIONS_STYLE', 'STYLE', 'TYPOGRAPHY', 'TYPOS', 'SEMANTICS',
       'CAT_MARQUES_DE_COMMERCE']#, 'CAT_REGIONALISMES'


df_train_categories = create_categories_df(df_train, ERRORS_CATEGORIES, 'SES_users', 'SES_users')
df_train_test_categories = create_categories_df(df_tot, ERRORS_CATEGORIES, 'SES_users', 'SES_users')



# 1) PLOT ERRORS PER CATEGORY PER SES

# sort columns (SES classes) in desired order
class_order = ['lower', 'medium', 'high']
color_map = {'lower': 'blue', 'medium': '#1fed1f', 'high': 'red'}

# TRAIN

# reorder columns if they exist
existing_classes = [c for c in class_order if c in df_train_categories.columns]
df_plot = df_train_categories[existing_classes]

label_map = {'lower': 'Low SES', 'medium': 'Medium SES', 'high': 'High SES'}

df_plot['total'] = df_plot[['lower', 'medium', 'high']].sum(axis=1)
print(df_plot)
df_plot = df_plot.sort_values('total', ascending=False)
df_plot = df_plot.drop(columns='total')


# plot
ax = df_plot.plot(
    kind="bar",
    color=[color_map[c] for c in existing_classes],
    figsize=(10, 6)
)
# rename legend labels

handles, labels = ax.get_legend_handles_labels()
print(labels)
ax.legend(
    handles,
    [label_map[l] for l in labels],
    title="SES",
    fontsize = 12,
    title_fontsize = 13
)

plt.ylabel("Count", fontsize=16)
plt.xlabel("Error categories", fontsize=16)
plt.xticks(rotation=45, ha="right")
#plt.legend(title="SES", fontsize=13)
plt.yscale('log')
plt.tight_layout()
plt.savefig('data statistics/TRAIN_ILE_DE_FRANCE_mistakes_category_count_log_scale.png')
plt.show()


# 2) TRAIN + DEV + TEST

# reorder columns if they exist
existing_classes = [c for c in class_order if c in df_train_categories.columns]
df_plot = df_train_test_categories[existing_classes]

label_map = {'lower': 'Low SES', 'medium': 'Medium SES', 'high': 'High SES'}

df_plot['total'] = df_plot[['lower', 'medium', 'high']].sum(axis=1)
print(df_plot)
df_plot = df_plot.sort_values('total', ascending=False)
df_plot = df_plot.drop(columns='total')
#df_plot = df_plot.sort_values('lower', ascending=False)

# plot
ax = df_plot.plot(
    kind="bar",
    color=[color_map[c] for c in existing_classes],
    figsize=(10, 6)
)
# rename legend labels

handles, labels = ax.get_legend_handles_labels()
print(labels)
ax.legend(
    handles,
    [label_map[l] for l in labels],
    title="SES",
    fontsize = 12,
    title_fontsize = 13
)

plt.ylabel("Count", fontsize=16)
plt.xlabel("Error categories", fontsize=16)
plt.xticks(rotation=45, ha="right")
#plt.legend(title="SES", fontsize=13)
plt.yscale('log')
plt.tight_layout()
plt.savefig('data statistics/TRAIN_DEV_TEST_ILE_DE_FRANCE_mistakes_category_count_log_scale.png')
plt.show()'''



# 2) COMPUTE CORRELATION FEATURES - SES/INCOME

ERRORS_CATEGORIES = ['AGREEMENT', 'CASING',
       'CAT_ELISION', 'CAT_GRAMMAIRE', 'CAT_HOMONYMES_PARONYMES',
       'CAT_MAJUSCULES', 'CAT_REGLES_DE_BASE',
       'CAT_TOURS_CRITIQUES', 'CAT_TYPOGRAPHIE', 'MISC', 'MULTITOKEN_SPELLING',
       'PONCTUATION_POINT', 'PONCTUATION_VIRGULE', 'PUNCTUATION',
       'REPETITIONS_STYLE', 'STYLE', 'TYPOGRAPHY', 'TYPOS', 'SEMANTICS',
       ]#, 'CAT_REGIONALISMES''CAT_PLEONASMES', 'CAT_MARQUES_DE_COMMERCE'


# TRAIN set

# rename columns of non-standard form features
print('ciao', df_train)
print(df_train.columns)

#df_train = df_train.rename(columns={'has_nonstandard_neg': 'NON_STANDARD_NEGATION', 'non_standard_plural': 'NON_STANDARD_PLURAL'})
print(df_train)
agg_df_train, results_train = aggregate_and_correlate(
    df_train,
    user_col        = 'user_id',
    tweet_col       = 'tweet_id',
    error_cols      = ERRORS_CATEGORIES, #['n_grammar', 'n_typos', 'n_spelling']   # your error columns
    binary_cols     = ['has_nonstandard_neg', 'non_standard_plural'], #['NON_STANDARD_NEGATION', 'NON_STANDARD_PLURAL']
    continuous_cols = ['VOCAB_SIZE', 'TWEET_NUM_WORDS'],
    target_cols     = ['ind_snv_mean', 'SES_users'],
    #first_cols      = ['country', 'user_type'],                  # any other user-level attributes
)

# access results per target
income_results_train = results_train['ind_snv_mean']
ses_results_train    = results_train['SES_users']

# save results
income_results_train.to_csv('correlations/feat_income_train.csv', index=False)
ses_results_train.to_csv('correlations/feat_ses_train.csv', index=False)


# TRAIN + DEV +TEST

agg_df_tot, results_tot = aggregate_and_correlate(
    df_tot,
    user_col        = 'user_id',
    tweet_col       = 'tweet_id',
    error_cols      = ERRORS_CATEGORIES, #['n_grammar', 'n_typos', 'n_spelling']   # your error columns
    binary_cols     = ['has_nonstandard_neg', 'non_standard_plural'], #['NON_STANDARD_NEGATION', 'NON_STANDARD_PLURAL'],
    continuous_cols = ['VOCAB_SIZE', 'TWEET_NUM_WORDS'],
    target_cols     = ['ind_snv_mean', 'SES_users'],
    #first_cols      = ['country', 'user_type'],                  # any other user-level attributes
)

# access results per target
income_results_tot = results_tot['ind_snv_mean']
ses_results_tot    = results_tot['SES_users']

# save results
income_results_tot.to_csv('correlations/feat_income_train_dev_test.csv', index=False)
ses_results_tot.to_csv('correlations/feat_ses_train_dev_test.csv', index=False)



# 3) PLOT CORRELATIONS

# TRAIN set

error_rate_cols = [f'{col}_rate' for col in ERRORS_CATEGORIES]
error_sum_cols = [f'{col}_sum' for col in ERRORS_CATEGORIES]
error_mean_cols = [f'{col}' for col in ERRORS_CATEGORIES]

other_features_cols = ['NON_STANDARD_PLURAL', 'NON_STANDARD_NEGATION', 'TWEET_NUM_WORDS', 'VOCAB_SIZE']
#other_features_cols = ['non_standard_plural', 'has_nonstandard_neg', 'TWEET_NUM_WORDS', 'VOCAB_SIZE']

for key in results_train.keys():
    results_train[key]["variable"] = results_train[key]["variable"].replace(
        "non_standard_plural", "NON_STANDARD_PLURAL"
    )
for key in results_train.keys():
    results_train[key]["variable"] = results_train[key]["variable"].replace(
        "has_nonstandard_neg", "NON_STANDARD_NEGATION"
    )

# separated plots

plot_correlations(results_train, target='ind_snv_mean',    feature_subset=error_rate_cols, type_feat='rate', type_df='train')
plot_correlations(results_train, target='SES_users', feature_subset=error_rate_cols, type_feat='rate', type_df='train')
plot_correlations(results_train, target='ind_snv_mean',    feature_subset=error_sum_cols, type_feat='sum', type_df='train')
plot_correlations(results_train, target='SES_users', feature_subset=error_sum_cols, type_feat='sum', type_df='train')
plot_correlations(results_train, target='ind_snv_mean',    feature_subset=error_mean_cols, type_feat='mean', type_df='train')
plot_correlations(results_train, target='SES_users', feature_subset=error_mean_cols, type_feat='mean', type_df='train')

plot_correlations(results_train, target='ind_snv_mean',    feature_subset=other_features_cols, type_feat='', type_df='train')
plot_correlations(results_train, target='SES_users', feature_subset=other_features_cols, type_feat='', type_df='train')

# plots together

plot_two_correlations(results_train, target='ind_snv_mean', subset_top=error_rate_cols, subset_bottom=other_features_cols, output_path='correlations/income_mean_train.pdf')


# TRAIN + DEV + TEST

error_rate_cols = [f'{col}_rate' for col in ERRORS_CATEGORIES]
error_sum_cols = [f'{col}_sum' for col in ERRORS_CATEGORIES]
error_mean_cols = [f'{col}' for col in ERRORS_CATEGORIES]

other_features_cols = ['NON_STANDARD_PLURAL', 'NON_STANDARD_NEGATION', 'TWEET_NUM_WORDS', 'VOCAB_SIZE']

for key in results_tot.keys():
    results_tot[key]["variable"] = results_tot[key]["variable"].replace(
        "non_standard_plural", "NON_STANDARD_PLURAL"
    )
for key in results_tot.keys():
    results_tot[key]["variable"] = results_tot[key]["variable"].replace(
        "has_nonstandard_neg", "NON_STANDARD_NEGATION"
    )

# separated plots

plot_correlations(results_tot, target='ind_snv_mean',    feature_subset=error_rate_cols, type_feat='rate', type_df='train_dev_test')
plot_correlations(results_tot, target='SES_users', feature_subset=error_rate_cols, type_feat='rate', type_df='train_dev_test')
plot_correlations(results_tot, target='ind_snv_mean',    feature_subset=error_sum_cols, type_feat='sum', type_df='train_dev_test')
plot_correlations(results_tot, target='SES_users', feature_subset=error_sum_cols, type_feat='sum', type_df='train_dev_test')
plot_correlations(results_tot, target='ind_snv_mean',    feature_subset=error_mean_cols, type_feat='mean', type_df='train_dev_test')
plot_correlations(results_tot, target='SES_users', feature_subset=error_mean_cols, type_feat='mean', type_df='train_dev_test')

plot_correlations(results_tot, target='ind_snv_mean',    feature_subset=other_features_cols, type_feat='', type_df='train_dev_test')
plot_correlations(results_tot, target='SES_users', feature_subset=other_features_cols, type_feat='', type_df='train_dev_test')

# plots together

plot_two_correlations(results_tot, target='ind_snv_mean', subset_top=error_rate_cols, subset_bottom=other_features_cols, output_path='correlations/income_mean_train_dev_test.pdf')


# or plot everything together
#plot_correlations(results, target='income')




# -----------------------------------------------------------------------------------
# 2) PLOT NUMBER OF USERS AT DIFFERENT THRESHOLDS OF NUMBER OF TWEETS BY CLASS SES
# -----------------------------------------------------------------------------------

'''# files with num tweets thresholds

df_min = pd.read_csv('data statistics/tweets < n.csv')
df_max = pd.read_csv('data statistics/tweets > n.csv')


# users with less than N tweets

thresholds = df_min['Less than N tweets -> N']
num_users_tot = df_min['tot']

print(thresholds)
print(num_users_tot)

x = np.arange(len(thresholds))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))

#ax.bar(x - width, lower,  width, label='Lower', color='#1baf7a')
ax.bar(x,        num_users_tot, width, color='#eb6834')
#ax.bar(x + width, high,   width, label='High',   color='#2a78d6')

ax.set_xticks(x)
ax.set_xticklabels(thresholds)
ax.set_xlabel('Tweets threshold (N)')
ax.set_ylabel('Number of users with number of tweets < N')
ax.set_title('Number of users per SES class across tweet thresholds')
#ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
plt.tight_layout()
plt.savefig('data statistics/tweet_threshold_less_N.png', dpi=150)

plt.show()


# users with less than N tweets divided by class

thresholds = df_min['Less than N tweets -> N']
lower  = df_min['lower class']
medium = df_min['medium class']
high   = df_min['high class']

x = np.arange(len(thresholds))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))

#ax.bar(x - width, lower,  width, label='Lower', color='#1baf7a')
ax.bar(x,         medium, width, label='Medium', color='#eb6834')
#ax.bar(x + width, high,   width, label='High',   color='#2a78d6')

ax.set_xticks(x)
ax.set_xticklabels(thresholds)
ax.set_xlabel('Tweets threshold (N)')
ax.set_ylabel('Number of users with number of tweets < N')
ax.set_title('Number of users per SES class across tweet thresholds')
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
plt.tight_layout()
#plt.savefig('data statistics/tweet_threshold_less_N_by_ses.png', dpi=150)
plt.savefig('data statistics/prova.png', dpi=150)

plt.show()'''


'''# users with more than N tweets

thresholds = df_max['More than N tweets -> N']
lower  = df_max['lower class']
medium = df_max['medium class']
high   = df_max['high class']

x = np.arange(len(thresholds))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(x - width, lower,  width, label='Lower', color='#1baf7a')
ax.bar(x,         medium, width, label='Medium', color='#eb6834')
ax.bar(x + width, high,   width, label='High',   color='#2a78d6')

ax.set_xticks(x)
ax.set_xticklabels(thresholds)
ax.set_xlabel('Tweets threshold (N)')
ax.set_ylabel('Number of users with number of tweets > N')
ax.set_title('Number of users per SES class across tweet thresholds')
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
plt.tight_layout()
plt.savefig('data statistics/tweet_threshold_more_N_by_ses.png', dpi=150)
plt.show()
'''

'''#INPUT_TEST_PATH = "dev_phase_normal/input with medium large/LLAMA_test_paris_LARGE_LARGE_only_text_with_medium.jsonl"

PREDICTIONS_PATH = 'predictions_llama/classes_2_2/llama_70B_zero_predictions_only_text_prompt_simple_2_2_classes.csv'


test_users = pl.read_ndjson(INPUT_TEST_PATH)
print(test_users["SES_users"].value_counts())


predictions_df = pl.read_csv(PREDICTIONS_PATH)
y_true = predictions_df['true']
y_pred = predictions_df['pred']
print(confusion_matrix(y_true, y_pred, labels=["low","medium","high"]))


misclassified_high = predictions_df.filter((pl.col("true")=="high") & (pl.col("pred")=="low"))
print(misclassified_high["prob_low"].describe())
'''







# NOT TO USE

# INCOME

'''# iterate over different categories and plot, for each user, income VS number of errors and compute pearson correlation
# only train
df_categories = df_train_categories
print(df_categories)
# train + dev + test
#df_categories = df_train_test_categories

# only train
errors_categories = list(df_categories['category'].unique())
# train + dev + test
#errors_categories = list(df_train_test_categories['category'].unique())

corr_df = pd.DataFrame()


for category in errors_categories:

    income_vector = list(df_categories['ind_snv_mean'])
    error_vector = list(df_categories[category])

    print(category)
    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(income_vector, error_vector, alpha=0.1, s=15)
    #ax.set_yscale('log')
    ax.set_xlabel('Income')
    ax.set_ylabel('Number of errors')
    ax.set_title(category)
    #fig.savefig(output_dir_plots / 'pop_distr.png')
    plt.show()

    try:
        pearson_coeff = scipy.stats.pearsonr(income_vector, error_vector)
        spearman_coeff = scipy.stats.spearmanr(income_vector, error_vector)
    except:
        print('Not enough data')

    res_df = pd.DataFrame({'category': [category], 'pearson_coeff': [pearson_coeff[0]], 'p-value_pearson': [pearson_coeff[1]], 'spearman_coeff': [spearman_coeff[0]], 'p-value_spearman': [spearman_coeff[1]]})
    corr_df = pd.concat([corr_df, res_df])
    print(corr_df)

    print(pearson_coeff[0], pearson_coeff[1])

    print('pearson', pearson_coeff)
    print('spearman', spearman_coeff)

corr_df = corr_df.sort_values(by='pearson_coeff', ascending=True)

print(corr_df)'''


'''# TWEETS LENGTH

print('TWEET LENGTH')

x = df_tot['TWEET_NUM_WORDS']  # 0/1
y = df_tot['ind_snv_mean']

# Pearson
pearson_r, pearson_p = stats.pearsonr(x, y)

# Spearman
spearman_r, spearman_p = stats.spearmanr(x, y)

print(f"Pearson  r={pearson_r:.4f}, p={pearson_p:.4f}")
print(f"Spearman r={spearman_r:.4f}, p={spearman_p:.4f}")

print('\n')


# VOCABULARY SIZE

print('VOCABULARY SIZE')

x = df_tot['VOCAB_SIZE']  # 0/1
y = df_tot['ind_snv_mean']

# Pearson
pearson_r, pearson_p = stats.pearsonr(x, y)

# Spearman
spearman_r, spearman_p = stats.spearmanr(x, y)

print(f"Pearson  r={pearson_r:.4f}, p={pearson_p:.4f}")
print(f"Spearman r={spearman_r:.4f}, p={spearman_p:.4f}")


# NON-STANDARD NEGATION

print('NON-STANDARD-NEGATION')

# Point-biserial: the correct version of Pearson for binary vs continuous
pb_r, pb_p = stats.pointbiserialr(df_tot['has_nonstandard_neg'], df_tot['ind_snv_mean'])
print(f"Point-biserial r={pb_r:.4f}, p={pb_p:.4f}")

# Mann-Whitney U: non-parametric test of whether income distributions
# differ between the two groups (negation=0 vs negation=1)
group0 = df_tot.loc[df_tot['has_nonstandard_neg'] == 0, 'ind_snv_mean']
group1 = df_tot.loc[df_tot['has_nonstandard_neg'] == 1, 'ind_snv_mean']
u_stat, mw_p = stats.mannwhitneyu(group0, group1, alternative='two-sided')
print(f"Mann-Whitney U={u_stat:.1f}, p={mw_p:.4f}")

# Quick descriptive to accompany the test
print(df_tot.groupby('has_nonstandard_neg')['ind_snv_mean'].describe())

n0, n1 = len(group0), len(group1)
rank_biserial = 1 - (2 * u_stat) / (n0 * n1)
print(f"Rank-biserial correlation: {rank_biserial:.4f}")

print('\n')

# NON-STANDARD PLURALIZATION

print('NON-STANDARD PLURALIZATION')

# Point-biserial: the correct version of Pearson for binary vs continuous
pb_r, pb_p = stats.pointbiserialr(df_tot['non_standard_plural'], df_tot['ind_snv_mean'])
print(f"Point-biserial r={pb_r:.4f}, p={pb_p:.4f}")

# Mann-Whitney U: non-parametric test of whether income distributions
# differ between the two groups (negation=0 vs negation=1)
group0 = df_tot.loc[df_tot['non_standard_plural'] == 0, 'ind_snv_mean']
group1 = df_tot.loc[df_tot['non_standard_plural'] == 1, 'ind_snv_mean']
u_stat, mw_p = stats.mannwhitneyu(group0, group1, alternative='two-sided')
print(f"Mann-Whitney U={u_stat:.1f}, p={mw_p:.4f}")

# Quick descriptive to accompany the test
print(df_tot.groupby('non_standard_plural')['ind_snv_mean'].describe())

n0, n1 = len(group0), len(group1)
rank_biserial = 1 - (2 * u_stat) / (n0 * n1)
print(f"Rank-biserial correlation: {rank_biserial:.4f}")'''