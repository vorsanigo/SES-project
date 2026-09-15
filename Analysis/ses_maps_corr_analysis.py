import polars as pl
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import scipy.stats
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table


def linear_regplot(x,y,confidence=0.95):
    # function modified by me to fit my data

    xs = np.array([x_ for x_, _ in sorted(zip(x, y))])
    ys = np.array([y_ for _, y_ in sorted(zip(x, y))])

    X = sm.add_constant(xs)
    res = sm.OLS(ys, X).fit()

    st, data, ss2 = summary_table(res, alpha=1-confidence)
    fittedvalues = data[:,2]
    predict_mean_se  = data[:,3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
    predict_ci_low, predict_ci_upp = data[:,6:8].T

    y_fit  = fittedvalues # 10**(fittedvalues)
    y_low  = predict_mean_ci_low #10**predict_mean_ci_low
    y_high = predict_mean_ci_upp #10**predict_mean_ci_upp
    stats  = {'intercept':res.params[0], 'slope': res.params[1], 'R2':res.rsquared}
    print('slope', stats['slope'])
    print('intercept', stats['intercept'])

    return xs, ys, y_fit, y_low, y_high, stats


'''def plot_regression(ax, df, title, color='sandybrown'):
    columns1 = 'gdp per capita'
    columns2 = 'tweets per day per person'
    df_plot = df[[columns1, columns2]]
    
    # Regression plot
    x, y, y_fit, y_low, y_high, stats = linear_regplot(np.log10(df_plot[columns1]), np.log10(df_plot[columns2]))
    b, m = stats['intercept'], stats['slope']
    
    ax.loglog(df[columns1], df[columns2], '.', markersize=8, label='Data', color='teal')
    ax.plot(10**x, y_fit, color='sandybrown', label=f'Power-law fit coeff. = {np.round(m,3)}')
    ax.fill_between(10**x, y_low, y_high, color='sandybrown', alpha=0.3, label="_nolegend_")
    
    xt = list(df[columns1].values)
    yt = list(df[columns2].values)
    t = list(df['ISO 2'].values)
    #texts = [ax.text(xt[i], yt[i], t[i], ha='left', va='bottom', size=10, color='black') for i in range(len(xt))]
    #adjust_text(texts, ax=ax)
    
    #ax.set_title(f'{title}', fontsize=16)
    ax.set_xlabel(, fontsize=14)
    
    if ax == axes[0]:  # Only set ylabel for the first plot
        ax.set_ylabel('# of tweets per day per person', fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12, loc='upper left', frameon=False)
    
    print(f"{title} stats:", stats)
    print(f"{title} Spearman correlation:", scipy.stats.spearmanr(np.log10(df_plot[columns1]), np.log10(df_plot[columns2])))'''


def plot_regression(ax, df, columns1, columns2, x_label, y_label, title, color='sandybrown'):

    df_plot = df[[columns1, columns2]]
    
    # Regression plot
    #x, y, y_fit, y_low, y_high, stats = linear_regplot(np.log10(df_plot[columns1]), np.log10(df_plot[columns2]))
    x, y, y_fit, y_low, y_high, stats = linear_regplot(df_plot[columns1], df_plot[columns2])
    b, m = stats['intercept'], stats['slope']

    rho, p = scipy.stats.pearsonr(df_plot[columns1], df_plot[columns2])
    
    ax.plot(df[columns1], df[columns2], '.', markersize=5, color='teal', alpha=0.7)#label='Data',
    ax.plot(x, y_fit, color='sandybrown', label=f"Pearson $r$ = {rho:.3f}\n"
        f"$p$ = {p:.2e}")#
    #ax.plot(10**x, y_fit, color='sandybrown', label=f'Power-law fit coeff. = {np.round(m,3)}')
    #ax.fill_between(10**x, y_low, y_high, color='sandybrown', alpha=0.3, label="_nolegend_")
    ax.fill_between(x, y_low, y_high, color='sandybrown', alpha=0.3, label="_nolegend_")

    '''xt = list(df[columns1].values)
    yt = list(df[columns2].values)
    t = list(df['ISO 2'].values)
    texts = [ax.text(xt[i], yt[i], t[i], ha='left', va='bottom', size=10, color='black') for i in range(len(xt))]
    adjust_text(texts, ax=ax)'''
    
    #ax.set_title(f'{title}', fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12, loc='upper left', frameon=False)
    
    print(f"{title} stats:", stats)
    print(f"{title} Spearman correlation:", scipy.stats.pearsonr(df_plot[columns1], df_plot[columns2]))

    '''rho, p = scipy.stats.pearsonr(df_plot[columns1], df_plot[columns2])

    textstr = (
        f"Pearson $r$ = {r:.3f}\n"
        f"$p$ = {'< 0.001' if p < 0.001 else f'{p:.3f}'}"
    )

    at = AnchoredText(
        textstr,
        loc='upper left',
        prop={'size': 10},
        frameon=True
    )

    ax.add_artist(at)'''



'''df = pl.read_csv('geo_ses_files/FR/dossier_complet.csv')
print(df)'''




'''dossier = pd.read_csv("geo_ses_files/FR/dossier_complet.csv", sep=";",
                      dtype={"CODGEO": str}, usecols=cols)
print(dossier)'''

'''import geopandas as gpd
import pandas as pd

# 1. load geometries
communes = gpd.read_file("geo_ses_files/FR/revenu-des-francais-a-la-commune.shp",
                         encoding="ISO-8859-1")
print(communes.columns.tolist())   # <-- FIND the commune-code column name
print(communes.head())

# 2. load education data (dossier complet), compute % higher ed
cols = ["CODGEO", "P22_NSCOL15P",
        "P22_NSCOL15P_SUP2", "P22_NSCOL15P_SUP34", "P22_NSCOL15P_SUP5"]
edu = pd.read_csv("geo_ses_files/FR/dossier_complet.csv", sep=";",
                  dtype={"CODGEO": str}, usecols=cols)

sup = ["P22_NSCOL15P_SUP2", "P22_NSCOL15P_SUP34", "P22_NSCOL15P_SUP5"]
edu["pct_higher_ed"] = edu[sup].sum(axis=1) / edu["P22_NSCOL15P"]

# 3. make sure both code columns are zero-padded strings
#    (rename the shapefile's code column to "CODGEO" first — adjust the left name)
communes = communes.rename(columns={communes.columns[1]: "CODGEO"})
print(communes.head())
print(edu.head())
communes["CODGEO"] = communes["CODGEO"].astype(str).str.zfill(5)
edu["CODGEO"] = edu["CODGEO"].astype(str).str.zfill(5)
print('ciao')
# 4. filter geometries to Île-de-France
idf = communes[communes["CODGEO"].str.startswith(
    ("75","77","78","91","92","93","94","95"))].copy()

# 5. merge education onto geometries
idf = idf.merge(edu[["CODGEO", "pct_higher_ed"]], on="CODGEO", how="left")

# 6. choropleth
fig, ax = plt.subplots(figsize=(12, 12))
idf.plot(
    column="pct_higher_ed",
    cmap="plasma_r",
    legend=True,
    ax=ax,
    edgecolor="grey",
    linewidth=0.3,
    missing_kwds={"color": "lightgrey"},
    legend_kwds={"label": "Share with higher education", "shrink": 0.6},
)
ax.set_axis_off()
ax.set_title("Share of population with higher education per commune — Île-de-France")
plt.tight_layout()
plt.show()

plt.savefig('prova.png')'''


'''# load dossier complet, keep CODGEO as string, only needed columns
cols = ["CODGEO", "P22_NSCOL15P",
        "P22_NSCOL15P_SUP2", "P22_NSCOL15P_SUP34", "P22_NSCOL15P_SUP5"]
dossier = pd.read_csv("geo_ses_files/FR/dossier_complet.csv", sep=";",
                      dtype={"CODGEO": str}, usecols=cols)

# filter to Île-de-France
idf = dossier[dossier["CODGEO"].str.startswith(
    ("75","77","78","91","92","93","94","95"))].copy()

# compute % higher education
sup_cols = ["P22_NSCOL15P_SUP2", "P22_NSCOL15P_SUP34", "P22_NSCOL15P_SUP5"]
idf["pct_higher_ed"] = idf[sup_cols].sum(axis=1) / idf["P22_NSCOL15P"]

print(idf)'''
'''import os
os.environ["SHAPE_RESTORE_SHX"] = "YES"'''

# ile de france users
#df = pl.read_parquet('geo_ses_files/FR/user_geo_income_200m_threshold_0.7_PARIS_LARGE_LARGE.parquet')


# 1) EXTRACT USERS TO PASS TO TEST -> SELECT 30 TWEETS PER USER
'''
# train test dev users
train_users = pl.read_csv('dev_phase_normal/input with medium large/train_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv')
test_users = pl.read_csv('dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv')
dev_users = pl.read_csv('dev_phase_normal/input with medium large/test_dev_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv')

# SES df

# convert to same type
df = df.with_columns(pl.col("user_id").cast(pl.Utf8))
train_users = train_users.with_columns(pl.col("user_id").cast(pl.Utf8))
dev_users = dev_users.with_columns(pl.col("user_id").cast(pl.Utf8))
test_users = test_users.with_columns(pl.col("user_id").cast(pl.Utf8))

# keep only users with at least 30 tweets
df = df.filter(pl.col('num_tweets_per_user') >= 30)

# remove users that are in train and dev sets
exclude_ids = pl.concat([
    train_users.select("user_id"),
    dev_users.select("user_id"),
]).unique()

# keep rows in df whose user_id is NOT in that set
filtered = df.filter(~pl.col("user_id").is_in(exclude_ids["user_id"]))

# expand list columns
filtered_expanded = filtered.explode(['tweets_ids', 'clean_text_light_list'])

# get 30 random tweets per user
filtered_expanded = filtered_expanded.with_columns(
    pl.Series(
        "random",
        np.random.random(filtered_expanded.height)
    )
)

df_30 = (
    filtered_expanded
    .sort(["user_id", "random"])
    .group_by("user_id", maintain_order=True)
    .head(30)
    .drop("random")
)
print(df_30)


# save df of ile de france users with 30 tweets each
# rename colums
df_30 = df_30.rename({'tweets_ids': 'tweet_id', 'clean_text_light_list': 'text'})
df_30 = df_30.select(['user_id', 'tweet_id', 'text', 'idINSPIRE', 'ind_snv_mean', 'ind'])

print(df_30)

# save
df_30.write_csv('dev_phase_normal/input with medium large/TOT_paris_LARGE_LARGE.csv')'''



# 2) ASSIGN COMMUNE TO USERS

# select model type
model = 'BERT' # BERT | LLAMA | LLAMA_SIMPLE

# select set of users
users = 'TOTAL' # TOTAL | TEST

# 1) read files

# read communes df (thsi file has communes and the commune of Paris is divided into its arrondissements)
communes = gpd.read_file("geo_ses_files/FR/revenu-des-francais-a-la-commune.shp", encoding="ISO-8859-1")   # from data.gouv.fr
# INSEE files use ; separator and latin-1 encoding often
income = pd.read_csv("geo_ses_files/FR/FILO2018_DISP_COM.csv", sep=";")
# users geo df
users_geo_df = gpd.read_parquet('geo_ses_files/FR/user_home_locations_and_attributes_200m.parquet')
# education data (dossier complet), compute % higher ed
cols = ["CODGEO", "P22_NSCOL15P", "P22_NSCOL15P_SUP2", "P22_NSCOL15P_SUP34", "P22_NSCOL15P_SUP5", "C22_POP15P_STAT_GSEC13_23", "C22_POP15P"]
edu_occ = pd.read_csv("geo_ses_files/FR/dossier_complet.csv", sep=";",
                  dtype={"CODGEO": str}, usecols=cols)

# predicted ses file
# bert total ile de france users
if model == 'BERT' and users == 'TOTAL':
    ses_pred_df = pd.read_csv('predictions_pre_avg_v2_ILE_DE_FRANCE_TOT/classes_2_2_30/len_vocab_neg_plur_mean/run_1/len_vocab_neg_plur_mean.csv')#[['user_id', ]]
# bert
elif model == 'BERT' and users == 'TEST':
    ses_pred_df = pd.read_csv('predictions_pre_avg_v2_ILE_DE_FRANCE/classes_2_2_30/len_vocab_neg_plur_mean/run_1/len_vocab_neg_plur_mean.csv')#[['user_id', ]]
# llama
elif model == 'LLAMA' or model == 'LLAMA_SIMPLE':
    ses_pred_df = pd.read_csv('predictions_llama/classes_2_2/llama_70B_zero_predictions_only_text_prompt_simple_2_2_classes.csv')#llama_70B_zero_predictions_all_prompt_specific_all_features_2_2_classes
    ses_pred_df = ses_pred_df.rename(columns={'pred': 'label'})
    ses_pred_df['label'] = ses_pred_df['label'].replace({'low': 0, 'high': 1})

# true ses file
# bert total ile de france test
if model == 'BERT' and users == 'TOTAL':
    ses_true_df = pd.read_csv('dev_phase_normal/input with medium large/TOT_paris_LARGE_LARGE_tot_errors_2_classes.csv')[['user_id', 'SES_users']]
# bert and llama test
elif (model == 'BERT' or model == 'LLAMA' or model == 'LLAMA_SIMPLE') and users == 'TEST':
    ses_true_df = pd.read_csv('dev_phase_normal/input with medium large/test_test_paris_LARGE_LARGE_tot_errors_with_medium_2_classes.csv')[['user_id', 'SES_users']]


# 2) prepare communes df

# rename column bcs has weird name
communes = communes.rename(columns={communes.columns[1]: "code"})
# extract ile de france communes
communes_idf = communes[communes["code"].str.startswith(
    ("75","77","78","91","92","93","94","95"))]
communes_idf = communes_idf[['code', 'geometry']]


# 3) prepare income, education, occupation df

# income
# force CODGEO to a 5-char zero-padded string
income["CODGEO"] = income["CODGEO"].astype(str).str.zfill(5)
# extract ile de france communes
idf_codes = ("75","77","78","91","92","93","94","95")
income_idf = income[income["CODGEO"].str.startswith(idf_codes)].copy()

# education
sup = ["P22_NSCOL15P_SUP2", "P22_NSCOL15P_SUP34", "P22_NSCOL15P_SUP5"]
edu_occ["pct_higher_ed"] = edu_occ[sup].sum(axis=1) / edu_occ["P22_NSCOL15P"]

# occupation
edu_occ["pct_cadres"] = edu_occ["C22_POP15P_STAT_GSEC13_23"] / edu_occ["C22_POP15P"]

# 4 ) merge communes and income df

# merge communes geometry with the income data (Q218), education, occupation data 
# commune code is "code" in shape file, "CODGEO" in the INSEE file
communes_idf = communes_idf.merge(
    income_idf[["CODGEO", "Q218"]],
    left_on="code", right_on="CODGEO",
    how="left"
)
communes_idf = communes_idf.merge(
    edu_occ[["CODGEO", "pct_higher_ed", "pct_cadres"]], 
    left_on="CODGEO", right_on="CODGEO",
    how="left"
)



# 5) prepare users df -> we get home geometry to assign users to right commune
users_geo_df = users_geo_df.rename(columns={'userId': 'user_id'})
users_geo_df = users_geo_df.set_geometry('home_geometry')
# extract only Ile de France users
# ile de france users
ile_users_df = pd.read_parquet('geo_ses_files/FR/user_geo_income_200m_threshold_0.7_PARIS_LARGE_LARGE.parquet')[['user_id']]
users_geo_df = pd.merge(users_geo_df, ile_users_df, on='user_id', how='right')

# make sure both layers share the same CRS
users_geo_df = users_geo_df.to_crs(communes_idf.crs)


# 6) spatial join: assign each user to the commune containing its point
joined = gpd.sjoin(users_geo_df, communes_idf, how="left", predicate="within")


# we now have, for each user: x and y home location, 200x200 m2 cell, commune

# count users per commune
counts_commune = (
    joined.groupby("code")          # commune INSEE code
          .size()
          .reset_index(name="n_users")
)
# optional: attach commune names and sort #, "nom"
counts_commune = counts_commune.merge(
    communes_idf[["code"]].drop_duplicates(), on="code", how="left"
).sort_values("n_users", ascending=False)

# count users per 200 cell
counts_200_cell = (
    joined.groupby('carreau_geometry')
        .size()
        .reset_index(name='n_users')
)


# 7) add labels for Ile de France users
print(ses_pred_df)
# pred ses
ses_pred_df['user_id'] = ses_pred_df['user_id'].astype(str)
users_geo_pred_df = pd.merge(users_geo_df, ses_pred_df, on='user_id')

# true ses for
ses_true_df = ses_true_df.rename(columns={'SES_users': 'label'})
ses_true_df["label"] = ses_true_df["label"].replace({"lower": 0, "high": 1})
ses_true_df['user_id'] = ses_true_df['user_id'].astype(str)
ses_true_df = ses_true_df.groupby('user_id').agg(label=pd.NamedAgg(column='label', aggfunc='first')).reset_index()
users_geo_true_df = pd.merge(users_geo_df, ses_true_df, on='user_id')


# 8) spatial join with labels
joined_pred = gpd.sjoin(users_geo_pred_df, communes_idf, how="left", predicate="within")
joined_true = gpd.sjoin(users_geo_true_df, communes_idf, how="left", predicate="within")



# plot choropleth of income per commune -> all in ile de france
'''fig, ax = plt.subplots(figsize=(9, 7))
communes_idf.plot(
    column="Q218",
    cmap="plasma_r",
    legend=True,
    ax=ax,
    edgecolor="grey",
    linewidth=0.3,
    missing_kwds={"color": "#ECECEC", "label": "No data"},
    legend_kwds={"label": "Median disposable income (€)", "shrink": 0.6},
)
cbar = ax.get_figure().axes[-1]        # the colorbar is the last axis
cbar.set_ylabel("Median disposable income (€)", fontsize=22)   # label size
cbar.tick_params(labelsize=15)         
ax.set_axis_off()
#ax.set_title("Median income per commune — Île-de-France (FiLoSoFi 2018)")
plt.tight_layout()
plt.show()
# save
plt.savefig(f'maps_plot/choro_income_commune_{users}.png')'''

# plot choropleth of education per commune -> all in ile de france
'''fig, ax = plt.subplots(figsize=(12, 12))
communes_idf.plot(
    column="pct_higher_ed",
    cmap="plasma_r",
    legend=True,
    ax=ax,
    edgecolor="grey",
    linewidth=0.3,
    missing_kwds={"color": "#ECECEC"},
    legend_kwds={"label": "Share with higher education", "shrink": 0.6},
)
ax.set_axis_off()
ax.set_title("Share of population with higher education per commune — Île-de-France")
plt.tight_layout()
plt.show()

plt.savefig(f'maps_plot/choro_education_commune_{users}.png')'''

'''# plot choropleth of occupation per commune -> all in ile de france
fig, ax = plt.subplots(figsize=(12, 12))
communes_idf.plot(
    column="pct_higher_ed",
    cmap="plasma_r",
    legend=True,
    ax=ax,
    edgecolor="grey",
    linewidth=0.3,
    missing_kwds={"color": "#ECECEC"},
    legend_kwds={"label": "Share with higher occupation", "shrink": 0.6},
)
ax.set_axis_off()
ax.set_title("Share of population with higher occupation per commune — Île-de-France")
plt.tight_layout()
plt.show()'''

# count fraction of users per commune with label == 1 (high ses)

# pred ses
frac_rich_pred = (
    joined_pred.groupby("code")
       .agg(
           n_users=("label", "size"),
           n_rich=("label", "sum"),          # label is 0/1, so sum = count of rich
       )
       .assign(frac_rich=lambda d: d["n_rich"] / d["n_users"])
       .reset_index()
)
# filter communes with at lest n users
n_users = 10
frac_rich_pred = frac_rich_pred[frac_rich_pred["n_users"] >= n_users]

# true ses
frac_rich_true = (
    joined_true.groupby("code")
       .agg(
           n_users=("label", "size"),
           n_rich=("label", "sum"),          # label is 0/1, so sum = count of rich
       )
       .assign(frac_rich=lambda d: d["n_rich"] / d["n_users"])
       .reset_index()
)
# filter communes with at lest n users
n_users = 10
frac_rich_true = frac_rich_true[frac_rich_true["n_users"] >= n_users]

# merge the fraction onto the commune polygons
# pred
communes_frac_pred = communes_idf.merge(frac_rich_pred, how="left")
# true
communes_frac_true = communes_idf.merge(frac_rich_true, how="left")


# merge income df and users df to extract only communes with users inside
communes_with_users_pred = communes_idf[['code', 'Q218', 'pct_higher_ed', 'pct_cadres']].merge(frac_rich_pred)
communes_with_users_true = communes_idf[['code', 'Q218', 'pct_higher_ed', 'pct_cadres']].merge(frac_rich_true)

print(communes_with_users_pred)

# re-merge with communes to get all the geometries, but income only of the ones with users
communes_income_users_pred = communes_with_users_pred.merge(communes_idf[['code', 'geometry']], on='code', how='right')
communes_income_users_true = communes_with_users_true.merge(communes_idf[['code', 'geometry']], on='code', how='right')

communes_income_users_pred = gpd.GeoDataFrame(
    communes_income_users_pred,
    geometry="geometry",       # name of the geometry column
    crs="EPSG:4326"            # or whatever CRS your geometries are in
)

communes_income_users_true = gpd.GeoDataFrame(
    communes_income_users_true,
    geometry="geometry",       # name of the geometry column
    crs="EPSG:4326"            # or whatever CRS your geometries are in
)

print(communes_income_users_pred)
print('ciao')
# plot choropleth of income per commune -> only communes with users in ile de france
fig, ax = plt.subplots(figsize=(9, 7))
communes_income_users_pred.plot(
    column="Q218",
    cmap="plasma_r",
    legend=True,
    ax=ax,
    edgecolor="grey",
    linewidth=0.3,
    missing_kwds={"color": "#ECECEC", "label": "No data"},
    legend_kwds={"label": "Median disposable income (€)", "shrink": 0.6, "pad": 0.01},
)
cbar = ax.get_figure().axes[-1]        # the colorbar is the last axis
'''pos = cbar.get_position()
cbar.set_position([
    pos.x0 -0.7,  # move left
    pos.y0,
    pos.width,
    pos.height
])'''
cbar.set_ylabel("Median disposable income (€)", fontsize=22)   # label size
cbar.tick_params(labelsize=15)         
ax.set_axis_off()
#ax.set_title("Median income per commune — Île-de-France (FiLoSoFi 2018)")
plt.tight_layout()
plt.show()
# save
plt.savefig(f'maps_plot/choro_income_commune_with_users_{users}.png')

# plot choropleth of education per commune -> only communes with users in ile de france
fig, ax = plt.subplots(figsize=(9, 7))
communes_income_users_pred.plot(
    column="pct_higher_ed",
    cmap="plasma_r",
    legend=True,
    ax=ax,
    edgecolor="grey",
    linewidth=0.3,
    missing_kwds={"color": "#ECECEC", "label": "No data"},
    legend_kwds={"shrink": 0.6, "pad": 0.01},
)
cbar = ax.get_figure().axes[-1]        # the colorbar is the last axis
label = "Fraction of inhabitants with" + chr(10) + "higher educational level"
cbar.set_ylabel(label, fontsize=22)   # label size
cbar.tick_params(labelsize=15)         
ax.set_axis_off()
#ax.set_title("Median income per commune — Île-de-France (FiLoSoFi 2018)")
plt.tight_layout()
plt.show()
# save
plt.savefig(f'maps_plot/choro_education_commune_with_users_{users}.png')


# plot choropleth of education per commune -> only communes with users in ile de france
fig, ax = plt.subplots(figsize=(9, 7))
communes_income_users_pred.plot(
    column="pct_cadres",
    cmap="plasma_r",
    legend=True,
    ax=ax,
    edgecolor="grey",
    linewidth=0.3,
    missing_kwds={"color": "#ECECEC", "label": "No data"},
    legend_kwds={"shrink": 0.6, "pad": 0.01},
)
cbar = ax.get_figure().axes[-1]        # the colorbar is the last axis
label = "Fraction of executives/" + chr(10) + "professionals (cadres)"
cbar.set_ylabel(label, fontsize=22)   # label size
cbar.tick_params(labelsize=15)         
ax.set_axis_off()
#ax.set_title("Median income per commune — Île-de-France (FiLoSoFi 2018)")
plt.tight_layout()
plt.show()
# save
plt.savefig(f'maps_plot/choro_occupation_commune_with_users_{users}.png')


# plot choropleth of the fraction of high ses users per commune

# pred

fig, ax = plt.subplots(figsize=(9, 7))

communes_frac_pred.plot(
    column="frac_rich",
    cmap="plasma_r",              # or "RdYlBu_r", "plasma"
    legend=True,
    ax=ax,
    edgecolor="grey",
    linewidth=0.3,
    missing_kwds={               # style for communes with no data (NaN)
        "color": "#ECECEC",
        "label": "< 10 users",
    },
    legend_kwds={"label": "Fraction of predicted higher-SES users", "shrink": 0.6, "pad":0.01},
)
cbar = ax.get_figure().axes[-1]        # the colorbar is the last axis
cbar.set_ylabel("Fraction of predicted higher-SES users", fontsize=22)   # label size
cbar.tick_params(labelsize=15)         
ax.set_axis_off()
#ax.set_title("Fraction of higher-SES users per commune (Île-de-France)")
plt.tight_layout()
plt.show()

plt.savefig(f'maps_plot/choro_commune_frac_pred_{users}_{model}.png')


# true

fig, ax = plt.subplots(figsize=(9, 7))

communes_frac_true.plot(
    column="frac_rich",
    cmap="plasma_r",              # or "RdYlBu_r", "plasma"
    legend=True,
    ax=ax,
    edgecolor="grey",
    linewidth=0.3,
    missing_kwds={               # style for communes with no data (NaN)
        "color": "#ECECEC",
        "label": "< 10 users",
    },
    legend_kwds={"label": "Fraction of observed higher-SES users", "shrink": 0.6, "pad":0.01},
)
cbar = ax.get_figure().axes[-1]        # the colorbar is the last axis
cbar.set_ylabel("Fraction of observed higher-SES users", fontsize=22)   # label size
cbar.tick_params(labelsize=15)   
ax.set_axis_off()
#ax.set_title("Fraction of higher-SES users per commune (Île-de-France)")
plt.tight_layout()
plt.show()

plt.savefig(f'maps_plot/choro_commune_frac_true_{users}_{model}.png')


# scatterplot fraction higher-SES users pred vs higher-SES users true -> per commune

# merge pred and true df
'''communes_income_users_pred = communes_income_users_pred.rename(columns={'frac_rich': 'frac_rich_pred'})
communes_income_users_true = communes_income_users_true.rename(columns={'frac_rich': 'frac_rich_true'})
communes_income_users_tot = pd.merge(communes_income_users_pred, communes_income_users_true, on='code')

print(communes_income_users_tot)
# drop NaNs for the correlation
valid_pred = communes_income_users_tot[["frac_rich_true", "frac_rich_pred"]].dropna()

r, p = spearmanr(valid_pred["frac_rich_true"], valid_pred["frac_rich_pred"])

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(valid_pred["frac_rich_true"], valid_pred["frac_rich_pred"], alpha=0.5, s=20)

# trend line
z = np.polyfit(valid_pred["Q218"], valid_pred["frac_rich"], 1)
x_line = np.linspace(valid_pred["Q218"].min(), valid_pred["Q218"].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5)

ax.set_xlabel("Commune median income (€)", fontsize=25)
ax.set_ylabel("Fraction of higher-SES users", fontsize=25)
ax.tick_params(axis='both', labelsize=17)
#ax.set_title(f"Spearman r = {r:.3f} (p = {p:.1e})")
plt.tight_layout()
plt.show()

plt.savefig(f'maps_plot/scatter_income_ses_pred_true_{users}_{model}.png')'''

# scatterplot fraction higher-SES users vs income -> per commune

# pred

# drop NaNs for the correlation
valid_pred = communes_income_users_pred[["Q218", "frac_rich"]].dropna()

r, p = spearmanr(valid_pred["Q218"], valid_pred["frac_rich"])

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(valid_pred["Q218"], valid_pred["frac_rich"], alpha=0.5, s=20)

# trend line
z = np.polyfit(valid_pred["Q218"], valid_pred["frac_rich"], 1)
x_line = np.linspace(valid_pred["Q218"].min(), valid_pred["Q218"].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5)

ax.set_xlabel("Commune median income (€)", fontsize=25)
ax.set_ylabel("Fraction of higher-SES users", fontsize=25)
ax.tick_params(axis='both', labelsize=17)
#ax.set_title(f"Spearman r = {r:.3f} (p = {p:.1e})")
plt.tight_layout()
plt.show()

plt.savefig(f'maps_plot/scatter_income_ses_pred_{users}_{model}.png')


# true

# drop NaNs for the correlation
'''valid_true = communes_income_users_true[["Q218", "frac_rich"]].dropna()

r, p = spearmanr(valid_true["Q218"], valid_true["frac_rich"])

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(valid_true["Q218"], valid_true["frac_rich"], alpha=0.5, s=20)

# trend line
z = np.polyfit(valid_true["Q218"], valid_true["frac_rich"], 1)
x_line = np.linspace(valid_true["Q218"].min(), valid_true["Q218"].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5)

ax.set_xlabel("Commune median income (€)", fontsize=25)
ax.set_ylabel("Fraction of higher-SES users", fontsize=25)
ax.tick_params(axis='both', labelsize=17)
#ax.set_title(f"Spearman r = {r:.3f} (p = {p:.1e})")
plt.tight_layout()
plt.show()

plt.savefig(f'maps_plot/scatter_income_ses_true_{users}_{model}.png')'''


# scatterplot fraction higher-SES users vs education -> per commune

# pred

# drop NaNs for the correlation
valid_pred = communes_income_users_pred[["pct_higher_ed", "frac_rich"]].dropna()

r, p = spearmanr(valid_pred["pct_higher_ed"], valid_pred["frac_rich"])

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(valid_pred["pct_higher_ed"], valid_pred["frac_rich"], alpha=0.5, s=20)

# trend line
z = np.polyfit(valid_pred["pct_higher_ed"], valid_pred["frac_rich"], 1)
x_line = np.linspace(valid_pred["pct_higher_ed"].min(), valid_pred["pct_higher_ed"].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5)

ax.set_xlabel("Commune educational level", fontsize=25)
ax.set_ylabel("Fraction of higher-SES users", fontsize=25)
ax.tick_params(axis='both', labelsize=17)
ax.set_title(f"Spearman r = {r:.3f} (p = {p:.1e})")
plt.tight_layout()
plt.show()

plt.savefig(f'maps_plot/scatter_education_ses_pred_{users}_{model}.png')

# scatterplot fraction higher-SES users vs occupation -> per commune

# pred

# drop NaNs for the correlation
valid_pred = communes_income_users_pred[["pct_cadres", "frac_rich"]].dropna()

r, p = spearmanr(valid_pred["pct_cadres"], valid_pred["frac_rich"])

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(valid_pred["pct_cadres"], valid_pred["frac_rich"], alpha=0.5, s=20)

# trend line
z = np.polyfit(valid_pred["pct_cadres"], valid_pred["frac_rich"], 1)
x_line = np.linspace(valid_pred["pct_cadres"].min(), valid_pred["pct_cadres"].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5)

ax.set_xlabel("Commune occupational level", fontsize=25)
ax.set_ylabel("Fraction of higher-SES users", fontsize=25)
ax.tick_params(axis='both', labelsize=17)
ax.set_title(f"Spearman r = {r:.3f} (p = {p:.1e})")
plt.tight_layout()
plt.show()

plt.savefig(f'maps_plot/scatter_occupation_ses_pred_{users}_{model}.png')

'''# merge your counts back onto the commune polygons
communes_counts = idf.merge(counts, on="code", how="left")
communes_counts["n_users"] = communes_counts["n_users"].fillna(0)

fig, ax = plt.subplots(figsize=(12, 12))
communes_counts.plot(
    column="n_users",       # color by number of users
    cmap="plasma_r",
    legend=True,
    ax=ax,
    edgecolor="grey",
    linewidth=0.3,
)
ax.set_axis_off()
ax.set_title("Users per commune")
plt.show()


fig, ax = plt.subplots(figsize=(12, 12))
idf.plot(ax=ax, color="lightgrey", edgecolor="grey", linewidth=0.3, zorder=1)
#users_geo_df.plot(ax=ax, color="red", markersize=3, alpha=0.3, zorder=2)
ax.set_axis_off()
plt.show()

plt.savefig('prova_plot.png')'''

# Assuming df1, df2, df3 are your three DataFrames
communes_income_users_pred = communes_income_users_pred.rename(columns={'frac_rich': 'frac_rich_pred'})
communes_income_users_true = communes_income_users_true[['code', 'frac_rich']].rename(columns={'frac_rich': 'frac_rich_true'})
# merge pred and true data
communes_income_users_tot = pd.merge(communes_income_users_pred, communes_income_users_true, on='code')

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=300)
communes_income_users_tot = communes_income_users_tot.dropna()
print(communes_income_users_tot)
'''plot_regression(axes[0], communes_income_users_pred, 'Q218', 'frac_rich', 'Median disposable income (€)', 'Fraction of higher-SES users', 'Income')
plot_regression(axes[1], communes_income_users_pred, 'pct_higher_ed', 'frac_rich', 'Fraction of inhabitants with higher educational level', 'Fraction of higher-SES users', 'Education')
plot_regression(axes[2], communes_income_users_pred, 'pct_cadres', 'frac_rich', 'Fraction of executives/professionals (cadres)', 'Fraction of higher-SES users', 'Occupation')
'''
plot_regression(axes[0], communes_income_users_tot, 'frac_rich_true', 'frac_rich_pred', 'Fraction of observed higher-SES users', 'Fraction of predicted higher-SES users', 'Observed')
plot_regression(axes[1], communes_income_users_tot, 'Q218', 'frac_rich_pred', 'Median disposable income (€)', 'Fraction of predicted higher-SES users', 'Income')
plot_regression(axes[2], communes_income_users_tot, 'pct_higher_ed', 'frac_rich_pred', 'Fraction of inhabitants with\n higher educational level', 'Fraction of predicted higher-SES users', 'Education')
#plot_regression(axes[3], communes_income_users_tot, 'pct_cadres', 'frac_rich_pred', 'Fraction of \n executives/professionals (cadres)', 'Fraction of predicted higher-SES users', 'Occupation')

for ax in axes:
    ax.set_ylim(0, 1.0)


# Keep y-axis ticks on all plots, but only show the y-axis label on the leftmost plot
for ax in axes[1:]:
    ax.yaxis.set_tick_params(labelleft=True)  # Ensure ticks are present on the left of second and third subplots
    ax.set_ylabel('')  # Remove y-axis label on the second and third plots

plt.tight_layout()
plt.show()

plt.savefig('prova_1.png')