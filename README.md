# Socioeconomic status inference from online socialmedia text

The aim of this work is to infer, through a classification task, the socioeconomic status (SES) of social media users from their text, not using any other additional metadata. We test two settings (2-class and 3-class setting) and two main strategies: a CamemBERT classifier and zero-shot classification with LLMs.


## Structure

```
├── Data preparation                       <- Scripts for location inference, text pre-processing, SES assignemnt, linguistic features extraction, preparation on train, dev,                                             
│   ├── add_features                          and test sets
│   ├── South America                              
│   ├── US
├── Analysis                               <- Scripts for preliminary statistics on data and analysis of the results
│   ├── Choroplets                              <- Notebooks to generate choroplet maps
│   │   ├── f_003_tweets_flux_EU.ipynb
│   │   ├── f_003_tweets_flux_SA.ipynb
│   │   ├── f_003_tweets_flux_US_no_WA.ipynb
│   ├── GDP VS tweets                           <- Notebook to generate GDP per capita VS tweets per day per country
│   │   ├── f_001_figure_GDP_tweets.ipynb                
│   ├── Gravity model                           <- Notebooks to run gravity models and get corresponding plots
│   │   ├── f_002_gravity model_EU_img.ipynb    
│   │   ├── f_002_gravity model_SA_img.ipynb
│   │   └── f_002_gravity model_US_img.ipynb
│   ├── Network measures                        <- Notebooks to compute network measures and generate network images
│   │   ├── EU_network_measures.ipynb
│   │   ├── SA_network_measures.ipynb
│   │   └── US_network_measures no wa.ipynb
│   └── Topics                                  <- Scripts for embeddings generation, topic modeling, and generation of labels of topics
│       ├── embeddings.py                       
│       ├── topics.py
|       └── labels_llm.py                                           
├── SES classification                             <- Scripts for the SES classification: CamemBERT and LLMs (Llama and GPT)
│   ├── main_geocoding.py
│   ├── new_ner.py
│   ├── nominatim_ner_geocoding.py
│   ├── preprocessing.py
│   ├── tweets_dataset.py
│   ├── tweets_flux.py
│   └── twitter_api.py
├── Figures                                <- Generated figures
├── README.md
└── requirements.txt
```


## Installation

1) Clone the repository
2) In the cloned folder, create a virtual environment through the command `conda create -n ses_venv python=3.10` and activate it through `conda activate ses_venv`
3) Inside the virtual environment, install the requirements through `pip install -r requirements.txt`
4) To run the scripts inside the folder `SES classification`, a GPU is required (apart from the file *gpt_zero_shots.py*)


## Execution

### Data preparation

1) Home location inference
2) Text pre-processing, SES assignment, users sampling to create train, dev, and test sets
    - Script `prepare_data.py`
3) Extraction of linguistic features
    - Scripts `lang_tool.py` + `create_tot_langtool_df.py` + `extract_merge_langtool_tweets.py` for LanguageTool error features extraction
    - Script `add_features.py` to add other linguistic features (tweet length, vocabulary size, non-standard negation, and non-standard pluralization)
4) Build prompts for LLMs
    - Script `build_user_prompts.py`

### SES classification

1) Classification with CamemBERT
    - Script `normal_training_pooling_2_classes.py` to perform train and test with the *a priori* user aggregation strategy in the 2-class setting
    - Script `normal_training_pooling_3_classes.py` to perform train and test with the *a priori* user aggregation strategy in the 3-class setting
    - Script `normal_training_classification_2_classes.py` to perform train and test with the *a posteriori* user aggregation strategy in the 2-class settting
    - Script `normal_training_classification_3_classes.py` to perform train and test with the *a posteriori* user aggregation strategy in the 3-class settting
2) Classification with LLMs
    - Script `llama_zero_shot.py` to perform zero-shot classification with Llama family models
    - Script `gpt_zero_shot.py` to perform zero-shot classification with GPT family models

### Data and results analysis

1) Data analysis
    - Script `statistics.py` to compute and plot correlation between linguistic features and users income
2) Results analysis
    - Script `compute_score.py` to compute macro-F1 and accuracy score
    - Script `confusion_grid.py` to compute confusion metric of LLMs results
    - Script `test_num_users.py` to compute and plot scores with different number of tweets per user at test time
    - Script `plot_maps.py` to plot maps of users coloured by their SES