from __future__ import print_function
from time import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np

# Read in the data
reviews = pd.read_csv('home_products_additional_features.csv', header=0, encoding="ISO-8859-1" )

# Read in Reviews broken out by sentence
sentences = pd.read_csv('sentence_home_products_additional_features.csv', header=0, encoding="ISO-8859-1" )

def run_nmf(nmf_features, nmf_topics, nmf_top_words, nmf_data_samples, nmf_max_df, nmf_min_df, nmf_alpha, nmf_l1_ratio):
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=nmf_max_df, min_df=nmf_min_df,
                                       max_features=nmf_features,
                                       stop_words='english')

    tfidf = tfidf_vectorizer.fit_transform(nmf_data_samples)

    nmf = NMF(n_components=nmf_topics, random_state=1,
              alpha=nmf_alpha, l1_ratio=nmf_l1_ratio).fit(tfidf)

    print("\nTopics in NMF model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, nmf_top_words)
    
    def run_lda(lda_features, lda_topics, lda_top_words, lda_data_samples, lda_max_df, lda_min_df, lda_max_iter, lda_learning_offset):
    print("Fitting LDA models with tf features...")
    tf_vectorizer = CountVectorizer(max_df=lda_max_df, min_df=lda_min_df,
                                    max_features=lda_features,
                                    stop_words='english')
    
    tf = tf_vectorizer.fit_transform(lda_data_samples)
    
    lda = LatentDirichletAllocation(n_topics=lda_topics, max_iter=lda_max_iter,
                                    learning_method='online',
                                    learning_offset=lda_learning_offset,
                                    random_state=0)
    
    lda.fit(tf)
    
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, lda_top_words)    
    
    def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
    def remove_nulls(df, column):
    data = df.dropna(subset = [column])
    data = list(data[column])
    return data    
    
    #////////////////////////SET NMF PARAMETERS AND RUN MODEL/////////////////////////////

# Uncomment one of the lines (and only one line) that begins with nmf_data_samples to change the data set the model runs on
#nmf_data_samples = remove_nulls(reviews, 'Text')
#nmf_data_samples = remove_nulls(reviews, 'Title')
nmf_data_samples = remove_nulls(reviews, 'text_and_title')
#nmf_data_samples = remove_nulls(reviews, 'double_title')
#nmf_data_samples = remove_nulls(reviews, 'text_and_title_no_stops')
#nmf_data_samples = remove_nulls(reviews, 'double_title_no_stops')
#nmf_data_samples = remove_nulls(reviews, 'text_and_title_negation')
#nmf_data_samples = remove_nulls(reviews, 'double_title_negation')
#nmf_data_samples = remove_nulls(reviews, 'text_and_title_negation_no_stops')
#nmf_data_samples = remove_nulls(reviews, 'double_title_negation_no_stops')
#nmf_data_samples = remove_nulls(reviews, 'lemma_text_title_no_stops')
#nmf_data_samples = remove_nulls(reviews, 'lemma_double_title_no_stops')
#nmf_data_samples = remove_nulls(reviews, 'nouns_and_adjectives')

nmf_features = 12000      # Size of the vocabulary
nmf_topics = 75           # Number of topics
nmf_top_words = 9         # Words to include in the topic
nmf_max_df=0.95           # Ignore terms that have a doc frequency (percent or int) strictly higher than the given threshold
nmf_min_df=2              # Ignore terms that have a doc frequency (percent or int) strictly lower than the given threshold
nmf_alpha=.1              # Constant that multiplies the regularization terms. Set to zero for no regularization.
nmf_l1_ratio=.5           # Regularization mixing parameter.  0 <= l1_ratio <= 1

run_nmf(nmf_features, nmf_topics, nmf_top_words, nmf_data_samples, nmf_max_df, nmf_min_df, nmf_alpha, nmf_l1_ratio)

#////////////////////////SET LDA PARAMETERS AND RUN MODEL/////////////////////////////

# Uncomment one of the lines (and only one line) that begins with lda_data_samples to change the data set the model runs on

#lda_data_samples = remove_nulls(reviews, 'Text')
#lda_data_samples = remove_nulls(reviews, 'Title')
#lda_data_samples = remove_nulls(reviews, 'text_and_title')
#lda_data_samples = remove_nulls(reviews, 'double_title')
#lda_data_samples = remove_nulls(reviews, 'text_and_title_no_stops')
#lda_data_samples = remove_nulls(reviews, 'double_title_no_stops')
#lda_data_samples = remove_nulls(reviews, 'text_and_title_negation')
#lda_data_samples = remove_nulls(reviews, 'double_title_negation')
#lda_data_samples = remove_nulls(reviews, 'text_and_title_negation_no_stops')
#lda_data_samples = remove_nulls(reviews, 'double_title_negation_no_stops')
#lda_data_samples = remove_nulls(reviews, 'lemma_text_title_no_stops')

#lda_data_samples = remove_nulls(reviews, 'lemma_double_title_no_stops')
#lda_data_samples = remove_nulls(reviews_20, 'lemma_double_title_no_stops')

#lda_data_samples = remove_nulls(reviews, 'nouns_and_adjectives')
#lda_data_samples = remove_nulls(reviews_100, 'nouns_and_adjectives')


#lda_data_samples = remove_nulls(sentences, 'Sentence')
#lda_data_samples = remove_nulls(sentences, 'sentence_no_stops')
#lda_data_samples = remove_nulls(sentences, 'sentence_lemma_no_stops')
lda_data_samples = remove_nulls(sentences, 'sentence_nouns_and_adjectives')


lda_features = 12000      # Size of the vocabulary 
lda_topics = 75           # Number of topics
lda_top_words = 9         # Words to include in the topic
lda_max_df= 0.80          # Ignore terms that have a doc frequency (percent or int) strictly higher than the given threshold
lda_min_df= 15             # Ignore terms that have a doc frequency (percent or int) strictly lower than the given threshold
lda_max_iter=6            # Number of iterations to compute
lda_learning_offset=40.   # A parameter that downweights early iterations in online learning. Should be > 1

run_lda(lda_features, lda_topics, lda_top_words, lda_data_samples, lda_max_df, lda_min_df, lda_max_iter, lda_learning_offset)