
import pandas as pd
pd.options.display.max_columns = 50

import re

import nltk
from nltk.corpus import stopwords # Import the stop word list
#nltk.download()  # Download text data sets, including stop words

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.stem import WordNetLemmatizer

from textblob import TextBlob

def text_and_title(text):
    """Add the title information to the beginning of the text"""
    try:
        x = text['Title']
        title = x + '. '
    except:
        title = 'NO_TITLE'
    try:
        result = title + ' ' + text['Text']
    except:
        result = title
    return (result)

def add_double_title(text):
    """Add the title information to the beginning of the text (and double it, called 'weighting up')"""
    try:
        x = text['Title']
        title = x + '. ' + x + '. '
    except:
        title = 'NO_TITLE'
    try:
        result = title + ' ' + text['Text']
    except:
        result = title
    return (result)

def letters_only(text, field):
    """Replace all non-alphanumeric characters with a space"""
    try:
        x = re.sub("[^a-zA-Z]",       # The pattern to search for
                   " ",               # The pattern to replace it with
                   text[field] )      # The text to search
    except:
        return ('byte_code_error_ignore_this_ record')
    return (x.lower())

def remove_stop_words(text, field, stopwords_set):
    """Remove stop words from the review text"""
    words = [w for w in text[field].split() if not w in stopwords_set]
    return( " ".join( words ))

def get_wordnet_pos(treebank_tag):
    """This is a helper function to translate part of speech (POS) for us in the make_lemmas function.
    nltk uses pos_tag to determine the POS of a word that is not compatible with the wordnet_lemmatizer."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def make_lemmas(text, field):#, stopwords_set):
    """Toeknizes all words in the review and then tags them with the part of speech (POS) they belong to
    as a tuple. Each tuple (word, pos) is then lemmatized before stop words are removed and the list is
    joined back into a single item/doc"""
    x = word_tokenize(text[field])
    #x = word_tokenize(x)
    x = nltk.pos_tag(x)
    wnl = WordNetLemmatizer()
    doc = []
    for word, part in x:
        #doc.append(wordnet_lemmatizer.lemmatize(word, pos=get_wordnet_pos(part)))
        doc.append(wnl.lemmatize(word, pos=get_wordnet_pos(part)))
    #words = [w for w in doc if not w.lower() in stopwords_set]
    #x = ( " ".join( words ))
    x = ( " ".join( doc ))
    return(x)

def create_neg_stops():
    """Combine the original list of stop words the negative suffix. For example: 'his', 'they', and 'me' become
    'his_neg', 'they_neg', and 'me_neg'."""
    orig_stops = stopwords.words("english")
    neg_stops = []
    for i in orig_stops:
        neg_stops.append(i+'_neg')
    orig_stops.extend(neg_stops)
    return(orig_stops)

def remove_negated_stop_words(text, field, neg_stops):
    """Remove all instances of stop words that have the '_neg' suffix."""
    # Make the text lowercase
    x = text[field]
    x = x.lower()
    
    stopwords_set = set(neg_stops)
    
    # List comprehension that splits the review into words and removes negative stop words
    words = [w for w in text[field].split() if not w in stopwords_set]
    return( " ".join( words ))

#/////////////////////ACCOUNT FOR NEGATION SENTIMENT///////////////////////////////////
def negatize(text, field):
    """Use the NLTK library's mark_negation to find negative words (like 'not' and 'nor') and append
    the '_neg' suffix to all words following the first negative word until it encounters a period or comma.
    Example: 'I don't like eating pizza, I love eating pizza' 
    becomes 'I don't like_neg eating_neg pizza_neg, I love pizza' """
    x = text[field]
    
    # The TextBlob class provides an easy way to split the reviews into sentences
    x = TextBlob(x)
    
    piece = []
    for sentence in x.sentences:
        
        # Split sentence on commas to ID phrases that need to be negated (if required)
        part = re.split(', ',str(sentence))
        for i in part:
            piece.append(mark_negation(i.split()))
    
    # Combine all terms/phrases back to one doc
    total = []
    for terms in piece:
        total.append(" ".join(terms))
    review = ''
    for phrase in total:
        review += phrase + ' '
    
    # mark_negation adds the _NEG suffix after the period, this catches those and fixes it
    review = review.replace("._NEG","_NEG.")
    review = review.lower()
    
    # return the entire entire review except for the last character which is always a space
    return (review[:-1])

def tag_scent(text, field):
    x = word_tokenize(text[field])
    scents = ['scent', 'scents', 'scented', 'smell', 'smells', 'smelled', 'odor', 'odors', 'fragrance', 'fragrances',
              'fragrant', 'aroma', 'aromas', 'perfume', 'perfumes', 'perfumed', 'whiff', 'fresh', 'freshness',
              'stench', 'stink', 'stinks', 'smelly', 'pungent', 'stinky', 'odoriferous']
    for word in x:
        if word.lower() in scents:
            return ('Yes')
    return ('No')

def tag_moisture(text, field):
    x = word_tokenize(text[field])
    moisture = ['moisture', 'moist', 'wet', 'dry', 'damp', 'dampish', 'wettish',  'drenched', 'dripping',
                'saturate', 'saturated', 'soaked', 'soaking', 'sodden', 'soggy', 'sopping', 'soppy',
                'arid', 'dryness', 'dried', 'waterless', 'bone-dry', 'dehydrated', 'ultradry', 'dank', 'supple',
                'quick-drying']
    for word in x:
        if word.lower() in moisture:
            return ('Yes')
    return ('No')

def tag_residue(text, field):
    x = word_tokenize(text[field])
    residue = ['streak', 'streaks', 'streaked', 'streaking', 'residue', 'film', 'cloudy', 'drop', 'drip', 'lines'
               'drips', 'dripped', 'gunk', 'tarnish', 'spot', 'spots', 'drip-streaks']
    for word in x:
        if word.lower() in residue:
            return ('Yes')
    return ('No')

def tag_value(text, field):
    x = word_tokenize(text[field])
    value = ['value', 'price', 'prices', 'worth', 'worthless', 'cost', 'costs', 'expense', 'expensive', 'costly',
             'overpriced', 'pricey', 'valuable', 'invaluable', 'cheap', 'economical', 'reasonable', 'inexpensive',
             'expenses', 'valued', 'priced', 'cheapen', 'economics', 'low-priced', 'bargain', 'low-cost', 'budget',
             'cheapest', 'bargains', 'money', 'budgeted', 'budgets']
    for word in x:
        if word.lower() in value:
            return ('Yes')
    return ('No')

def tag_sensitivity(text, field):
    x = word_tokenize(text[field])
    sensitivity = ['sensitivity', 'sensitive', 'skin', 'soft', 'softest', 'soften', 'rough', 'rougher', 'smooth', 'touch']
    for word in x:
        if word.lower() in sensitivity:
            return ('Yes')
    return ('No')

def tag_topics(text, field):
    x = word_tokenize(text[field])
    tags = {}
    scent_tag = []
    moisture_tag = []
    residue_tag = []
    value_tag = []
    sensitivity_tag = []
    scents = ['scent', 'scents', 'scented', 'smell', 'smells', 'smelled', 'odor', 'odors', 'fragrance', 'fragrances',
              'fragrant', 'aroma', 'aromas', 'perfume', 'perfumes', 'perfumed', 'whiff', 'fresh', 'freshness',
              'stench', 'stink', 'stinks', 'smelly', 'pungent', 'stinky', 'odoriferous']
    moisture = ['moisture', 'moist', 'wet', 'dry', 'damp', 'dampish', 'wettish',  'drenched', 'dripping',
                'saturate', 'saturated', 'soaked', 'soaking', 'sodden', 'soggy', 'sopping', 'soppy',
                'arid', 'dryness', 'dried', 'waterless', 'bone-dry', 'dehydrated', 'ultradry', 'dank', 'supple',
                'quick-drying']   
    residue = ['streak', 'streaks', 'streaked', 'streaking', 'residue', 'film', 'cloudy', 'drop', 'drip', 'lines'
               'drips', 'dripped', 'gunk', 'tarnish', 'spot', 'spots', 'drip-streaks']
    value = ['value', 'price', 'prices', 'worth', 'worthless', 'cost', 'costs', 'expense', 'expensive', 'costly',
             'overpriced', 'pricey', 'valuable', 'invaluable', 'cheap', 'economical', 'reasonable', 'inexpensive',
             'expenses', 'valued', 'priced', 'cheapen', 'economics', 'low-priced', 'bargain', 'low-cost', 'budget',
             'cheapest', 'bargains', 'money', 'budgeted', 'budgets', 'discount']
    sensitivity = ['sensitivity', 'sensitive', 'skin', 'soft', 'softest', 'soften', 'rough', 'smooth', 'touch']
    
    for word in x:
        if word.lower() in scents:
            scent_tag.append(word.lower())
        if word.lower() in moisture:
            moisture_tag.append(word.lower())
        if word.lower() in residue:
            residue_tag.append(word.lower())
        if word.lower() in value:
            value_tag.append(word.lower())
        if word.lower() in sensitivity:
            sensitivity_tag.append(word.lower())
    if len(scent_tag) > 0:
        tags['scent'] = scent_tag
    if len(moisture_tag) > 0:
        tags['moisture'] = moisture_tag
    if len(residue_tag) > 0:
        tags['residue'] = residue_tag
    if len(value_tag) > 0:
        tags['value'] = value_tag
    if len(sensitivity_tag) > 0:
        tags['sensitivity'] = sensitivity_tag
        
    if len(tags) == 0:
        return ('')
    return (tags)

def get_topics(text):
    topics = []
    if (text['scent']) == 'Yes':
        topics.append('scent')
    if (text['moisture']) == 'Yes':
        topics.append('moisture')
    if (text['residue']) == 'Yes':
        topics.append('residue')
    if (text['value']) == 'Yes':
        topics.append('value')
    if (text['sensitivity']) == 'Yes':
        topics.append('sensitivity')    
    return( ", ".join( topics ))
    
    def nouns_and_adjectives(text,field,stopwords_set):
    """Starts with the data that alread has had the stops removed. Tokenizes the text and uses nltk to
    tag the parts of speech. If the word has a part of speech that is not a noun or adjective, it is not 
    included in the result."""
    x = word_tokenize(text[field])
    #x = word_tokenize(x)
    x = nltk.pos_tag(x)
    doc = []
    for word, part in x:
        if part.startswith('J'):
            # This is an adjective
            doc.append(word)
        if part.startswith('N'):
            # This is a noun
            doc.append(word)
        #doc.append(wordnet_lemmatizer.lemmatize(word, pos=get_wordnet_pos(part)))
    words = [w for w in doc if not w.lower() in stopwords_set]
    x = ( " ".join( words ))
    return(x)
    
    from time import time
t0 = time()
t1 = time()
#/////////////////////READ THE DATA////////////////////////////////////////////////////
print ('Reading the data...')
wipes = pd.read_csv("home_products.csv", header=0, encoding="ISO-8859-1" )
sentences = pd.read_csv("only_sentences.csv", header=0, encoding="ISO-8859-1" )
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////ADD THE TITLE TO THE TEXT////////////////////////////////////////
t1 = time()
print ('Adding titles to text...')
wipes['text_and_title'] = wipes.apply(lambda text: text_and_title(text), axis=1)
wipes['double_title'] = wipes.apply(lambda text: add_double_title(text), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////REMOVE NON-ALPHANUMERICS AND PUNCTUATION/////////////////////////
t1 = time()
print ('Removing Non-Alphanumerics...')
wipes['text_and_title_no_stops'] = wipes.apply(lambda text: letters_only(text, 'text_and_title'), axis=1)
wipes['double_title_no_stops'] = wipes.apply(lambda text: letters_only(text, 'double_title'), axis=1)
sentences['lowercase_no_punctuation'] = sentences.apply(lambda text: letters_only(text, 'Sentence'), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////LOAD THE STOPWORDS PROVIDED BY NLTK//////////////////////////////
stop1 = ['wipes', 'wipe', 'love', 'like', 'good', 'nice', 'amazon', 'loved', 'loves', 'likes', 'liked']
stops = stopwords.words("english")
stops.extend(stop1)
stopwords_set = set(stops)

#/////////////////////REMOVE STOP WORDS///////////////////////////////////////////////
t1 = time()
print ('Removing Stop Words...')
wipes['text_and_title_no_stops'] = wipes.apply(lambda text: remove_stop_words(text, 'text_and_title_no_stops', stopwords_set), axis=1)
wipes['double_title_no_stops'] = wipes.apply(lambda text: remove_stop_words(text, 'double_title_no_stops', stopwords_set), axis=1)
sentences['lowercase_no_stops'] = sentences.apply(lambda text: remove_stop_words(text, 'lowercase_no_punctuation', stopwords_set), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////NEGATE TEXT AND TITLES///////////////////////////////////////////
#t1 = time()
#print ('Tagging Negative Text...')
#wipes['text_and_title_negation'] = wipes.apply(lambda text: negatize(text, 'text_and_title'), axis=1)
#wipes['double_title_negation'] = wipes.apply(lambda text: negatize(text, 'double_title'), axis=1)
#print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////REMOVE NEGATIVE STOPS////////////////////////////////////////////
#t1 = time()
#print ('Removing Negative Stop Words...')
#wipes['text_and_title_negation_no_stops'] = wipes.apply(lambda text: remove_negated_stop_words(text, 'text_and_title_negation', create_neg_stops()), axis=1)
#wipes['double_title_negation_no_stops'] = wipes.apply(lambda text: remove_negated_stop_words(text, 'double_title_negation', create_neg_stops()), axis=1)
#print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////LEMMATIZE THE TEXT REVIEWS///////////////////////////////////////
t1 = time()
print ('Lemmatizing The Reviews...')
wipes['lemma_text_title_no_stops'] = wipes.apply(lambda text: make_lemmas(text, 'text_and_title_no_stops'), axis=1)
wipes['lemma_double_title_no_stops'] = wipes.apply(lambda text: make_lemmas(text, 'double_title_no_stops'), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

t1 = time()
print ('Lemmatizing The Sentences...')
sentences['lemmatized'] = sentences.apply(lambda text: make_lemmas(text, 'lowercase_no_stops'), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////NOUNS AND AJECTIVES ONLY/////////////////////////////////////////
t1 = time()
print ('Use only Nouns and Adjectives...')
wipes['nouns_and_adjectives'] = wipes.apply(lambda text: nouns_and_adjectives(text, 'Text', stopwords_set), axis=1)
sentences['nouns_and_adjectives'] = sentences.apply(lambda text: nouns_and_adjectives(text, 'Sentence', stopwords_set), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

#/////////////////////TAG TOPICS///////////////////////////////////////////////////////
t1 = time()
print ('Tagging topics for each review...')
wipes['scent'] = wipes.apply(lambda text: tag_scent(text, 'Text'), axis=1)
wipes['moisture'] = wipes.apply(lambda text: tag_moisture(text, 'Text'), axis=1)
wipes['residue'] = wipes.apply(lambda text: tag_residue(text, 'Text'), axis=1)
wipes['value'] = wipes.apply(lambda text: tag_value(text, 'Text'), axis=1)
wipes['sensitivity'] = wipes.apply(lambda text: tag_sensitivity(text, 'Text'), axis=1)
wipes['tags'] = wipes.apply(lambda text: tag_topics(text, 'Text'), axis=1)
wipes['topics'] = wipes.apply(lambda text: get_topics(text), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))

t1 = time()
print ('Tagging topics for each sentence...')
sentences['scent'] = sentences.apply(lambda text: tag_scent(text, 'Sentence'), axis=1)
sentences['moisture'] = sentences.apply(lambda text: tag_moisture(text, 'Sentence'), axis=1)
sentences['residue'] = sentences.apply(lambda text: tag_residue(text, 'Sentence'), axis=1)
sentences['value'] = sentences.apply(lambda text: tag_value(text, 'Sentence'), axis=1)
sentences['sensitivity'] = sentences.apply(lambda text: tag_sensitivity(text, 'Sentence'), axis=1)
sentences['tags'] = sentences.apply(lambda text: tag_topics(text, 'Sentence'), axis=1)
sentences['topics'] = sentences.apply(lambda text: get_topics(text), axis=1)
print("Finished In:     %0.3fs." % (time()-t1))


#/////////////////////WRITE THE DATA///////////////////////////////////////////////////
wipes.to_csv('home_products_additional_features.csv', index=False)
sentences.to_csv('sentences_additional_features.csv', index=False)

print("FINISHED: \nTime Elapsed:    %0.3fs." % (time() - t0))