#!pip install h5py
#!pip install typing-extensions
#!pip install wheel
#!pip install flair
#!pip install transformers == 4.30.2
# with larger model
#*** ENSURE YOU PIP INSTALL 'transformers==4.30.2' ****
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.nn import Classifier
from transformers import XLMRobertaConfig, XLMRobertaModel,XLMRobertaTokenizer
import pandas as pd
import re
import nltk
import string
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#load model
#xlm_roberta_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
#xlm_roberta_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")


tweets = pd.read_csv('C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/combined.csv', low_memory=False, encoding='latin-1')

#delete na
tweets = tweets.dropna(subset=['Title'])
tweets = tweets.iloc[1:20]

#create functions
def engineer_sentence(entity, tag, tweet, index=0):
    indices_object = re.finditer(pattern=r"\s{}\s".format(entity), string=tweet)
    indices = [[index.start() + 1, index.start() + len(entity) + 1] for index in indices_object]
    if index < len(indices):
        i = indices[index]
        tweet = tweet[:i[0]] + '[' + tweet[i[0]:i[1]] + ']/{}'.format(tag) + tweet[i[1]:]
        tweet = re.sub(r'\/{}\/'.format(tag), '/', tweet)
        tweet = re.sub(r'\[\[', '[', tweet)
        tweet = re.sub(r'\]\]', ']', tweet)
        return engineer_sentence(entity, tag, tweet, index + 1)
    else:
        return tweet

def create_ner_df(lis):
    tweet_id_ = []
    entity_ = []
    tag_ = []
    dates_ = []
    engineered_sentences = []
    index = 0
    for tweet, date in zip(lis, tweets['Date']):
        print(index)
        # tokenizes sentence to preform different n-grams
        sentence = Sentence(tweet)

        # load the NER tagger
        tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")


        # run NER over sentence
        tagger.predict(sentence)
        replace = " " + sentence.to_original_text() + " "
        for label in sentence.get_labels():
            replace = engineer_sentence(label.data_point.text, label.value, replace)
            replace = re.sub(r'/(\w*)]', '', replace)
            tweet_id_.append(index)
            entity_.append(label.data_point.text)
            tag_.append(label.value)
            dates_.append(date)

        engineered_sentences.append(replace)
        index = index + 1

    engineered_sentences_df = {"labeled_setence": engineered_sentences}
    engineered_sentences_df = pd.DataFrame.from_dict(engineered_sentences_df, orient='index')
    engineered_sentences_df = engineered_sentences_df.transpose()

    count_df = {"tweet_id": tweet_id_, 'entity': entity_, 'tag': tag_, 'date': dates_}
    count_df = pd.DataFrame.from_dict(count_df, orient='index')
    count_df = count_df.transpose()
    return engineered_sentences_df, count_df

#run functions
engineered_sentences_df, count_df = create_ner_df(tweets['Title'].to_list())

count_df.to_csv('C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/raw NER/large_ner_output.csv', index = None)
engineered_sentences_df.to_csv('C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/raw NER/large_labelled_sentence_output.csv', index = None)
