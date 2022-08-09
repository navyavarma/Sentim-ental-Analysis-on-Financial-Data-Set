#library that contains punctuation
import pandas as pd
import string
import re
import nltk

string.punctuation

data_set=pd.read_csv(r'C:\Users\acer\Desktop\Sentiment Analysis on Finanicial Data-NLP\data\FinancialData.csv')

def remove_punctuation(text):
    pun_free=''.join([i for i in text if i not in string.punctuation])
    return pun_free

data_set['clean_msg']=data_set['review'].apply(lambda x: remove_punctuation(x))
data_set.head()

#Lowering the text
data_set['msg_lower']= data_set['clean_msg'].apply(lambda x: x.lower())

#Tokenization: In this step, the text is split into smaller units.
#defining function for tokenization
import re

def tokenization(text):
    tokens=re.split('W+',text)  #writing
    return tokens

#applying function to the column
data_set['msg_tokenied']=data_set['msg_lower'].apply(lambda x: tokenization(x))
data_set.head()


import nltk
nltk.download('stopwords')  #downloading stopwords

#Stop word removal: Stopwords are the commonly used words and are removed from the text as they do not add any value to the analysis. These words carry less or no meaning.
import nltk
from nltk.corpus import stopwords
#Stop words present in the library
stopwords = stopwords.words('english')
len(stopwords)

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output
data_set['no_stopwords']=data_set['msg_tokenied'].apply(lambda x: remove_stopwords(x))
data_set.head()


#Stemming: It is also known as the text standardization step where the words are stemmed or diminished to their root/base form.
#importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer
#defining the object for stemming
porter_stemmer = PorterStemmer()

#defining a function for stemming
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

data_set['msg_stemmed']=data_set['no_stopwords'].apply(lambda x: stemming(x))
data_set.head()


