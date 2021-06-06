

import string
from nltk.corpus import stopwords as nltk_stopwords

stopwords =  set(nltk_stopwords.words('english')) | set(string.punctuation)
