import pandas as pd
import seaborn as sns
import string
import re
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from naive_bayes import MultinomialNaiveBayes
from pylab import rcParams

sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 14,8
RANDOM_SEED = 66
np.random.seed(RANDOM_SEED)
nltk.download('stopwords')