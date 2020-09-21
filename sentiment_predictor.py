import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
from pylab import rcParams
from nltk.corpus import stopwords

from naive_bayes import MultinomialNaiveBayes
from tokenizer import Tokenizer

sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 14,8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

train = pd.read_csv("./data/imdb_review_train.tsv", delimiter="\t")
test = pd.read_csv("./data/imdb_review_test.tsv", delimiter="\t")

text = " ".join(review for review in train.review)

wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="white", stopwords=stopwords.words("english")).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("./output/wordcloud.png")

X = train['review'].values
y = train['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

MNB = MultinomialNaiveBayes(classes=np.unique(y), tokenizer=Tokenizer())
MNB.fit(X_train, y_train)

pred_sent = MNB.predict(X_test)
print(accuracy_score(y_test, pred_sent))

print(classification_report(y_test, pred_sent))

cnf_matrix = confusion_matrix(y_test, pred_sent)
class_names = ['negative', 'positive']
fig,ax = plt.subplots()

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Blues", fmt="d", cbar=False, xticklabels=class_names, yticklabels=class_names)
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.ylabel('Actual sentiment')
plt.xlabel('Predicted sentiment')
plt.savefig("./output/confusion_matrix.png")

test_pred = MNB.predict(test['review'].values)
test['sentiment'] = test_pred
test.to_csv("./output/test_data_with_sentiment.csv", index=False)