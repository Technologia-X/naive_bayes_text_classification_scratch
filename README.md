# NAIVE BAYES FROM SCRATCH

### Basic of Naive Bayes

![Image of Naive Bayes](https://miro.medium.com/max/792/1*9lA2z-Yz254sXCyHqYMoeQ.png)

[Multinomial Naive Bayes](https://github.com/fadilparves/naive_bayes_sentiment/blob/master/naive_bayes.py) to train and use to predict the sentiment (can be used for multi class text classification not just 1 and 0)

[Tokenizer](https://github.com/fadilparves/naive_bayes_sentiment/blob/master/tokenizer.py) to remove all html tags and special chars as well english stopwords

### Test Case

#### IMDB Movie Review Sentiment Analyzer

Naive Bayes used to predict whether the sentiment from a review is positive or negative (2 class predictor)

Accuracy (86%):

```
                precision    recall  f1-score   support

           0       0.89      0.81      0.85      2481
           1       0.83      0.90      0.86      2519

    accuracy                           0.86      5000
   macro avg       0.86      0.86      0.86      5000
weighted avg       0.86      0.86      0.86      5000
```

Confusion matrix

![Conf](https://github.com/fadilparves/naive_bayes_sentiment/blob/master/output/confusion_matrix.png)

460 wrong negative predicted on test data
262 wrong positive predicted on test data

At the end, test data was used to see how well the model can predict sentiment on unseen data

[Output](https://raw.githubusercontent.com/fadilparves/naive_bayes_sentiment/master/output/test_data_with_sentiment.csv) is here

### How to use

1. Make sure you have numpy, seaborn, pandas, scikit-learn and wordcloud installed (if not just pip install _libname_)
2. Clone the repo
3. Download new data from kaggle or anywhere or you can use the data provided in data folder
4. Run sentiment_predictor.py

## Contributor
<a href="https://github.com/fadilparves/naive_bayes_sentiment/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=fadilparves/naive_bayes_sentiment" />
</a>
