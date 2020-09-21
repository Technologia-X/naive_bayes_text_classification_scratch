import numpy as np
import math

class MultinomialNaiveBayes:

    def __init__(self, classes, tokenizer):
        self.tokenizer = tokenizer
        self.classes = classes

    def group_by_class(self, X, y):
        data = dict()
        for c in self.classes:
            data[c] = X[np.where(y == c)]
        return data
    
    def fit(self, X, y):
        self.n_class_items = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()

        n = len(X)
        grouped_data = self.group_by_class(X, y)

        for c, data in grouped_data.items():
            self.n_class_items[c] = len(data)
            self.log_class_priors[c] = math.log(self.n_class_items[c] / n)
            self.word_counts[c] = defaultdict(lambda: 0)

            for text in data:
                counts = Counter(self.tokenizer.tokenize(text))
                for word, count in counts.item():
                    if word no in self.vocab:
                        self.vocab.add(word)
                    
                    self.word_counts[c][word] += count
