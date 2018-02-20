import nltk
hypothesis = ['This', 'is', 'cat']
reference = ['This', 'is', 'cat']
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
print(BLEUscore)