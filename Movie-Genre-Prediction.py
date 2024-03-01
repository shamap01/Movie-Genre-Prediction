import random
import spacy
from spacy.util import minibatch, compounding

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in doc]
    return ' '.join(tokens)

def get_labels():
    labels = ['action', 'comedy', 'drama', 'romance']
    return labels

def train_model(train_data, dev_data, labels, iterations=20):
    textcat = nlp.create_pipe('textcat', config={'architecture': 'simple_cnn'})
    nlp.add_pipe(textcat, last=True)
    texts, cats = zip(*train_data)
    textcat.add_labels(labels)
    X_train = nlp.pipe(texts, batch_size=1000, disable=['tagger', 'parser', 'ner'])
    y_train = [cat for _, cat in train_data]
    print('Training model...')
    for i in range(iterations):
        optimizer = nlp.begin_training()
        batch_size = min(len(X_train), 1000)
        batches = minibatch(list(zip(X_train, y_train)), size=batch_size)
        for batch in batches:
            texts, labels = zip(*batch)
            nlp.update(texts, labels, drop=0.5, sgd=optimizer, losses=compounding(0.0001, 0.002, 1.001))
    print('Evaluation model...')
    X_dev = nlp.pipe(dev_data, disable=['tagger', 'parser', 'ner'])
    y_dev = [cat for _, cat in dev_data]
    scores = {'cats': {x: 0 for x in labels}, 'total_correct': 0, 'total_samples': 0}
    for text, label in zip(X_dev, y_dev):
        pred = textcat(text)
        guess = pred.cats[pred.cats.argmax()]
        if guess == label:
            scores['total_correct'] += 1
        scores['cats'][label] += 1
        scores['total_samples'] += 1
    for label in labels:
        print(f'Label: {label} - Accuracy: {scores["cats"][label] / scores["total_samples"]}')
    print(f'Total accuracy: {scores["total_correct"] / scores["total_samples"]}')

def main():
    train_data = [
        ('A group of young friends on a camping trip decide to create the ultimate lamb sacrifice', 'horror'),
        ('A group of friends on a camping trip decide to create the ultimate lamb dinner', 'comedy'),
        ('A group of young friends on a camping trip decide to create the ultimate go-kart', 'action'),
        ('A group of friends on a camping trip decide to create the ultimate scavenger hunt', 'adventure'),
        ('A group of friends on a camping trip decide to create the ultimate bonfire', 'drama'),
    ]
    dev_data = [
        ('A group of friends on a camping trip decide to create the ultimate ghost story', 'horror'),
        ('A group of friends on a camping trip decide to create the ultimate love story', 'romance'),
        ('A group of friends on a camping trip decide to create the ultimate comedy sketch', 'comedy'),
        ('A group of friends on a camping trip decide to create the ultimate survival guide', 'adventure'),
    ]
    labels = get_labels()
    train_data = [(process_text(text), label) for text, label in train_data]
    dev_data = [(process_text(text), label) for text, label in dev_data]
    train_model(train_data, dev_
