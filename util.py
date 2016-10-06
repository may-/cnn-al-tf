# -*- coding: utf-8 -*-


import os
import re
import json
from codecs import open
from collections import defaultdict, Counter
import cPickle as pickle
import numpy as np
import scipy.stats


# Special vocabulary symbols.
PAD_TOKEN = '<pad>' # pad symbol
UNK_TOKEN = '<unk>' # unknown word
BOS_TOKEN = '<bos>' # begin-of-sentence symbol
EOS_TOKEN = '<eos>' # end-of-sentence symbol
NUM_TOKEN = '<num>' # numbers

# we always put them at the start.
_START_VOCAB = [PAD_TOKEN, UNK_TOKEN]
PAD_ID = 0
UNK_ID = 1

# Regular expressions used to tokenize.
_DIGIT_RE = re.compile(br"^\d+$")


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
RANDOM_SEED = 1234


##########################################################
#
# Vocabulary preprocessing
#
##########################################################

def basic_tokenizer(sequence, bos=True, eos=True):
    sequence = re.sub(r'\s{2}', ' ' + EOS_TOKEN + ' ' + BOS_TOKEN + ' ', sequence)
    if bos:
        sequence = BOS_TOKEN + ' ' + sequence.strip()
    if eos:
        sequence = sequence.strip() + ' ' + EOS_TOKEN
    return sequence.lower().split()


def create_vocabulary(vocabulary_path, data_path, tokenizer, max_vocab_size=40000):
    """Create vocabulary file (if it does not exist yet) from data file.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    if not os.path.exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with open(data_path, "rb", encoding="utf-8") as f:
            for line in f.readlines():
                tokens = tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, NUM_TOKEN, w)
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocab_size:
                print("  %d words found. Truncate to %d." % (len(vocab_list), max_vocab_size))
                vocab_list = vocab_list[:max_vocab_size]
            with open(vocabulary_path, "wb", encoding="utf-8") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, "rb", encoding="utf-8") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer):
    """Convert a string to list of integers representing token-ids.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    words = tokenizer(sentence)
    return [vocabulary.get(re.sub(_DIGIT_RE, NUM_TOKEN, w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    if not os.path.exists(target_path):
        print("Vectorizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with open(data_path, "rb", encoding="utf-8") as data_file:
            with open(target_path, "wb", encoding="utf-8") as tokens_file:
                for line in data_file:
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


##########################################################
#
# Read preprocessed data
#
##########################################################

def shuffle_split(X, y, n=[], shuffle=True):
    """Shuffle and split data into train and test subset"""
    _X = np.array(X)
    _y = np.array(y)
    data_size = _y.shape[0]
    assert _X.shape[0] == data_size

    _n = [None] * data_size
    if len(n) == data_size:
        _n = np.array(n)
        assert _n.shape == _y.shape

    # shuffle data
    data = np.array(zip(_X, _y, _n))
    if shuffle:
        np.random.seed(RANDOM_SEED)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        return data[shuffle_indices]
    else:
        return data


def read_data(data_dir, name, sent_len, negative=False, hierarchical=False, shuffle=True):
    """Read source and target.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py
    """
    _X = []
    _y = []
    _n = []
    source_path = os.path.join(data_dir, '%s_source.ids' % name)
    target_path = os.path.join(data_dir, '%s_target.txt' % name)
    class_names = load_from_dump(os.path.join(data_dir, 'classes.cPickle'))
    if negative:
        relations = dict(load_from_dump(os.path.join(data_dir, 'relations.cPickle')))

    with open(source_path, mode="r", encoding="utf-8") as source_file:
        with open(target_path, mode="r", encoding="utf-8") as target_file:
            source, target = source_file.readline(), target_file.readline()
            print "Loading %s data ..." % name,
            while source and target:
                source_ids = [np.int64(x.strip()) for x in source.split()]
                if sent_len > len(source_ids):
                    source_ids += [PAD_ID] * (sent_len - len(source_ids))
                assert len(source_ids) == sent_len

                labels = [y.strip() for y in target.split(',')]
                target_ids = binarize_label(class_names, labels, hierarchical=hierarchical)

                _X.append(source_ids)
                _y.append(target_ids)
                if negative:
                    negative_ids = pseudo_negative_sampling(labels, class_names, relations, hierarchical=hierarchical)
                    _n.append(negative_ids)
                source, target = source_file.readline(), target_file.readline()

    assert len(_X) == len(_y)
    print "\t%d examples found." % len(_y)

    return shuffle_split(_X, _y, n=_n, shuffle=shuffle)


def shuffle_split_contextwise(X, y, n=[], shuffle=True):
    """Shuffle and split data into train and test subset"""

    _left = np.array(X['left'])
    _right = np.array(X['right'])
    _y = np.array(y)
    data_size = _y.shape[0]
    assert _left.shape[0] == data_size
    assert _right.shape[0] == data_size

    _n = [None] * data_size
    if len(n) == data_size:
        _n = np.array(n)
        assert _n.shape == _y.shape

    # shuffle data
    data = np.array(zip(_left, _right, _y, _n))
    if shuffle:
        np.random.seed(RANDOM_SEED)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        return data[shuffle_indices]
    else:
        return data


def read_data_contextwise(data_dir, name, sent_len, negative=False, hierarchical=False, shuffle=True):
    """Read source file and pad the sequence to sent_len,
       combine them with target (and attention if given).

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py
    """
    print "Loading %s data ..." % name,
    _X = {'left': [], 'right': []}
    for context in _X.keys():
        path = os.path.join(data_dir, '%s_%s.ids' % (name, context))
        with open(path, mode="r", encoding="utf-8") as source_file:
            for source in source_file.readlines():
                source_ids = [np.int64(x.strip()) for x in source.split()]
                if sent_len > len(source_ids):
                    source_ids += [PAD_ID] * (sent_len - len(source_ids))
                assert len(source_ids) == sent_len
                _X[context].append(source_ids)
    assert len(_X['left']) == len(_X['right'])

    _y = []
    _n = []
    class_names = load_from_dump(os.path.join(data_dir, 'classes.cPickle'))
    if negative:
        relations = dict(load_from_dump(os.path.join(data_dir, 'relations.cPickle')))
    with open(os.path.join(data_dir, '%s_target.txt' % name), mode="r", encoding="utf-8") as target_file:
        for target in target_file.readlines():
            labels = [y.strip() for y in target.split(',')]
            target_ids = binarize_label(class_names, labels, hierarchical=hierarchical)
            _y.append(target_ids)
            if negative:
                negative_ids = pseudo_negative_sampling(labels, class_names, relations, hierarchical=hierarchical)
                _n.append(negative_ids)
    assert len(_X['left']) == len(_y)
    print "\t%d examples found." % len(_y)

    return shuffle_split_contextwise(_X, _y, n=_n, shuffle=shuffle)



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Generates a batch iterator.

    Original taken from
    https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(np.ceil(float(data_size)/batch_size))
    for epoch in range(num_epochs):
        # Shuffle data at each epoch
        if shuffle:
            np.random.seed(RANDOM_SEED + epoch)
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


##########################################################
#
# Pretrained embeddings
#
##########################################################

def dump_to_file(filename, obj):
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, file=outfile)
    return

def load_from_dump(filename):
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)
    return obj

def _load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec

    Original taken from
    https://github.com/yuhaozhang/sentence-convnet/blob/master/text_input.py
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return (word_vecs, layer1_size)

def _add_random_vec(word_vecs, vocab, emb_size=300):
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,emb_size)
    return word_vecs

def prepare_pretrained_embedding(data_dir, fname, word2id):
    if not os.path.exists(os.path.join(data_dir, 'emb.npy')):
        if os.path.exists(fname):
            print 'Reading pretrained word vectors from file ...'
            word_vecs, emb_size = _load_bin_vec(fname, word2id)
            word_vecs = _add_random_vec(word_vecs, word2id, emb_size)
            embedding = np.zeros([len(word2id), emb_size])
            for w,idx in word2id.iteritems():
                embedding[idx,:] = word_vecs[w]
            print 'Generated embeddings with shape ' + str(embedding.shape)
            np.save(os.path.join(data_dir, 'emb.npy'), embedding)
        else:
            print "Pretrained embeddings file %s not found." % fname


##########################################################
#
# Calculate AUC-PR
#
##########################################################

def offset(array, pre, post):
    ret = np.array(array)
    ret = np.insert(ret, 0, pre)
    ret = np.append(ret, post)
    return ret

def calc_auc_pr(precision, recall):
    assert len(precision) == len(recall)
    return np.trapz(offset(precision, 1, 0), x=offset(recall, 0, 1), dx=5)


##########################################################
#
# Active learning
#
##########################################################

def minmax_scale(seq, offset=1e-5):
    maximum = np.max(seq, axis=0)
    minimum = np.min(seq, axis=0)
    std = (seq - minimum) / (maximum - minimum)
    scaled = std * (1.0 - offset) + offset
    return scaled


def most_informative(pool_data, config, strategy='max_entropy', class_names=None, relations=None):
    import predict

    if strategy == 'model_change':
        grad_len = predict.emb(pool_data, config, class_names=class_names, relations=relations)
        idx = np.argsort(grad_len)
        idx = list(idx)[-1 * config['batch_size']:] # take last-k examples

    else:
        scores, _, _ = predict.predict(pool_data, config)


        if strategy == 'max_entropy':
            prob = [scipy.stats.entropy(s) for s in scores]
            idx = np.argsort(prob)
            idx = list(idx)[-1 * config['batch_size']:] # take last-k examples

        elif strategy == 'least_confident':
            prob = np.argmax(scores, axis=1)
            idx = np.argsort(prob)
            idx = list(idx)[:config['batch_size']] # take first-k examples

        elif strategy == 'smallest_margin':
            prob = [s[-1] - s[-2] for s in np.sort(scores, axis=1)]
            idx = np.argsort(prob)
            idx = list(idx)[:config['batch_size']] # take first-k examples

        else:
            raise Exception('strategy not defined.')

    return idx


##########################################################
#
# Negative sampling
#
##########################################################

def binarize_label(class_names, labels, hierarchical=False):
    target = [0] * len(class_names)
    for label in labels:
        target[class_names.index(label)] = 1
    if hierarchical:
        for label in labels:
            l = label.split('.')
            if len(l) > 1:
                if l[0] in class_names:
                    target[class_names.index(l[0])] = 1
                if '.'.join(l[:2]) in class_names:
                    target[class_names.index('.'.join(l[:2]))] = 1
        return np.array(target, dtype=np.float32)
    else:
        return np.array(target, dtype=np.float32)



def pseudo_negative_sampling(y_true, class_names, relations, hierarchical=False):
    """Sample negative class"""
    inverse = {k: np.power(1.0/v, 0.5) for k, v in relations.iteritems() if k != tuple(y_true)}
    s = sum(inverse.values())
    candidates = {k: v/float(s) for k, v in inverse.iteritems()}
    i = np.random.choice(np.arange(0, len(candidates), dtype=int), p=candidates.values())
    labels = candidates.keys()[i]
    target = binarize_label(class_names, labels, hierarchical=hierarchical)
    return target


##########################################################
#
# Restore pdtb data
#
##########################################################

def restore_token_dict(data, parses):
    """Restore token list and store it in dict"""
    tokens = defaultdict(list)
    for doc_id in set([line['DocID'] for line in data]):
        all_tokens = [token for sentence in parses[doc_id]['sentences'] for token in sentence['words']]

        for token in all_tokens:
            for linker in token[1]['Linkers']:
                role, relation_id = linker.split('_')
                tokens[int(relation_id)].append((token[0], role))
    return tokens

def save_classes(data_dir, data):
    """Save class names and class frequency"""
    cooccur_relations = []
    relations = []
    for line in data:
        relations.extend(line['Sense'])
        cooccur_relations.append(tuple(line['Sense']))
    classes = Counter(relations).most_common(25)
    cooccur_classes = Counter(cooccur_relations).most_common(110)
    dump_to_file(os.path.join(data_dir, 'classes.cPickle'), [c[0] for c in classes])
    dump_to_file(os.path.join(data_dir, 'relations.cPickle'), cooccur_classes)

def filter_out(tokens, limit=100):
    """Filter out too log sentences"""
    target = {}
    for k, v in tokens.iteritems():
        arg1 = len([t for t in v if t[1] == 'arg1'])
        arg2 = len([t for t in v if t[1] == 'arg2'])
        if arg1 <= limit and arg2 <= limit:
            target[k] = v
    return target

def save_text_data(data_dir, name, data, parses, limit=100):
    """Save text data and target label ids"""
    tokens = restore_token_dict(data, parses)
    tokens = filter_out(tokens, limit)
    with open(os.path.join(data_dir, '%s_source.txt' % name), 'w', encoding='utf-8') as f:
        for i in sorted(tokens):
            f.write(' '.join([t[0] for t in tokens[i]]) + '\n')
    with open(os.path.join(data_dir, '%s_left.txt' % name), 'w', encoding='utf-8') as f:
        for i in sorted(tokens):
            f.write(' '.join([t[0] for t in tokens[i] if t[1] == 'arg1']) + '\n')
    with open(os.path.join(data_dir, '%s_right.txt' % name), 'w', encoding='utf-8') as f:
        for i in sorted(tokens):
            f.write(' '.join([t[0] for t in tokens[i] if t[1] == 'arg2']) + '\n')
    with open(os.path.join(data_dir, '%s_target.txt' % name), 'w', encoding='utf-8') as f:
        target = {int(line['ID']):line['Sense'] for line in data if int(line['ID']) in tokens.keys()}
        for i in sorted(target):
            f.write(', '.join(target[i]) + '\n')
    with open(os.path.join(data_dir, '%s_type.txt' % name), 'w', encoding='utf-8') as f:
        orig = {int(line['ID']):[str(line['ID']), line['DocID'], line['Type']] for line in data
                if int(line['ID']) in tokens.keys()}
        for i in sorted(orig):
            f.write(', '.join(orig[i]) + '\n')

def load_from_json(data_dir, name):
    """Load pdtb data from json"""
    with open(os.path.join(data_dir, 'conll', 'pdtb-%s.json' % name), encoding='utf8') as f:
        data = json.loads(f.read())
    with open(os.path.join(data_dir, 'conll', 'pdtb-%s-parses.json' % name), encoding='utf8') as f:
        parses = json.loads(f.read())[0]
    return data, parses




##########################################################
#
# main
#
##########################################################

def main():
    data_dir = os.path.join(THIS_DIR, 'data')

    # load conll dataset
    for name in ['train', 'dev']:
        print 'Restoring %s data ...' % name
        data, parses = load_from_json(data_dir, name)
        if name == 'train':
            save_classes(data_dir, data)
        save_text_data(data_dir, name, data, parses, limit=100)

    # vectorization
    vocab_path = os.path.join(data_dir, 'vocab.txt')
    data_path = os.path.join(data_dir, 'train_source.txt')
    max_vocab_size = 35000
    tokenizer = lambda x: x.lower().split()
    create_vocabulary(vocab_path, data_path, tokenizer, max_vocab_size)
    for name in ['train', 'dev']:
        for context in ['left', 'right', 'source']:
            data_path = os.path.join(data_dir, '%s_%s.txt' % (name, context))
            target_path = os.path.join(data_dir, '%s_%s.ids' % (name, context))
            data_to_token_ids(data_path, target_path, vocab_path, tokenizer=tokenizer)

    # pretrained embeddings
    embedding_path = os.path.join(THIS_DIR, 'word2vec', 'GoogleNews-vectors-negative300.bin')
    word2id, _ = initialize_vocabulary(vocab_path)
    prepare_pretrained_embedding(data_dir, embedding_path, word2id)




if __name__ == '__main__':
    main()
