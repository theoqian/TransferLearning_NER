import codecs
import random
import math
from collections import Counter
import numpy as np
import re


UNKNOWN_CHAR = '<UNK>'


tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


boson_tag2label = {"O": 0,
                   "B-product_name": 1, "I-product_name": 2,
                   "B-time": 3, "I-time": 4,
                   "B-person_name": 5, "I-person_name": 6,
                   "B-org_name": 7, "I-org_name": 8,
                   "B-company_name": 9, "I-company_name": 10,
                   "B-location": 11, "I-location": 12}


def load_wordvec(wordvec_path, id_to_word, vec_dim, old_embeddings):
    """Load word vectors from pre-trained file.
    """
    embeddings = old_embeddings
    word_vectors = {}
    for i, line in enumerate(codecs.open(wordvec_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == vec_dim + 1:
            word_vectors[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
    for i in range(len(id_to_word)):
        if id_to_word[i] in word_vectors:
            embeddings[i] = word_vectors[id_to_word[i]]
    return embeddings


def load_sentence(path):
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = line.rstrip()
        if not line:
            if sentence:
                sentences.append(sentence)
                sentence = []
        else:
            word = line.split(' ')
            assert len(word) == 2
            sentence.append(word)
    if sentence:
        sentences.append(sentence)
    return sentences


def char_mapping(sentences):
    chars = [c[0] for s in sentences for c in s]
    chars_counter = Counter(chars)
    sorted_chars = sorted(chars_counter.items(), key=lambda x: (-x[1], x[0]))
    id2char, _ = list(zip(*sorted_chars))
    id2char = list(id2char)
    id2char.append(UNKNOWN_CHAR)
    vocab_len = len(id2char)
    char2id = dict(zip(id2char, range(vocab_len)))
    char2id[UNKNOWN_CHAR] = vocab_len - 1
    return id2char, char2id


def tag_mapping():
    tag2id = tag2label
    id2tag = [0 for _ in range(len(tag2id))]
    for k in tag2id:
        id2tag[tag2id[k]] = k
    return id2tag, tag2id


def preprocess_data(sentences, char2id, tag2id):
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char2id[w if w in char2id else UNKNOWN_CHAR] for w in string]
        tags = [tag2id[w[1]] for w in s]
        data.append((string, chars, tags))
    return data


class BatchManager(object):
    def __init__(self, data, batch_size):
        assert batch_size > 0
        self.batch_data = self._sort_and_pad(data, batch_size)
        self.batch_count = len(self.batch_data)

    def _sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batchs = []
        for i in range(num_batch):
            batchs.append(self._pad_data(sorted_data[i * batch_size: (i+1) * batch_size]))
        return batchs

    def _pad_data(self, data):
        strings, chars, tags, lengths = [], [], [], []
        max_len = max([len(s[0]) for s in data])
        for line in data:
            s, c, t = line
            padding = [0] * (max_len - len(s))
            strings.append(s)
            chars.append(c + padding)
            tags.append(t + padding)
            lengths.append(len(s))
        return [strings, chars, tags, lengths]

    def iter_batch(self, shuffle=True):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(len(self.batch_data)):
            yield self.batch_data[idx]


def input_from_line(line, char2id):
    inputs = list()
    inputs.append([line])
    inputs.append([[char2id[char] if char in char2id else char2id[UNKNOWN_CHAR] for char in line]])
    inputs.append([[]])
    inputs.append([len(line)])
    return inputs


def convert_bonson_data(src_file, train_file, test_file):
    with open(src_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in lines[:-200]:
            for sentence in line.rstrip().split("。"):
                if sentence != "":
                    sentence = sentence + "。"
                    f.write(convert_to_samples(sentence.replace('\\n', '')) + '\n\n')
    with open(test_file, 'w', encoding='utf-8') as f:
        for line in lines[-200:]:
            for sentence in line.rstrip().split("。"):
                if sentence != "":
                    sentence = sentence + "。"
                    f.write(convert_to_samples(sentence.replace('\\n', '')) + '\n\n')


def convert_to_samples(line):
    start, end = 0, 0
    samples = []
    line = line.replace(' ', '')
    for elem in re.finditer("{{[^{}]*:[^{}]*}}", line):
        start, _ = elem.span()
        for i in range(end, start):
            samples.append(line[i] + ' O')
        _, end = elem.span()
        sample = elem.group()
        pos = sample.find(":")
        tag, string = sample[2: pos], sample[pos+1: -2]
        samples.append(string[0] + " B-" + tag)
        for i in range(1, len(string)):
            samples.append(string[i] + " I-" + tag)
    for i in range(end, len(line)):
        samples.append(line[i] + ' O')
    return "\n".join(samples)

def boson_tag_mapping():
    tag2id = boson_tag2label
    id2tag = [0 for _ in range(len(tag2id))]
    for k in tag2id:
        id2tag[tag2id[k]] = k
    return id2tag, tag2id
