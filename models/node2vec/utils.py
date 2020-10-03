import numpy as np
import torch
from torch.utils.data import Dataset
from time import time


np.random.seed(12345)


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName
        t0 = time()
        self.read_words(min_count)
        t1 = time()
        #print('data_reader.read_words in %.4f' %(t1 - t0))
        self.initTableNegatives()
        t2 = time()
        #print('data_reader.initTableNegatives in %.4f' % (t2 - t1))
        self.initTableDiscards()
        t3 = time()
        #print('data_reader.initTableDiscards in %.4f' % (t3 - t2))

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


# -----------------------------------------------------------------------------------------------------------------

class Word2vecDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    boundary = np.random.randint(1, self.window_size)

                    item = [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]

                    #print('len item: ', len(item))

                    return item

    @staticmethod
    def collate1(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        #print('\t\tlen u, v, neg_v: %s - %s - %s' %(len(all_u), len(all_v), len(all_neg_v)))

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)


    @staticmethod
    def collate(batches):
        all_u = torch.LongTensor([u for batch in batches for u, _, _ in batch if len(batch) > 0]).unsqueeze(1)
        all_v = torch.LongTensor([v for batch in batches for _, v, _ in batch if len(batch) > 0]).unsqueeze(1)
        all_neg_v = torch.LongTensor([neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0])

        source_node = all_u.repeat((1,6)).flatten()
        target_node = torch.cat((all_v, all_neg_v), dim=1).flatten()
        label = torch.cat((torch.ones_like(all_v), -torch.ones_like(all_neg_v)), dim=1).flatten()

        #print(source_node.shape, target_node.shape, label.shape)
        #print(source_node[:6])
        #print(target_node[:6])
        #print(label[:6])

        return source_node.tolist(), target_node.tolist(), label


    def node_distribution_power(self, r):
        word_fre = np.array(list(self.data.word_frequency.values())) ** r
        sum = np.sum(word_fre)
        return word_fre/sum

    def embedding_mapping(self, embedding):
        return {w: embedding[wid] for wid, w in self.data.id2word.items()}



