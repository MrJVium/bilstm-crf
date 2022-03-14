from os.path import join, exists
import numpy as np
from tqdm import tqdm
import torch
import json
import re

START_TAG = "<START>"
STOP_TAG = "<STOP>"

PAD = "<PAD>"
OOV = "<OOV>"  # out of vocabulary

FILE_VOCAB = "seg-data/training/vocab.json"
FILE_TAGS = "seg-data/training/tags.json"
FILE_DATASET_CANDIDATE = ["seg-data/training/msr_training.utf8", "seg-data/training/pku_training.utf8"]
FILE_DATASET = "seg-data/training/dataset.uft8"
FILE_DATASET_CACHE = "seg-data/training/dataset_cache_{}.npz"

def save_json_file(obj, file_path):
    with open(file_path, "w", encoding='utf8') as f:
        f.write(json.dumps(obj, ensure_ascii=False))

def load_json_file(file_path):
    with open(file_path, encoding='utf8') as f:
        return json.load(f)

def get_data():
    """
        将两个训练集合并，当长度大于 300将其 按 [。，？！、]符号进行分割
        之后processor读取  若长度以旧大于300 则抛弃
    """
    dataset = []

    for path in FILE_DATASET_CANDIDATE:
        for line in open(path):
            s = ''.join(line.strip().split())
            if len(s) > 300:
                dataset += re.split("(?<=[。！？ ，、]\s\s)", line)
            else:
                dataset.append(line)

    f = open(FILE_DATASET, 'w')
    for i in dataset:
        f.writelines(i)
    f.close()


class Processor():
    def __init__(self):
        self.vocab, self.vocab_dict = self.__load_list_file(FILE_VOCAB, offset=1)
        self.tags = ["S", "B", "M", "E"]
        self.tags_dict = {t: i for i, t in enumerate(self.tags)}

        self.PAD_IDX = 0
        self.OOV_IDX = len(self.vocab_dict)
        self.__adjust_vocab()

    def __load_list_file(self, path, offset=0):
        element = load_json_file(path)
        element_dict = {w: i + offset for i, w in enumerate(element)}
        return element, element_dict

    def __adjust_vocab(self):
        self.vocab.insert(0, PAD)
        self.vocab_dict[PAD] = 0

        self.vocab.append(OOV)
        self.vocab_dict[OOV] = self.OOV_IDX

    @staticmethod
    def __cache_file_path(max_seq_len):
        return FILE_DATASET_CACHE.format(max_seq_len)

    def decode_tags(self, batch_tags):
        batch_tags = [
            [self.tags[t] for t in tags]
            for tags in batch_tags
        ]
        return batch_tags

    def load_dataset(self, val_split, test_split, max_seq_len):
        ds_path = self.__cache_file_path(max_seq_len)

        if not exists(ds_path):
            xs, ys = self.__build_corpus(max_seq_len)
        else:
            dataset = np.load(ds_path)
            xs, ys = dataset["xs"], dataset['ys']

        xs, ys = map(
            torch.tensor, (xs, ys)
        )

        # split the dataset
        total_count = len(xs)
        assert total_count == len(ys)
        val_count = int(total_count * val_split)
        test_count = int(total_count * test_split)
        train_count = total_count - val_count - test_count
        assert train_count > 0 and val_count > 0

        indices = np.cumsum([0, train_count, val_count, test_count])
        datasets = [(xs[s:e], ys[s:e]) for s, e in zip(indices[:-1], indices[1:])]
        print("datasets loaded:")
        for (xs_, ys_), name in zip(datasets, ["train", "val", "test"]):
            print("\t{}: {}, {}".format(name, xs_.shape, ys_.shape))
        return datasets

    def sent_to_vec(self, sent, max_len=0):
        max_seq_len = max_len if max_len > 0 else len(sent)
        vec = [self.vocab_dict[s] for s in sent[:max_seq_len]]
        return vec + [self.PAD_IDX] * (max_seq_len - len(sent))

    def tags_to_vec(self, tag, max_len=0):
        max_seq_len = max_len if max_len > 0 else len(tag)
        vec = [self.tags_dict[t] for t in tag[:max_seq_len]]
        return vec + [0] * (max_seq_len - len(tag))

    def __get_status(self, words):
        status = []
        for word in words:
            if len(word) == 1:
                status.append("S")
            elif len(word) == 2:
                status.extend(["B", "E"])
            else:
                status.extend(["B"] + ["M"] * (len(word) - 2) + ["E"])
        return status

    def __build_corpus(self, max_seq_len):
        xs = []
        ys = []
        for line in open(FILE_DATASET):
            s = ''.join(line.strip().split())
            if len(s) < max_seq_len and lens(s) > 0:
                xs.append(self.sent_to_vec(s, max_seq_len))
                tag = self.__get_status(line.strip().split())
                ys.append(self.tags_to_vec(
                    tag, max_seq_len
                ))
                assert len(tag) == len(s)

        xs, ys = np.asarray(xs), np.asarray(ys)
        cache_file = self.__cache_file_path(max_seq_len)
        np.savez(cache_file, xs=xs, ys=ys)
        return xs, ys