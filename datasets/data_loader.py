# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import re
# import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data

sys.path.append('.')


from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.word_utils import Corpus

# from CLIP-VG.models.clip import *
import clip

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line  # reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class DatasetNotFoundError(Exception):
    pass


class TransVGDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        # TODO: 数据集不一样，全部多了 train_pseudo
        'referit': {'splits': ('train', 'val', 'trainval', 'test', 'train_pseudo')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB', 'train_pseudo'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB', 'train_pseudo'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val', 'train_pseudo'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test', 'train_pseudo'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test', 'train_pseudo')}
    }

    """ 数据集核心处理部分 """
    def __init__(self, data_root, split_root='data', dataset='referit',
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, prompt_template=None, lstm=False,
                 bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.prompt_template = prompt_template
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx = return_idx

        assert self.transform is not None

        if split in ['train', 'train_pseudo']:
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            # TODO: 这里把 flickr30k_images 改为 flickr30k-images
            self.im_dir = osp.join(self.dataset_root, 'flickr30k-images')
        else:  ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('The dataset {} is not found!'.format(osp.join(self.split_root, self.dataset)))
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            # print('\nimgset_path:\n', imgset_path)
            """ imgset_path: /data_SSD1/lhxiao/pseudo-q/data/pseudo_samples/unc/unc_testA.pth """
            # 从侧面印证了所加载的照片，只是从预先分割和设定好的 pth 包中进行加载
            self.images += torch.load(imgset_path)

        if self.prompt_template:
            self.images = self.prompt(self.images)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    # TODO: 这句是关键
    def pull_item(self, idx):
        if self.dataset == 'flickr':  # flickr 的数据格式
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri = self.images[idx]  # 最原始的数据分割文件格式
        ## box format: to x1y1x2y2
        bbox_ori = bbox
        # TODO: 对于 refcoco数据集，统一将 bbox 从 xywh（横向x，纵向y） 转成 x1y1x2y2 格式
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        return img_file, img, phrase, bbox, bbox_ori

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    # TODO: 新增
    def prompt(self, sample_list):
        n = len(sample_list)
        new_sample_list = []

        for i in range(n):
            if self.dataset == 'flickr':
                tmp_sample = (sample_list[i][0], sample_list[i][1], self.prompt_template.replace('{pseudo_query}', sample_list[i][2]))
            else:
                # print("\nsample_list:\n", sample_list[i])
                #  ('COCO_train2014_000000000839.jpg', '482127.pth', [303.58, 69.03, 293.29, 425.79],
                #  'guy flopping around on the right', [('r1', ['guy']), ('r2', ['none']), ('r3', ['none']),
                #  ('r4', ['none']), ('r5', ['none']), ('r6', ['none']), ('r7', ['none']),
                #  ('r8', ['right', 'flopping', 'around'])])
                tmp_sample = (sample_list[i][0], sample_list[i][1], sample_list[i][2],
                              self.prompt_template.replace('{pseudo_query}', sample_list[i][3]), sample_list[i][4])
            new_sample_list.append(tmp_sample)
        return new_sample_list

    def __len__(self):
        return len(self.images)

    # TODO: 这个函数是核心
    def __getitem__(self, idx):
        # 此时的处理是一条一条进行处理的，如下都是一维tensor，不带 batch_size
        img_file, img, phrase, bbox, bbox_ori = self.pull_item(idx)
        # print("\n img shape: ", img.size)  #  (640, 480)
        # print("\n img data: ", img)   # <PIL.Image.Image image mode=RGB size=640x480 at 0x7F2CCB4D8F70>
        #       img_arr = np.asarray(img)
        # 有数值，是HWC格式，且是0~255整数,img_arr.size= 921600, 819840, 894720 大小各不同, img_arr.shape=(640, 480, 3)各不同
        # print("img_arr: ", img_arr.shape)
        # print("\n bbox before transform: ", bbox)  # tensor([  4.,  87., 230., 408.]), 此时的 bbox 是属于 x1y1x2y2 的格式，一维 tensor
        phrase = phrase.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase}
        # print("bbox before: ", input_dict['box'])  # bbox before:  tensor([  0.,  93., 185., 419.])
        # print("img size before: ", input_dict['img'].shape)
        # transform 之前，各种形状都有，640 * 554， 640*394,431*640
        # print("img size before: ", input_dict['img'])  # <PIL.Image.Image image mode=RGB size=500x375 at 0x7F9482F74580>
        # TODO: 核心关键的一步，对图片，bbox 都做了数据增强, transform 由标准库中继承而来，直接对字典进行处理，
        #  打印结果表明，bbox也被处理了，这一操作在dataset/transform.py 中 RandomResize(object)实现，在后续的 normalize 中也处理了bbox,
        #  使用了 xyxy2cx_xy_wh，cx,cy 不是原来的x1y1
        input_dict = self.transform(input_dict)
        # print("bbox after: ", input_dict['box'])  # bbox after:  tensor([0.1445, 0.5652, 0.2891, 0.5094])
        # print("img size before: ", input_dict['img'].shape)  # torch.Size([3, 224, 224])

        img = input_dict['img']
        img_mask = input_dict['mask']  # 原来 mask 在 transform 阶段已经处理好了
        bbox = input_dict['box']
        phrase = input_dict['text']
        # print("\n img size: ", img.shape)  # torch.Size([3, 224, 224])，此时已经全部被处理了
        # print("\n img data: ", img)  # 此时打印出来的值全是 0
        # print("\n bbox after transform: ", bbox)  # tensor([0.8195, 0.6216, 0.3611, 0.5135]) 做了处理

        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id > 0, dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask

        # TODO: Modifies，此处可能写错了，此处代码为，有内容的为 1，没有内容的为 0，在 MDETR 中，是需要mask掉的为 1
        text_token = clip.tokenize(phrase)  # 计算结果是二维数组，1*77
        text = text_token.int()[0].tolist()
        text_mask = (text_token.clone() > 0).int()[0].tolist()
        # text_mask = torch.tensor(text_token > 0).int()[0].numpy().tolist()
        # print("\ntext_token: ", text_token)
        # print('\ntext: ', text)
        """ # old code
        if self.testmode:
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                   np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            # print(img.shape)
            return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)
        """

        if self.testmode:  # 默认是False
            return img, np.array(text, dtype=int), np.array(text_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                   np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:  # 避免7个变量
            # print(img.shape)
            return img, np.array(img_mask), np.array(text, dtype=int), np.array(text_mask, dtype=int), np.array(bbox, dtype=np.float32), img_file, phrase, bbox_ori


