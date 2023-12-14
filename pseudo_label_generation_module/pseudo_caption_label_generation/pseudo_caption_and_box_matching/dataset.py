from PIL import Image
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO as pyCOCO
import json
import itertools
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from torchvision.transforms import functional as F


LENGTH_LIMIT = 75


def parse(ann_dir, img_root):
    # 如下几行代码是将coco数据集分成 train 和 val 2个部分， ids 是一个二元元组，包含train 和val元组，每个元组又包含图片的id信息
    # (array([787980, 789366, 789888, ..., 5716, 8002, 9277]),
    #   array([681330, 686718, 688839, ..., 647813, 648512, 650354]))
    ids = (
        np.load(ann_dir / "coco_train_ids.npy"),
        np.concatenate([
            np.load(ann_dir / "coco_restval_ids.npy"),
            np.load(ann_dir / "coco_dev_ids.npy"),
            np.load(ann_dir / "coco_test_ids.npy")
        ]),
    )
    coco = (
        pyCOCO(ann_dir / "captions_train2014.json"),
        pyCOCO(ann_dir / "captions_val2014.json"),
    )
    img_root = (img_root / "train2014", img_root / "val2014")

    # 最终再依次从train 和 val 2 部分中整理出 图片和 caption 数据列表，并返回缓存
    data = {}
    for i in range(len(ids)):  # ids 是长度为2的元组
        for idx in ids[i]:
            # ids[0]:  [787980 789366 789888 ...   5716   8002   9277]
            # ids[1]:  [681330 686718 688839 ... 647813 648512 650354]，由三个向量拼成一个一维向量
            img_id = coco[i].anns[idx]["image_id"]
            img_file = img_root[i] / coco[i].loadImgs(img_id)[0]["file_name"]
            caption = coco[i].anns[idx]["caption"].strip()

            if img_id in data:
                data[img_id]["captions"].append(caption)
            else:
                data[img_id] = {
                    "image_id": img_id,
                    "image_file": img_file,
                    "captions": [caption, ]
                }

    data = list(data.values())  # data.values() 将字典的values全部提取并转换为元组，list(data.values())再将元组转换为包含字典的列表
    data.sort(key=lambda x: x["image_id"])  # .sort()只能对list进行排序，lambda x是匿名函数，后接匿名函数，代表排序规则

    return data  # 最终输出一组带有[{image_id, image_file, captions}, ...]，同时按image_id排序好的字典列表


def LoadCocoCaption(ann_dir, img_root, transform=None):
    transform = transform
    # 执行如下解析，提取训练集和验证集数据
    data = parse(Path(ann_dir), Path(img_root))
    return data

class CocoImageCrops(Dataset):
    def __init__(self, ann_dir, img_root, transform=None):
        self.transform = transform
        # 执行如下解析，提取训练集和验证集数据
        self.data = self.parse(Path(ann_dir), Path(img_root))

    @staticmethod
    def parse(ann_dir, img_root):
        # 如下几行代码是将coco数据集分成 train 和 val 2个部分， ids 是一个二元元组，包含train 和val元组，每个元组又包含图片的id信息
        # (array([787980, 789366, 789888, ..., 5716, 8002, 9277]),
        #   array([681330, 686718, 688839, ..., 647813, 648512, 650354]))
        ids = (
            np.load(ann_dir / "coco_train_ids.npy"),
            np.concatenate([
                np.load(ann_dir / "coco_restval_ids.npy"),
                np.load(ann_dir / "coco_dev_ids.npy"),
                np.load(ann_dir / "coco_test_ids.npy")
            ]),
        )
        coco = (
            pyCOCO(ann_dir / "captions_train2014.json"),
            pyCOCO(ann_dir / "captions_val2014.json"),
        )
        img_root = (img_root / "train2014", img_root / "val2014")

        # 最终再依次从train 和 val 2 部分中整理出 图片和 caption 数据列表，并返回缓存
        data = {}
        for i in range(len(ids)):  # ids 是长度为2的元组
            for idx in ids[i]:
                # ids[0]:  [787980 789366 789888 ...   5716   8002   9277]
                # ids[1]:  [681330 686718 688839 ... 647813 648512 650354]，由三个向量拼成一个一维向量
                img_id = coco[i].anns[idx]["image_id"]
                img_file = img_root[i] / coco[i].loadImgs(img_id)[0]["file_name"]
                caption = coco[i].anns[idx]["caption"].strip()

                if img_id in data:
                    data[img_id]["captions"].append(caption)
                else:
                    data[img_id] = {
                        "image_id": img_id,
                        "image_file": img_file,
                        "captions": [caption, ]
                    }

        data = list(data.values())  # data.values() 将字典的values全部提取并转换为元组，list(data.values())再将元组转换为包含字典的列表
        data.sort(key=lambda x: x["image_id"])  # .sort()只能对list进行排序，lambda x是匿名函数，后接匿名函数，代表排序规则

        return data  # 最终输出一组带有[{image_id, image_file, captions}, ...]，同时按image_id排序好的字典列表

    def five_crop(self, image, ratio=0.6):
        w, h = image.size
        hw = (h * ratio, w * ratio)

        return F.five_crop(image, hw)

    def nine_crop(self, image, ratio=0.4):
        w, h = image.size

        t = (0, int((0.5 - ratio / 2) * h), int((1.0 - ratio) * h))
        b = (int(ratio * h), int((0.5 + ratio / 2) * h), h)
        l = (0, int((0.5 - ratio / 2) * w), int((1.0 - ratio) * w))
        r = (int(ratio * w), int((0.5 + ratio / 2) * w), w)
        h, w = list(zip(t, b)), list(zip(l, r))

        images = []
        for s in itertools.product(h, w):
            h, w = s
            top, left = h[0], w[0]
            height, width = h[1] - h[0], w[1] - w[0]
            images.append(F.crop(image, top, left, height, width))

        return images

    def __len__(self):
        return len(self.data)

    '''
    def __getitem__(self, index):
        image = Image.open(self.data[index]["image_file"])
        image = image.convert("RGB")

        five_images = self.five_crop(image)
        nine_images = self.nine_crop(image)

        if self.transform is not None:
            orig_image = self.transform(image)
            five_images = torch.stack([self.transform(x) for x in five_images])
            nine_images = torch.stack([self.transform(x) for x in nine_images])

        captions = self.data[index]["captions"]
        idx = self.data[index]["image_id"]

        return orig_image, five_images, nine_images, captions, idx
    '''

