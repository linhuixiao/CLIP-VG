# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torchvision.transforms import Compose, ToTensor, Normalize

import datasets.transforms as T
from .data_loader import TransVGDataset

""""CLIP 自带的 transform"""
# def _transform(n_px):
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])


def make_transforms(args, image_set, is_onestage=False):
    if is_onestage:
        normalize = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return normalize

    imsize = args.imsize

    # TODO: 不一样，多了train_pseudo
    if image_set in ['train', 'train_pseudo']:
        scales = []
        if args.aug_scale:
            for i in range(7):
                scales.append(imsize - 32 * i)
        else:
            scales = [imsize]

        if args.aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.

        # RandomResize 默认 with_long_side = True, 区别在于 resize 是按照长边压缩还是按照短边压缩，最后整体都需要压缩
        # TODO: 不管如何压缩，都需要使用到长边压缩
        return T.Compose([
            T.RandomSelect(
                # 这一步保证了后续所有的裁剪都会裁剪为比目标size更小
                T.RandomResize(scales),
                T.Compose([
                    # 即便被放大了，后续也要压缩回去
                    T.RandomResize([400, 500, 600], with_long_side=False),
                    T.RandomSizeCrop(384, 600),
                    # 这一步也确保了随机选择此组的话，最终的边小于目标size，感觉变来变去是多余的
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.GaussianBlur(aug_blur=args.aug_blur),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize, aug_translate=args.aug_translate)
        ])


    if image_set in ['val', 'test', 'testA', 'testB']:
        # 现在还只是初始化一个预处理器，将图片沿着长边resize到目标大小，之后再将图片转化为tensor，之后再将短边补齐pad到目标size
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize),
        ])

    raise ValueError(f'unknown {image_set}')


# args.data_root default='./ln_data/', args.split_root default='data', '--dataset', default='referit'
# split = test, testA, val, args.max_query_len = 20
# TODO: 多加了 prompt_template=args.prompt, 其余和 TransVG 一模一样
def build_dataset(split, args):
    return TransVGDataset(data_root=args.data_root,
                          split_root=args.split_root,
                          dataset=args.dataset,
                          split=split,
                          transform=make_transforms(args, split),
                          max_query_len=args.max_query_len,
                          prompt_template=args.prompt)
