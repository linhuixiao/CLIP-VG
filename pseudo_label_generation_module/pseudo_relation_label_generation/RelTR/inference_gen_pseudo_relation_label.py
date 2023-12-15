# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
import os
import time

import imageio
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    parser.add_argument('--vg_dataset_path', dest='vg_dataset_path',
                        help='unlabeled dataset path',
                        default='/hdd/lhxiao/pseudo-q/image_data/', type=str)
                        # default='/home/data/referit_data/', type=str)
    parser.add_argument('--vg_dataset', dest='vg_dataset',
                        help='unlabeled dataset',
                        default='unc', type=str)
    parser.add_argument('--split_ind', dest='split_ind',
                        default=0, type=int)

    # image path
    # parser.add_argument('--img_path', type=str, default='demo/customized.jpg', help="Path of the test image")
    # parser.add_argument('--img_path', type=str, default='demo/vg1.jpg', help="Path of the test image")
    parser.add_argument('--img_path', type=str, default='demo/vg2.jpg', help="Path of the test image")
    # parser.add_argument('--img_path', type=str, default='demo/COCO_train2014_000000581857.jpg', help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # parser.add_argument('--resume', default='ckpt/checkpoint0149.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default='/hdd/lhxiao/reltr/ckpt/checkpoint0149.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser


def main(args):
    args.vg_dataset_path = '/hdd/lhxiao/pseudo-q/image_data'
    if args.vg_dataset in ['unc', 'unc+', 'gref', 'gref_umd']:
        args.image_dir = os.path.join(args.vg_dataset_path, 'other/images/mscoco/images/train2014')
    elif args.vg_dataset == 'referit':
        args.image_dir = os.path.join(args.vg_dataset_path, 'referit/images')
    else:  # flickr
        args.image_dir = os.path.join(args.vg_dataset_path, 'Flickr30k/flickr30k-images/')

    args.out_path = '/hdd/lhxiao/pseudo-q/reltr_output/{}'.format(args.vg_dataset)

    args.image_list_file = '/hdd/lhxiao/pseudo-q/detection_results/splits/{}/{}_train_imagelist_split{}.txt'.format(
        args.vg_dataset, args.vg_dataset, args.split_ind)

    train_image_list = open(args.image_list_file, 'r')
    train_image_files = train_image_list.readlines()

    pseudo_train_samples = []
    count = 0
    transform1 = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform2 = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size  # 500 * 375
        b = box_cxcywh_to_xyxy(out_bbox)  # 12 * 4
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # VG classes,{list: 151}
    CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

    # {list: 51}
    REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['model'])
    model.eval()

    expression_train_samples = []

    start_time = time.time()
    topk = 10
    print("Begin to processing: ", args.image_list_file)
    for image_ind, image_file in enumerate(train_image_files):
        if image_ind % 100 == 0:
            left_time = ((time.time() - start_time) * (len(train_image_files) - image_ind - 1) / (image_ind + 1)) / 3600
            print('Processing {}-th image, Left Time = {:.2f} hour ...'.format(image_ind, left_time))

        # 预处理部分获取图片路径，读取图片，读取现有的检测结果和属性检测结果
        args.image_file = image_file[:-1]
        im_file = os.path.join(args.image_dir, args.image_file)
        im = Image.open(im_file)
        im_cp = np.array(imageio.v2.imread(im_file))  # HWC

        # mean-std normalize the input image (batch-size: 1)
        if im_cp.shape[-1] == 3:  # 灰度图只是二维array，如512*420
            img = transform1(im).unsqueeze(0)
        else:
            img = transform2(im).unsqueeze(0)   # 灰度图扩充3维

        # propagate through the model
        outputs = model(img)  # {dict: 8}

        # keep only predictions with 0.+ confidence，如下取0.3
        probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]  # sub_logits: [1 200 152]，probas_sub：200 151
        probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]  # obj_logits: [1 200 4]，probas_obj：200 151
        threshold = 0.2
        keep = torch.logical_and(probas.max(-1).values > threshold, torch.logical_and(probas_sub.max(-1).values > threshold,
                                                                            probas_obj.max(-1).values > threshold))
        # convert boxes from [0; 1] to image scales
        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)  # sub_boxes [1 200 4]
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)  # obj_boxes [1 200 4]


        keep_queries = torch.nonzero(keep, as_tuple=True)[0]
        indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
        keep_queries = keep_queries[indices]

        with torch.no_grad():
            for idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                    zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
                bbox_xyxy = [float(sxmin), float(symin), float(sxmax), float(symax)]
                # bbox_xywh = [float(sxmin), float(symin), float(sxmax - sxmin), float(symax - symin)]
                expression_string = CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()]
                tmp_expression_sample = [args.image_file, 'useless placeholder', bbox_xyxy, expression_string, 'useless placeholder']
                expression_train_samples.append(tmp_expression_sample)

        # fig.tight_layout()
        # plt.show()
        # plt.savefig(img_path+'_pred.png')

    output_path = os.path.join(args.out_path, 'top{}'.format(topk))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    torch.save(expression_train_samples,
               os.path.join(output_path, '{}_train_pseudo_split_{}.pth'.format(args.vg_dataset, args.split_ind)))
    print('Save file to {}'.format(
        os.path.join(output_path, '{}_train_pseudo_split_{}.pth'.format(args.vg_dataset, args.split_ind))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
