import time
import json
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import evaluate

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'  # batch_size max=768，0、1在用


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--sup_type', default='un', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)  # 融合模块学习率
    parser.add_argument('--lr_bert', default=0., type=float)
    parser.add_argument('--lr_visu_cnn', default=0., type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--lr_drop', default=80, type=int)

    # Augmentation options
    # TODO: 暂时还没有验证
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")
    # TODO: 新增
    # only support ViT-B/16 and ViT-L/14
    parser.add_argument('--model', type=str, default='ViT-B/16',
                        help="Name of model to be exploited.")
    # Model parameters
    parser.add_argument('--model_name', type=str, default='CLIP-VG',
                        help="Name of model to be exploited.")

    # Transformers in two branches
    """两个分支 bert 和 detr 的数量"""
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)

    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    # 默认是用 sine embedding
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    # TODO: DETR 编码层和解码层的数量，解码层数量为0
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int,
                        help="Number of decoding layers in the transformer")
    """ 前馈层的维度 """
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    # 查询的数量
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    """ 图像的大小 """
    parser.add_argument('--imsize', default=224, type=int, help='image size')
    """ embedding size"""
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    # Vision-Language Transformer
    parser.add_argument('--use_vl_type_embed', action='store_true', help="If true, use vl_type embedding")
    """ VL 融合模块参数 """
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    # TODO：这里为了对齐也从256 改成512
    parser.add_argument('--vl_hidden_dim', default=512, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--vl_dec_layers', default=6, type=int,
                        help='Number of decoders in the vision-language transformer')

    # Dataset parameters
    # TODO: --data_root /hdd/lhxiao/pseudo-q/data, Flickr30k  other  pseudo_samples
    parser.add_argument('--data_root', type=str, default='./data/image_data/',
                        help='path to ReferIt splits data folder')
    # TODO: --split_root /hdd/lhxiao/pseudo-q/data/pseudo_samples, flickr  gref  gref_umd  referit  unc  unc+
    #  这个目录和原始 ref 目录下的 'refs(google).p'  'refs(umd).p' 不同，划分更细致 unc_testA.pth  unc_testB.pth
    #  unc_train_pseudo.pth  unc_val.pth，这个目录包含了 拆分和引用表达的句子
    parser.add_argument('--split_root', type=str, default='./data/pseudo_samples/',
                        help='location of pre-parsed dataset info')
    # TODO: 数据集选择
    parser.add_argument('--dataset', default='referit', type=str, help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--max_query_len', default=77, type=int,
                        help='maximum time steps (lang length) per batch')

    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default='./saved_models/detr-r50.pth', type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    """ num workers 默认设为 2 """
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # evalutaion options
    # TODO: 评估子集的选择
    parser.add_argument('--eval_set', default='test', type=str)  # 'testA', 'testB', 'val'
    parser.add_argument('--eval_model', default='', type=str)

    # TODO: 从这里开始不一样
    # Prompt Engineering
    parser.add_argument('--prompt', type=str, default='{pseudo_query}',
                        help="Prompt template")

    # Cross module structure
    # parser.add_argument('--cross_num_attention_heads', default=1, type=int, help='cross module attention head number')
    # parser.add_argument('--cross_vis_hidden_size', default=256, type=int, help='cross module hidden size')
    # parser.add_argument('--cross_text_hidden_size', default=768, type=int, help='cross module hidden size')
    # parser.add_argument('--cross_hidden_dropout_prob', default=0.1, type=float,
    #                     help='cross module hidden dropout probability')
    # parser.add_argument('--cross_attention_probs_dropout_prob', default=0.1, type=float)
    # TODO: 不一样结束

    return parser


def main(args):
    """ GPU 分布式初始化代码 """
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if (args.model == "ViT-L/14" or args.model == "ViT-L/14@336px"):
        args.vl_hidden_dim = 768

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    # seed 默认是 13
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    # TODO： 核心一步，初始化模型结构，CLIP-VG(args)
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        # 分布式数据并行
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of all params: ', n_parameters)

    # build dataset
    # TODO：构建数据集，build_dataset load 数据拆分文件，获取图像位置， args.eval_set = val, testA, testB, 就是split
    dataset_test = build_dataset(args.eval_set, args)

    if args.distributed:
        # 分布式数据并行，分布式数据并行采样
        # dataset_test 是数据集的一个对象，分布式数据集采样器
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, args.batch_size, drop_last=False)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    # TODO： 加载 checkpoint
    model_without_ddp.load_state_dict(checkpoint['model'])
    print("Current model training epoch is: ", checkpoint['epoch'])

    # output log
    eval_model = args.eval_model
    eval_model = eval_model.split('/')[-1].split('.')[0]
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_{}_{}_{}_log.txt".format(args.dataset, args.eval_set, eval_model)).open("a") as f:
            f.write(str(args) + "\n")
            f.flush()

    start_time = time.time()

    # perform evaluation
    # TODO：核心推理计算
    accuracy = evaluate(args, model, data_loader_test, device)

    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Testing time {}'.format(total_time_str))

        log_stats = {'test_model:': args.eval_model,
                     '%s_set_accuracy'%args.eval_set: accuracy,
                     }
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "eval_{}_{}_{}_log.txt".format(args.dataset, args.eval_set, eval_model)).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIP-VG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        # pathlib 中循环创建目录的代码
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
