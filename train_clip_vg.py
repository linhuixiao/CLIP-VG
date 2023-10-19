import os
import time
import math
import json
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, validate


def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-VG Args', add_help=False)
    parser.add_argument('--sup_type', default='un', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    # TODO: 这里面bert，resnet，transformer部分全都设置了学习率，在训练时，加载模型之后还是会全都训练
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-5, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--lr_exponential', default=0.9, type=float, help='lr exponential')
    parser.add_argument('--clip_max_norm', default=0., type=float, help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_drop', default=60, type=int)

    # Augmentation options
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
    parser.add_argument('--model_name', type=str, default='TransVG',
                        help="Name of model to be exploited.")

    # Transformers in two branches
    """两个分支 bert 和 detr 的数量 """
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
    parser.add_argument('--hidden_dim', default=256, type=int,
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
    """ VL 融合模块参数 """
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=512, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--vl_dec_layers', default=6, type=int,
                        help='Number of decoders in the vision-language transformer')

    # Dataset parameters
    """ 数据集根目录 """
    # TODO: --data_root /hdd/lhxiao/pseudo-q/data, Flickr30k  other  pseudo_samples，只使用了里面的图片，并没有使用里面的分割
    parser.add_argument('--data_root', type=str, default='./data/image_data/', help='path to ReferIt splits data folder')
    # TODO： Split root 是拆分索引文件的文件目录，位于 ./data
    # TODO: --split_root /hdd/lhxiao/pseudo-q/data/pseudo_samples, flickr  gref  gref_umd  referit  unc  unc+
    #  这个目录和原始 ref 目录下的 'refs(google).p'  'refs(umd).p' 不同，划分更细致 unc_testA.pth  unc_testB.pth
    #  unc_train_pseudo.pth  unc_val.pth，这个目录中的pth文件包含了 拆分的图片索引、bbox和引用表达的句子
    parser.add_argument('--split_root', type=str, default='./data/pseudo_samples/',  help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str, help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument('--max_query_len', default=77, type=int,
                        help='maximum time steps (lang length) per batch')

    # Prompt Engineering
    parser.add_argument('--prompt', type=str, default='', help="Prompt template")

    # Cross module structure
    parser.add_argument('--cross_num_attention_heads', default=1, type=int, help='cross module attention head number')
    # parser.add_argument('--cross_vis_hidden_size', default=256, type=int, help='cross module hidden size')
    parser.add_argument('--cross_vis_hidden_size', default=512, type=int, help='cross module hidden size')
    # parser.add_argument('--cross_text_hidden_size', default=768, type=int, help='cross module hidden size')
    parser.add_argument('--cross_text_hidden_size', default=512, type=int, help='cross module hidden size')
    parser.add_argument('--cross_hidden_dropout_prob', default=0.1, type=float, help='cross module hidden dropout probability')
    parser.add_argument('--cross_attention_probs_dropout_prob', default=0.1, type=float)

    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--retrain', default='', help='retrain from checkpoint')
    # TODO： detr model 和 bert model 是什么意思
    # parser.add_argument('--detr_model', default='./saved_models/detr-r50.pth', type=str, help='detr model')
    # parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    """ num workers 默认设为 8 """
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
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
    print('### INFO ### torch.backends.cudnn.benchmark = {}'.format(torch.backends.cudnn.benchmark))

    # build model
    # TODO： 核心一步，初始化模型结构，TransVG(args)
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    # args.distributed 在 init_distributed_mode 中赋值为 true，args.gpu 也是
    if args.distributed:
        # 分布式数据并行
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # TODO：统计的参数默认都是需要梯度的，但是 Bert 在模型内部（bert.py）设置了根据学习率是否需要更新
    n_parameters_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of requires_grad params: ', n_parameters_grad)
    print('number of all params: ', n_parameters)

    # TODO：开始构建训练模型，从此开始代码和测试时不一样
    # TODO：什么意思？
    visu_cnn_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" in n) and ("backbone" in n) and p.requires_grad)]
    visu_tra_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" in n) and ("backbone" not in n) and p.requires_grad)]
    text_tra_param = [p for n, p in model_without_ddp.named_parameters() if (("textmodel" in n) and p.requires_grad)]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]

    param_list = [{"params": rest_param},
                  {"params": visu_cnn_param, "lr": args.lr_visu_cnn},
                  {"params": visu_tra_param, "lr": args.lr_visu_tra},
                  {"params": text_tra_param, "lr": args.lr_bert},
                  ]
    visu_param = [p for n, p in model_without_ddp.named_parameters() if "visumodel" in n and p.requires_grad]
    text_param = [p for n, p in model_without_ddp.named_parameters() if "textmodel" in n and p.requires_grad]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]

    # using RMSProp or AdamW
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    # using polynomial lr scheduler or half decay every 10 epochs or step
    # 多项式学习率调度，或者每10 epoch 衰减一半
    if args.lr_scheduler == 'poly':
        lr_func = lambda epoch: (1 - epoch / args.epochs) ** args.lr_power
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'halfdecay':
        lr_func = lambda epoch: 0.5 ** (epoch // (args.epochs // 10))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'cosine':
        lr_func = lambda epoch: 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'exponential':
        lr_func = lambda epoch: args.lr_exponential ** epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    # build dataset
    print('build dataset...')
    # TODO: 有监督和无监督版本的区别
    # TODO: 构建训练数据集和验证数据集，此部分代码和测试不一样，这里只是把原先的train变成train_pseudo，后续路径同样会这样变化
    if (args.sup_type == 'full'):
        print("perform fullly supervised setting.")
        dataset_train = build_dataset('train', args)
    else:  # un
        dataset_train = build_dataset('train_pseudo', args)

    dataset_val = build_dataset('val', args)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  = build_dataset('test', args)

    if args.distributed:
        # 分布式数据并行，分布式数据并行采样
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    # TODO：为什么在训练时，需要使用 batch_sampler，而在测试时直接用 DataLoader 进行batch_size load
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # TODO: 注意，如下并没有对bert模型部分进行判定，因为bert部分在模型声明时已经加载进入，同时设置了参数可根据学习率是否大于0更新
    best_accu = 0
    if args.resume:
        # 断点续训
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            # args.start_epoch = 0  # 微调训练
        val_stats = validate(args, model, data_loader_val, device)
        best_accu = val_stats['accu']
        print("best_accu: {}".format(best_accu))
    # TODO: 这里的 DETR模型是 visual_model部分，属于定制的预训练模型，包含了 Resnet 和 TransformerEncoder，需要对state_dict重新加载，
    #  同时训练时也设置了会更新
    # TODO: 开始不同
    # elif args.detr_model is not None:
    #     # 用训好的 detr 模型
    #     checkpoint = torch.load(args.detr_model, map_location='cpu')
    #     missing_keys, unexpected_keys = model_without_ddp.visumodel.load_state_dict(checkpoint['model'], strict=False)
    #     print('Missing keys when loading detr model:')
    #     print(missing_keys)

    # 写法1：有bug
    # if args.retrain:
        # 加载部分模型参数，代码参考：
        # https://zhuanlan.zhihu.com/p/34147880
        # https://blog.csdn.net/LXX516/article/details/80124768
        # https://blog.csdn.net/amds123/article/details/63684716
        # 进一步参考 vlmo 写法（暂未弄明白）：
        # https://github.com/microsoft/unilm/blob/master/vlmo/vlmo/modules/vlmo_module.py
        # https://github.com/microsoft/unilm/blob/master/beit/utils.py
        # checkpoint = torch.load(args.retrain, map_location='cpu')
        # print("model structure: \n", checkpoint['model'].items())
        # print("model_without_ddp structure: \n", model_without_ddp)  # 模型参数结构
        # dict_tmp = model.state_dict().copy()
        # print(dict_tmp)
        # pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k == 'vl_transformer'}
        # pretrained_dict = {k for k, v in checkpoint['model'].items()}
        # print(pretrained_dict)
        # dict_tmp.update(pretrained_dict)
        # model_without_ddp.load_state_dict(dict_tmp, strict=False)

    if args.retrain:
        model_cache = build_model(args)
        model_cache.to(device)
        checkpoint = torch.load(args.retrain, map_location='cpu')
        model_cache.load_state_dict(checkpoint['model'])
        # print("model.vl_transformer before :\n", model_without_ddp.vl_transformer)  # 打印模型结构
        # print("model.vl_transformer before :\n", model_without_ddp.vl_transformer.values())  # 打印模型具体权值
        # model.vl_transformer = model_cache.vl_transformer  # 这种写法是错的，训练 1 个 ep acc 为 25。load 权值时应该给 without DDP 模型
        model_without_ddp.vl_transformer = model_cache.vl_transformer  # 这种写法可以，训练 1 个 ep acc 为 76
        # 也可以采用 tansvg 中对 detr 模型的 load 代码，使用 model_without_ddp.vl_transformer.load_state_dict()：
        # missing_keys, unexpected_keys = model_without_ddp.visumodel.load_state_dict(checkpoint['model'], strict=False)
        # print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nmodel.vl_transformer after :\n", model_without_ddp.vl_transformer)  # 打印模型结构
        # print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nmodel.vl_transformer after :\n", model_without_ddp.vl_transformer.state_dict())  # 打印模型具体权值

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(str(args) + "\n")

    print("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        # TODO: 核心部分，开始训练，此时传入的 data_loader_train都是加载好的 image_data 和 text_data
        train_stats = train_one_epoch(args, model, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        # TODO：进行验证
        val_stats = validate(args, model, data_loader_val, device)

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'validation_{k}': v for k, v in val_stats.items()},
                     'n_parameters': n_parameters}
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            # 当前epoch的模型有可能要保存三份：best_checkpoint，checkpoint，逢整数的checkpoint，实际只需保存2份即可
            checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint.pth')]
            # extra checkpoint before LR drop and every 10 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                # checkpoint_paths.append(os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            if val_stats['accu'] > best_accu:
                checkpoint_paths.append(os.path.join(args.output_dir, 'best_checkpoint.pth'))
                best_accu = val_stats['accu']

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu']
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIP-VG training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
