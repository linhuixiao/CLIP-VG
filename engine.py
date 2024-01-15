# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
from utils.box_utils import xywh2xyxy


def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, max_norm: float = 0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # data processing
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        img_data, text_data, target = batch

        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        # model forward
        output = model(img_data, text_data)

        loss_dict = loss_utils.trans_vg_loss(output, target)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:  # max_norm defaults to 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        pred_boxes = model(img_data, text_data)
        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, target)

        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        output = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])

    return accuracy


@torch.no_grad()  # this is iou-based
def evaluate_for_filtering(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):

    # TODO: ATTENTION: evaluate_for_filtering() only can running on the single GPU env, to prevent the data missing.
    #   besides, evaluate_for_filtering must use collate_fn=utils.collate_fn_filtering

    model.eval()
    pred_box_list = []
    gt_box_list = []
    img_file_list = []
    phrase_list = []
    img_iou_list = []
    bbox_ori_list = []

    train_pseudo_bad_label_0_0 = []
    train_pseudo_bad_label_0_4 = []

    train_pseudo_refine = []
    reliability = [i / 10.0 for i in range(11)]  # default delta is 0.1, the reliability interval is from 0.0 to 1.0
    for i in reliability:
        train_pseudo_refine.append([])

    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, img_file, phrase, bbox_ori = batch
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        output = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())
        for img_name in img_file:
            img_file_list.append(img_name)
        for phrase_i in phrase:
            phrase_list.append(phrase_i)
        for bbox_ori_i in bbox_ori:
            bbox_ori_list.append(bbox_ori_i)

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num, iou_list = eval_utils.trans_vg_eval_test_iou(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)
    for i in range(len(img_file_list)):
        if not (args.dataset == 'flickr'):  # unc/+/g， bbox xywh, referit:x1y1
            img_iou_list.append([img_file_list[i], 'useless placeholder', bbox_ori_list[i], phrase_list[i],
                                 gt_boxes[i].cpu(), pred_boxes[i].cpu(), float(iou_list[i])])
            if float(iou_list[i]) == 0:
                tmp_pseudo_label = [img_file_list[i], 'useless placeholder', bbox_ori_list[i], phrase_list[i],
                                    'useless placeholder', float(iou_list[i])]
                train_pseudo_bad_label_0_0.append(tmp_pseudo_label)
            if float(iou_list[i]) < 0.4:
                tmp_pseudo_label = [img_file_list[i], 'useless placeholder', bbox_ori_list[i], phrase_list[i],
                                    'useless placeholder', float(iou_list[i])]
                train_pseudo_bad_label_0_4.append(tmp_pseudo_label)

            for j in range(len(reliability) - 1):
                if float(iou_list[i]) > reliability[j]:
                    tmp_pseudo_label = [img_file_list[i], 'useless placeholder', bbox_ori_list[i], phrase_list[i],
                                        'useless placeholder']
                    train_pseudo_refine[j].append(tmp_pseudo_label)
            if float(iou_list[i]) == 1.0:
                tmp_pseudo_label = [img_file_list[i], 'useless placeholder', bbox_ori_list[i], phrase_list[i],
                                    'useless placeholder']
                train_pseudo_refine[10].append(tmp_pseudo_label)

        else:
            if args.dataset == 'flickr':
                # tmp_sample = [sample[0], sample[2], sample[3]], ['img_file', [x1y1x2y2 bbox], 'expression']
                img_iou_list.append([img_file_list[i], 'useless placeholder', bbox_ori_list[i], phrase_list[i],
                                     gt_boxes[i].cpu(), pred_boxes[i].cpu(), float(iou_list[i])])
                if float(iou_list[i]) == 0:
                    tmp_pseudo_label = [img_file_list[i], 'useless placeholder', bbox_ori_list[i], phrase_list[i],
                                        'useless placeholder', float(iou_list[i])]
                    train_pseudo_bad_label_0_0.append(tmp_pseudo_label)
                if float(iou_list[i]) < 0.4:
                    tmp_pseudo_label = [img_file_list[i], 'useless placeholder', bbox_ori_list[i], phrase_list[i],
                                        'useless placeholder', float(iou_list[i])]
                    train_pseudo_bad_label_0_4.append(tmp_pseudo_label)

                for j in range(len(reliability) - 1):
                    if float(iou_list[i]) > reliability[j]:
                        tmp_pseudo_label = [img_file_list[i], bbox_ori_list[i], phrase_list[i]]
                        train_pseudo_refine[j].append(tmp_pseudo_label)
                if float(iou_list[i]) == 1.0:
                    tmp_pseudo_label = [img_file_list[i], bbox_ori_list[i], phrase_list[i]]
                    train_pseudo_refine[10].append(tmp_pseudo_label)

    print("img_file_list: ", len(img_file_list))
    print("iou_list: ", iou_list.size())
    print("img_iou_list size: ", len(img_iou_list))
    print("train_pseudo_bad_label_0_0 size: ", len(train_pseudo_bad_label_0_0))
    print("train_pseudo_bad_label_0_4 size: ", len(train_pseudo_bad_label_0_4))
    for j in range(len(reliability)):
        print("the number of train_pseudo_refine sample when reliability > ",  str(reliability[j]), " is: ",
              len(train_pseudo_refine[j]))

    torch.save(train_pseudo_bad_label_0_0, os.path.join(args.output_dir, '{}_train_pseudo_bad_label_0_0.pth'.format(args.dataset)))
    torch.save(train_pseudo_bad_label_0_4, os.path.join(args.output_dir, '{}_train_pseudo_bad_label_0_4.pth'.format(args.dataset)))

    for j in range(len(reliability)):
        torch.save(train_pseudo_refine[j], os.path.join(args.output_dir,
                   '{}_train_pseudo_refine_{}.pth'.format(args.dataset, str(reliability[j]).replace(".", "_"))))

    torch.save(img_iou_list, os.path.join(args.output_dir, '{}_train_pseudo_iou_result.pth'.format(args.dataset)))

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)
    all_img_iou_list = []
    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    print("img_iou_list: ", len(all_img_iou_list))

    return accuracy, all_img_iou_list



