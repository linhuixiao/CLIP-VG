import os
import torch
import argparse


parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
parser.add_argument('root_path', default=None, type=str)
parser.add_argument('dataset', default=None, type=str)
args = parser.parse_args()


def reform_sample():
    file = torch.load(os.path.join(args.root_path, '{}_train_pseudo.pth'.format(args.dataset)))
    os.system('mv {} {}'.format(os.path.join(args.root_path, '{}_train_pseudo.pth'.format(args.dataset)),
                                os.path.join(args.root_path, '{}_train_pseudo_origin.pth'.format(args.dataset))))
    print('### INFO ### Length of pseudo train sample: {}'.format(len(file)))

    reformat_file = []
    for sample in file:
        original_bbox = sample[2]
        # 检查无效参数，没有干啥，只是打印出来
        if original_bbox[0] > original_bbox[2] or original_bbox[1] > original_bbox[3]:
            print('### INFO ### Bad bbox of {} = {}'.format(sample[0], original_bbox))

        # 如果不是 referit 和 flickr，即如果是unc/+/g，则将坐标从 xyxy 转为 xywh
        if not (args.dataset == 'referit' or args.dataset == 'flickr'):
            bbox = [sample[2][0], sample[2][1], sample[2][2] - sample[2][0], sample[2][3] - sample[2][1]]  # xyxy2xywh
            tmp_sample = [sample[0], sample[1], sample[2], sample[3], sample[4]]
            tmp_sample[2] = bbox
        else:  # 如果是 flickr， 则 剔除 useless holder
            if args.dataset == 'flickr':
                tmp_sample = [sample[0], sample[2], sample[3]]
            else:  # referit，保持不变
                tmp_sample = [sample[0], sample[1], sample[2], sample[3], sample[4]]

        reformat_file.append(tmp_sample)

    return reformat_file


if __name__ == '__main__':
    reformat_file = reform_sample()
    torch.save(reformat_file, os.path.join(args.root_path, '{}_train_pseudo.pth'.format(args.dataset)))
