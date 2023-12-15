'''
# @title Drive Downloader

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

download_with_pydrive = True  # @param {type:"boolean"}


class Downloader(object):
    def __init__(self, use_pydrive):
        self.use_pydrive = use_pydrive

        if self.use_pydrive:
            self.authenticate()

    def authenticate(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def download_file(self, file_id, file_dst):
        if self.use_pydrive:
            downloaded = self.drive.CreateFile({'id': file_id})
            downloaded.FetchMetadata(fetch_all=True)
            downloaded.GetContentFile(file_dst)
        else:
            !gdown - -id $file_id - O $file_dst


downloader = Downloader(download_with_pydrive)
'''

# @title Imports

import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
# from google.colab import files
import skimage.io as io
import PIL.Image
from IPython.display import Image
import matplotlib.pyplot as plt
import time

import argparse

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device
print(CUDA)

current_directory = os.getcwd()
save_path = os.path.join(os.path.dirname(current_directory), "pretrained_models")
os.makedirs(save_path, exist_ok=True)
model_path = os.path.join(save_path, 'model_wieghts.pt')


# @title Model

class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: T) -> T:
        return self.model(x)


class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP(
                (prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


# @title Caption prediction

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    # print("generated_list shape: ", generated_list.shape)
    # print("\n generated_list: ", generated_list)
    return generated_list[0]


'''
#@title Choose pretrained model - COCO or Coneptual captions

# pretrained_model = 'Conceptual captions'  # @param ['COCO', 'Conceptual captions']
pretrained_model = 'COCO'  # @param ['COCO', 'Conceptual captions']

if pretrained_model == 'Conceptual captions':
  downloader.download_file("14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT", model_path)
else:
  downloader.download_file("1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX", model_path)
'''


def get_args_parser():
    parser = argparse.ArgumentParser('Set CLIP-CAP datapath', add_help=False)
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
    return parser

# @title GPU/CPU
if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIP-CAP inference', parents=[get_args_parser()])
    args = parser.parse_args()

    args.vg_dataset_path = '/hdd/lhxiao/pseudo-q/image_data'
    if args.vg_dataset in ['unc', 'unc+', 'gref', 'gref_umd']:
        args.image_dir = os.path.join(args.vg_dataset_path, 'other/images/mscoco/images/train2014')
    elif args.vg_dataset == 'referit':
        args.image_dir = os.path.join(args.vg_dataset_path, 'referit/images')
    else:  # flickr
        args.image_dir = os.path.join(args.vg_dataset_path, 'Flickr30k/flickr30k-images/')
    args.out_path = '/hdd/lhxiao/pseudo-q/caption_clip-cap_output/{}'.format(args.vg_dataset)
    args.image_list_file = '/hdd/lhxiao/pseudo-q/detection_results/splits/{}/{}_train_imagelist.txt'.format(
        args.vg_dataset, args.vg_dataset)
    train_image_list = open(args.image_list_file, 'r')
    train_image_files = train_image_list.readlines()

    '''Load model'''
    is_gpu = True  # @param {type:"boolean"}
    # @title CLIP model + GPT2 tokenizer
    device = CUDA(0) if is_gpu else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    print("==> loading gpt2 model......")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("==> load gpt2 model complete.")
    # @title Load model weights
    prefix_length = 10
    model = ClipCaptionModel(prefix_length)
    model_path = "/hdd/lhxiao/clip_prefix_caption/checkpoint/coco_weights.pt"
    model.load_state_dict(torch.load(model_path, map_location=CPU))

    model = model.eval()
    device = CUDA(0) if is_gpu else "cpu"
    model = model.to(device)

    # @title Inference
    use_beam_search = False  # @param {type:"boolean"}

    IMG_FILE = '../Images/COCO_val2014_000000060623.jpg'
    print("Begin to processing: ", args.image_list_file)
    start_time = time.time()

    caption_data = []
    for image_ind, image_file in enumerate(train_image_files):
        if image_ind % 100 == 0:
            left_time = ((time.time() - start_time) * (len(train_image_files) - image_ind - 1) / (image_ind + 1)) / 3600
            print('Processing {}-th image, Left Time = {:.2f} hour ...'.format(image_ind, left_time))

        args.image_file = image_file[:-1]
        IMG_FILE = os.path.join(args.image_dir, args.image_file)
        single_img_caption = {}
        single_img_caption['image_id'] = image_ind
        single_img_caption['image_file'] = IMG_FILE
        single_img_caption['captions'] = []

        # image = io.imread(UPLOADED_FILE)
        image = io.imread(IMG_FILE)
        pil_image = PIL.Image.fromarray(image)
        # plt.imshow(pil_image)
        # plt.show()
        # pil_img = Image(filename=UPLOADED_FILE)
        # display(pil_image)

        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            # if type(model) is ClipCaptionE2E:
            #     prefix_embed = model.forward_image(image)
            # else:
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        if use_beam_search:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

        single_img_caption['captions'].append(generated_text_prefix)
        caption_data.append(single_img_caption)
        # print('\n')
        # print(generated_text_prefix)


    output_path = os.path.join(args.out_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    torch.save(caption_data,
               os.path.join(output_path, '{}_caption_data.pth'.format(args.vg_dataset)))
    print('Save file to {}'.format(
        os.path.join(output_path, '{}_caption_data.pth'.format(args.vg_dataset))))

