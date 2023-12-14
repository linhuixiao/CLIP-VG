import torch
import torch.nn as nn
import torch.nn.functional as F
from .vl_transformer import build_vl_transformer

from .clip import *
from torchvision.transforms import Resize
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from collections import OrderedDict


class MultiLevel_Transformer(nn.Module):
    def __init__(self, clip_vit, extract_layer):
        super().__init__()
        heads = clip_vit.width // 64
        self.width = clip_vit.width
        self.layers = clip_vit.layers
        self.resblocks = clip_vit.resblocks
        self.extract_layer = extract_layer

    def forward(self, x: torch.Tensor):
        ml_feature = []
        for i in range(max(self.extract_layer)+1):
            x = self.resblocks[i](x)
            if i in self.extract_layer:
                ml_feature.append(x)
        return ml_feature


class MultiLevel_ImageEncoder_modified(nn.Module):
    def __init__(self, clip_visu_model, extract_layer):
        super().__init__()
        self.input_resolution = clip_visu_model.input_resolution
        self.output_dim = clip_visu_model.output_dim
        self.conv1 = clip_visu_model.conv1
        self.class_embedding = clip_visu_model.class_embedding
        self.positional_embedding = clip_visu_model.positional_embedding
        self.ln_pre = clip_visu_model.ln_pre
        self.transformer = MultiLevel_Transformer(clip_visu_model.transformer, extract_layer)
        self.ln_post = clip_visu_model.ln_post
        self.proj = clip_visu_model.proj
        self.positional_embedding.requires_grad_(True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                       dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND, B L H -> L B H
        ml_x = self.transformer(x)
        x = torch.cat(ml_x, dim=2)
        x = x.permute(1, 0, 2)  # LND -> NLD, L B H -> B 4*L H
        return x


class MutiLevel_TextEncoder_modified(nn.Module):
    def __init__(self, clip_model, extract_layer):
        super().__init__()
        self.transformer = MultiLevel_Transformer(clip_model.transformer, extract_layer)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, x):
        x = self.token_embedding(x).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        ml_x = self.transformer(x)
        x = torch.cat(ml_x, dim=2)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x


class ImageEncoder_modified(nn.Module):
    def __init__(self, clip_visu_model):
        super().__init__()
        self.input_resolution = clip_visu_model.input_resolution
        self.output_dim = clip_visu_model.output_dim
        self.conv1 = clip_visu_model.conv1
        self.class_embedding = clip_visu_model.class_embedding
        self.positional_embedding = clip_visu_model.positional_embedding
        self.ln_pre = clip_visu_model.ln_pre
        self.transformer = clip_visu_model.transformer
        self.ln_post = clip_visu_model.ln_post
        self.proj = clip_visu_model.proj

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # TODO: ModifiedCLIP
        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj

        return x


class TextEncoder_modified(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, x):
        x = self.token_embedding(x).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # TODO: ModifiedCLIP
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        x = x @ self.text_projection

        return x


class ML_CLIP_VG(nn.Module):
    def __init__(self, args):
        super(ML_CLIP_VG, self).__init__()
        print("This is the ML_CLIP_VG model.")
        # CLIP Model name: ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        if (args.model == "ViT-L/14"):
            print("init ViT-L/14")
            self.clip, _ = clip.load("ViT-L/14", device=args.device)
            self.extract_layer = [0, 7, 15, 23]
            self.patch_size = 14
        elif (args.model == "ViT-B/32"):
            print("init ViT-B/32")
            self.clip, _ = clip.load("ViT-B/32", device=args.device)
            self.extract_layer = [0, 3, 7, 11]
            self.patch_size = 32
        else:  # default
            print("init ViT-B/16")
            self.clip, _ = clip.load("ViT-B/16", device=args.device)
            self.extract_layer = [0, 3, 7, 11]
            self.patch_size = 16

        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

        hidden_dim = self.clip.transformer.width
        self.visu_proj = nn.Linear(512, hidden_dim)
        self.text_proj = nn.Linear(self.clip.transformer.width, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)
        self.imsize = args.imsize
        self.num_visu_token = int((self.imsize / self.patch_size) ** 2)
        self.num_text_token = args.max_query_len
        num_total = self.num_visu_token + 1 + self.num_text_token + 1  # VISU token + [cls] + TEXT token + [REG]token
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)

        self.vl_transformer = build_vl_transformer(args)

        self.image_encoder_clip_vg = MultiLevel_ImageEncoder_modified(self.clip.visual, self.extract_layer)
        self.text_encoder_clip_vg = TextEncoder_modified(self.clip)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.ml_visu_proj = nn.Linear(len(self.extract_layer) * self.clip.visual.transformer.width, hidden_dim)

    def tensorize_inputs(self, images: NestedTensor, texts: NestedTensor):
        image_tensors = images.tensors
        texts_tensors = texts.tensors

        return image_tensors, texts_tensors

    def get_masks(self, images: NestedTensor, texts: NestedTensor):
        torch_resize = Resize([int(self.imsize / self.patch_size), int(self.imsize / self.patch_size)])
        visu_masks = torch_resize(images.mask)  # 14 * 14 = 196， or， 16 * 16 = 256
        visu_masks = visu_masks.to(torch.bool)
        visu_masks = visu_masks.flatten(1)  # visu_mask：B*L, torch.Size([B, 196])
        text_masks = texts.mask.to(torch.bool)
        text_masks = ~text_masks
        text_masks = text_masks.flatten(1)  # text_mask：B*L, torch.Size([B, 77])
        assert text_masks is not None

        return visu_masks, text_masks

    def forward(self, img_data, text_data):
        batch_size = img_data.tensors.shape[0]
        image_tensors, text_tensors = self.tensorize_inputs(img_data, text_data)
        image_features = self.image_encoder_clip_vg(image_tensors.type(self.clip.dtype))  # B * 197 * 512
        text_features = self.text_encoder_clip_vg(text_tensors)  # B * 77 * 512
        visu_src = self.ml_visu_proj(image_features.float())  # L B 4H -> L B H
        text_src = self.text_proj(text_features.float())  # B * 77 * 512
        # permute BxLenxC to LenxBxC
        visu_src = visu_src.permute(1, 0, 2)  # 197 * 4 * 512
        text_src = text_src.permute(1, 0, 2)  # 77 * 4 * 512
        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, batch_size, 1)  # 1 * B * hidden_dim
        # (1 + 77 + 197) * B * 512 = 275 * B * 512; VIT-L/14: (1 + 77 + 257) * B * 768
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        visu_mask, text_mask = self.get_masks(img_data, text_data)
        tgt_mask = torch.zeros((batch_size, 1)).to(tgt_src.device).to(torch.bool)
        cls_mask = torch.zeros((batch_size, 1)).to(tgt_src.device).to(torch.bool)
        vl_mask = torch.cat([tgt_mask, text_mask, cls_mask, visu_mask], dim=1)
        # (1 + 77 + 1 + 196) * B * H = 275 * B * 512
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        vg_hs = vg_hs[0]
        pred_box = self.bbox_embed(vg_hs).sigmoid()

        return pred_box


# TODO: Add learnable prompt for ML_CLIP_VG, benefit for the following researchers.
#  It's observed that has several performance gains on RefCOCOg.
class ML_CLIP_VG_PROMPT(nn.Module):
    def __init__(self, args):
        super(ML_CLIP_VG_PROMPT, self).__init__()
        print("This is the ML_CLIP_VG_PROMPT model.")
        if (args.model == "ViT-L/14"):
            print("init ViT-L/14")
            self.clip, _ = clip.load("ViT-L/14", device=args.device)
            self.extract_layer = [0, 7, 15, 23]
            self.patch_size = 14
        elif (args.model == "ViT-B/32"):
            print("init ViT-B/32")
            self.clip, _ = clip.load("ViT-B/32", device=args.device)
            self.extract_layer = [0, 3, 7, 11]
            self.patch_size = 32
        else:  # default
            print("init ViT-B/16")
            self.clip, _ = clip.load("ViT-B/16", device=args.device)
            self.extract_layer = [0, 3, 7, 11]
            self.patch_size = 16

        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

        hidden_dim = self.clip.transformer.width
        self.visu_proj = nn.Linear(512, hidden_dim)
        self.text_proj = nn.Linear(self.clip.transformer.width, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.prompt_length = 6  # 1 + 5
        prompt_init_vector = [49406, 3797,  4341,   851, 17436,   531]  # "fine region corresponds to" initial
        language_prompt = torch.Tensor(prompt_init_vector).to(args.device)
        self.language_prompt = nn.Parameter(language_prompt)
        self.language_prompt.requires_grad_(True)

        self.imsize = args.imsize
        self.num_visu_token = int((self.imsize / self.patch_size) ** 2)
        self.num_text_token = args.max_query_len
        num_total = self.num_visu_token + 1 + self.num_text_token + 1  # visu token + [cls] + text token + [REG]token
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)

        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.image_encoder_clip_vg = MultiLevel_ImageEncoder_modified(self.clip.visual, self.extract_layer)
        self.text_encoder_clip_vg = TextEncoder_modified(self.clip)
        self.ml_visu_proj = nn.Linear(len(self.extract_layer) * self.clip.visual.transformer.width, hidden_dim)

    def tensorize_inputs(self, images: NestedTensor, texts: NestedTensor):
        image_tensors = images.tensors
        texts_tensors = texts.tensors

        return image_tensors, texts_tensors

    def get_masks(self, images: NestedTensor, texts: NestedTensor):
        torch_resize = Resize([int(self.imsize / self.patch_size), int(self.imsize / self.patch_size)])
        visu_masks = torch_resize(images.mask)
        visu_masks = visu_masks.to(torch.bool)
        visu_masks = visu_masks.flatten(1)  # visu_mask：B*L, torch.Size([B, 196])
        # support learnable prompt
        batch_size = texts.tensors.shape[0]
        prompt_mask = torch.ones((batch_size, self.prompt_length)).to(texts.tensors.device)
        text_masks = torch.cat([prompt_mask, texts.mask], dim=1)[:, :self.num_text_token]
        text_masks = text_masks.to(torch.bool)
        text_masks = ~text_masks
        text_masks = text_masks.flatten(1)
        assert text_masks is not None

        return visu_masks, text_masks

    def forward(self, img_data, text_data):
        batch_size = img_data.tensors.shape[0]  # 得到batch_size
        image_tensors, text_tokens = self.tensorize_inputs(img_data, text_data)
        image_features = self.image_encoder_clip_vg(image_tensors.type(self.clip.dtype))  # B * 197 * 512
        text_prompt = self.language_prompt.repeat(batch_size, 1).long()  # B * 5 * hidden_dim
        text_tokens = torch.cat([text_prompt, text_tokens], dim=1)[:, :self.num_text_token]
        text_features = self.text_encoder_clip_vg(text_tokens)  # B * 77 * 512

        visu_src = self.ml_visu_proj(image_features.float())  # L B 4H -> L B H
        text_src = self.text_proj(text_features.float())
        visu_src = visu_src.permute(1, 0, 2)  # 197 * 4 * 512
        text_src = text_src.permute(1, 0, 2)  # 77 * 4 * 512
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, batch_size, 1)  # 1 * B * hidden_dim
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        visu_mask, text_mask = self.get_masks(img_data, text_data)

        tgt_mask = torch.zeros((batch_size, 1)).to(tgt_src.device).to(torch.bool)
        cls_mask = torch.zeros((batch_size, 1)).to(tgt_src.device).to(torch.bool)
        vl_mask = torch.cat([tgt_mask, text_mask, cls_mask, visu_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        vg_hs = vg_hs[0]
        pred_box = self.bbox_embed(vg_hs).sigmoid()

        return pred_box


# TODO: We provide the last layer feature version benefit for the following researchers to perform ablation study.
class CLIP_VG(nn.Module):
    def __init__(self, args):
        super(CLIP_VG, self).__init__()
        self.clip, _ = clip.load("ViT-B/16", device=args.device)
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        hidden_dim = args.vl_hidden_dim
        self.visu_proj = nn.Linear(512, hidden_dim)
        self.text_proj = nn.Linear(512, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        divisor = 16
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len
        num_total = self.num_visu_token + 1 + self.num_text_token + 1  # visu token + [cls] + text token + [REG]token
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def tensorize_inputs(self, images: NestedTensor, texts: NestedTensor):
        image_tensors = images.tensors
        texts_tensors = texts.tensors

        return image_tensors, texts_tensors

    def get_masks(self, images: NestedTensor, texts: NestedTensor):
        torch_resize = Resize([14, 14])
        visu_masks = torch_resize(images.mask)
        visu_masks = visu_masks.to(torch.bool)
        visu_masks = visu_masks.flatten(1)
        text_masks = texts.mask.to(torch.bool)
        text_masks = ~text_masks
        text_masks = text_masks.flatten(1)
        assert text_masks is not None

        return visu_masks, text_masks

    def forward(self, img_data, text_data):
        batch_size = img_data.tensors.shape[0]
        image_tensors, text_tensors = self.tensorize_inputs(img_data, text_data)
        image_features = self.clip.encode_image(image_tensors)  # B * 197 * 512
        text_features = self.clip.encode_text(text_tensors)  # B * 77 * 512
        visu_src = self.visu_proj(image_features.float())
        text_src = self.text_proj(text_features.float())
        visu_src = visu_src.permute(1, 0, 2)  # 197 * 4 * 512
        text_src = text_src.permute(1, 0, 2)  # 77 * 4 * 512
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, batch_size, 1)  # 1 * B * hidden_dim
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        visu_mask, text_mask = self.get_masks(img_data, text_data)
        tgt_mask = torch.zeros((batch_size, 1)).to(tgt_src.device).to(torch.bool)
        cls_mask = torch.zeros((batch_size, 1)).to(tgt_src.device).to(torch.bool)
        vl_mask = torch.cat([tgt_mask, text_mask, cls_mask, visu_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        vg_hs = vg_hs[0]
        pred_box = self.bbox_embed(vg_hs).sigmoid()

        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
