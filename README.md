# CLIP-VG: Self-paced Curriculum Adapting of CLIP for Visual Grounding
<p align="center"> <img src='docs/model.jpg' align="center" width="70%"> </p>
CLIP for Visual Grounding.

This repository is the official Pytorch implementation for the paper [**CLIP-VG: Self-paced Curriculum Adapting of CLIP 
for Visual Grounding**](https://arxiv.org/abs/2305.08685). 
(Primary Contact: [Linhui Xiao](https://github.com/linhuixiao))


<h3 align="left">
Links: <a href="https://arxiv.org/abs/2305.08685">arXiv</a> 
</h3>

**Please leave a <font color='orange'>STAR ⭐</font> if you like this project!**


## Highlight
- **CLIP for Visual Grounding.** a state-of-the-art baseline for unsupervised and fully supervised visual grounding.
- **Single-source and Multi-source pseudo-language labels.** The generation and usage of multi-source pseudo-labels.
- **Self-paced Curriculum Adapting Algorithm.** A plugin-like algorithmic idea that can be applied to any pseudo-label scenario.

## Update
- The code and models will be released soon ...


## TODO
- [ ] Release model code and inference code.
- [ ] Release unsupervised and fully supervised checkpoints.
- [ ] Release the complete multi-source pseudo-language labels and its generation code.
- [ ] Release the reliability measurement code.
- [ ] Release the self-paced training code.



## Introduction

In order to utilize vision and language pre-trained models to address the grounding problem, and reasonably take 
advantage of pseudo-labels, we propose **CLIP-VG**, a novel method that can conduct self-paced curriculum adapting of CLIP 
with pseudo-language labels. We propose a simple yet efficient end-to-end network architecture to realize the transfer 
of CLIP to the visual grounding. Based on the CLIP-based architecture, we further propose single-source and 
multi-source curriculum adapting algorithms, which can progressively find more reliable pseudo-labels to learn an 
optimal model, thereby achieving a balance between reliability and diversity for the pseudo-language labels. Our method 
outperforms the current state-of-the-art unsupervised method by a significant margin on RefCOCO/+/g datasets in both 
single-source and multi-source scenarios. Furthermore, our approach even outperforms existing weakly supervised methods.
For more details. please refer to [our paper](https://arxiv.org/abs/2305.08685).

## Usage
Instructions for datasets preparation and script to run evaluation and training will be found at [Usage Instructions](docs/Usage.md)

## Checkpoints
### unsupervised setting

#### Single-source scenario

| Dateset  | RefCOCO   | RefCOCO+  | RefCOCOg  | ReferIt   | Flickr    |
|----------|-----------|-----------|-----------|-----------|-----------|
| url      | [model]() | [model]() | [model]() | [model]() | [model]() |   
| size     | -         | -         | -         | -         | -         | 

#### Multi-source scenario

| Dateset  | RefCOCO   | RefCOCO+  | RefCOCOg  | ReferIt   | Flickr    |
|----------|-----------|-----------|-----------|-----------|-----------|
| url      | [model]() | [model]() | [model]() | [model]() | [model]() |   
| size     | -         | -         | -         | -         | -         | 

### Fully supervised setting

| Dateset  | RefCOCO   | RefCOCO+  | RefCOCOg  | ReferIt   | Flickr    |
|----------|-----------|-----------|-----------|-----------|-----------|
| url      | [model]() | [model]() | [model]() | [model]() | [model]() |   
| size     | -         | -         | -         | -         | -         | 



## Results

<details open>
<summary><font size="4">
RefCOCO, RefCOCO+, and RefCOCOg datasets
</font></summary>
<img src="docs/refcoco.png" alt="COCO" width="100%">
</details>

<details open>
<summary><font size="4">
ReferIt and Flickr datasets
</font></summary>
<div align=center>
<img src="docs/referit.png" alt="COCO" width="50%"></div>
</details>

## Methods 
<p align="center"> <img src='docs/algorithm.jpg' align="center" width="100%"> </p>


## Visualization
<p align="center"> <img src='docs/sample1.jpg' align="center" width="100%"> </p>
<p align="center"> <img src='docs/sample2.jpg' align="center" width="100%"> </p>
<p align="center"> <img src='docs/sample3.jpg' align="center" width="100%"> </p>


## Acknowledgement

Our model is related to [CLIP](https://github.com/openai/CLIP), [Pseudo-Q](https://github.com/LeapLabTHU/Pseudo-Q), [TransVG](https://github.com/linhuixiao/TransVG). Thanks for their great work!

We also thank the great previous work including [DETR](https://github.com/facebookresearch/detr), [QRNet](https://github.com/LukeForeverYoung/QRNet), [M2](https://github.com/aimagelab/meshed-memory-transformer), [CLIPCap](https://github.com/rmokady/CLIP_prefix_caption), [RelTR](https://github.com/yrcong/RelTR), [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention), [ReSC](https://github.com/zyang-ur/ReSC), etc. 

Thanks [OpenAI](https://github.com/openai) for their awesome models.




## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@article{xiao2023clip,
  title={CLIP-VG: Self-paced Curriculum Adapting of CLIP for Visual Grounding},
  author={Xiao, Linhui and Yang, Xiaoshan and Peng, Fang and Yan, Ming and Wang, Yaowei and Xu, Changsheng},
  journal={arXiv preprint arXiv:2305.08685},
  year={2023}
}
```






