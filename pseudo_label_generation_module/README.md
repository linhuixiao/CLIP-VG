# Pseudo-language Label Generation Module

The following are the **pseudo-language label generation module** that generates **pseudo-language labels** in an unsupervised setting.

The **single-source scenario** includes a pseudo-template label derived from [Pseudo-Q](https://github.com/LeapLabTHU/Pseudo-Q). 

**Multi-source scenario** include pseudo-template labels, pseudo-relation labels, and pseudo-caption labels, 
which derived from [Pseudo-Q](https://github.com/LeapLabTHU/Pseudo-Q), [RelTR](https://github.com/yrcong/RelTR), [CLIPCap](https://github.com/rmokady/CLIP_prefix_caption) / [M2](https://github.com/aimagelab/meshed-memory-transformer), respectively. 


## pseudo-template label generation

the generation of pseudo-template label are detailed provided in [pseudo_template_label_generation](pseudo_template_label_generation/README.md).


## pseudo-relation labels generation

First, you should complete the environment preparation of the RelTR model as instructed by [RelTR README](pseudo_relation_label_generation/RelTR/README.md). 
Simultaneously, ensure that the visual grounding image data preparation is completed and download the split subset of Pseudo-Q according to the dataset split in [Pseudo-Q README](pseudo_template_label_generation/README.md). 
As the above is complete, replace the dataset and output directory in [inference_gen_pseudo_relation_label.py](pseudo_relation_label_generation/RelTR/inference_gen_pseudo_relation_label.py). 
Finally, run inference_gen_pseudo_relation_label.py by using following instruction:
    
    python inference_gen_pseudo_relation_label.py


## pseudo-caption labels generation

the generation of pseudo-caption labels are detailed provided in [pseudo_relation_label_generation](pseudo_caption_label_generation/README.md).

1. First, you should complete the environment preparation of the CLIPcap model as instructed by [CLIP_prefix_caption README](pseudo_caption_label_generation/CLIP_prefix_caption/README.md). 
Simultaneously, ensure that the visual grounding image data preparation is completed and download the split subset of Pseudo-Q according to the dataset split in [Pseudo-Q README](pseudo_template_label_generation/README.md). 
As the above is complete, replace the dataset and output directory in [clip_prefix_captioning_for_dataset.py](pseudo_caption_label_generation/clip_prefix_captioning_for_dataset.py). 
Finally, run clip_prefix_captioning_for_dataset.py by using following instruction:
    
    python clip_prefix_captioning_for_dataset.py

2. Since the generated caption does not include bounding box information, we need to use a language parser such as 
Spacy to parse the generated caption and extract the subject. Then, we pair the subject with the object detection label 
which used in the pseudo-template label. If successful, we match the caption with the corresponding bounding box 
of the object detection label. 

3. Therefore, First, replace file path with your dataset and output directory in [caption_generation.py](pseudo_caption_label_generation/pseudo_caption_and_box_matching/caption_generation.py), 
and verify the correctness of the code according to one of the instructions in [generate_caption_data_all.sh](pseudo_caption_and_box_matching/generate_caption_data_all.sh).
After passing validation, use the following script instruction to pair captions and bounding box for all datasets:
   ```angular2html
   bash generate_caption_data_all.sh
   ```
   The implementation pipeline for [M2](https://github.com/aimagelab/meshed-memory-transformer) is the same as above.
