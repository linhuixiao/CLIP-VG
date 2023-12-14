#!/usr/bin/env bash

# $ cd pseudo_sample_generation
# $ bash scripts/generate_pseudo_data_unc.sh
# 原始 --vg_dataset_path ../data/image_data
# 鹏城 229服务器：--vg_dataset_path /hdd/lhxiao/pseudo-q/data  # 主要是使用 data目录下detection_results中物体检测结果和属性检测结果
# 北京 226服务器：--vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data

# python ./utils/merge_file.py ../data/pseudo_samples/unc/top3_query6/ unc;
# python ./utils/post_process.py ../data/pseudo_samples/unc/top3_query6/unc/ unc;

OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 0;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 1;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 2;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 3;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 4;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 5;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 6;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 7;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 8;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 9;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 10;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 11;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 12;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 13;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 14;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc --split_ind 15;
python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/unc/top10/ unc;
python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/unc/top10/unc unc;

OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 0;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 1;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 2;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 3;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 4;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 5;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 6;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 7;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 8;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 9;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 10;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 11;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 12;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 13;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 14;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset unc+ --split_ind 15;
python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/unc+/top10/ unc+;
python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/unc+/top10/unc+ unc+;


OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 0;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 1;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 2;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 3;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 4;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 5;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 6;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 7;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 8;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 9;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 10;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 11;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 12;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 13;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 14;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref --split_ind 15;
python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/gref/top10/ gref;
python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/gref/top10/gref gref;


OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 0;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 1;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 2;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 3;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 4;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 5;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 6;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 7;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 8;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 9;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 10;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 11;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 12;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 13;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 14;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset gref_umd --split_ind 15;
python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/gref_umd/top10/ gref_umd;
python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/gref_umd/top10/gref_umd gref_umd;



OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 0;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 1;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 2;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 3;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 4;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 5;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 6;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 7;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 8;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 9;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 10;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 11;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 12;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 13;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 14;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset referit --split_ind 15;
python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/referit/top10/ referit;
python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/referit/top10/referit referit;


OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 0;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 1;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 2;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 3;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 4;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 5;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 6;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 7;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 8;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 9;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 10;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 11;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 12;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 13;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 14;
OMP_NUM_THREADS=8 python inference_gen_ref_pseudo_cap.py --vg_dataset flickr --split_ind 15;
python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/flickr/top10/ flickr;
python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/flickr/top10/flickr flickr;


