#!/usr/bin/env bash

OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind  0;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind  1;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind  2;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind  3;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind  4;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind  5;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind  6;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind  7;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind  8;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind  9;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind 10;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind 11;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind 12;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind 13;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind 14;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc --split_ind 15;

python ./utils/merge_file.py /hdd/lhxiao/pseudo-q/caption_gen/unc unc;
python ./utils/post_process.py /hdd/lhxiao/pseudo-q/caption_gen/unc/unc unc;

OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind  0;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind  1;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind  2;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind  3;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind  4;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind  5;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind  6;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind  7;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind  8;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind  9;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind 10;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind 11;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind 12;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind 13;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind 14;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset unc+ --split_ind 15;

python ./utils/merge_file.py /hdd/lhxiao/pseudo-q/caption_gen/unc+ unc+;
python ./utils/post_process.py /hdd/lhxiao/pseudo-q/caption_gen/unc+/unc+ unc+;

OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind  0;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind  1;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind  2;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind  3;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind  4;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind  5;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind  6;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind  7;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind  8;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind  9;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind 10;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind 11;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind 12;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind 13;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind 14;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref --split_ind 15;

python ./utils/merge_file.py /hdd/lhxiao/pseudo-q/caption_gen/gref gref;
python ./utils/post_process.py /hdd/lhxiao/pseudo-q/caption_gen/gref/gref gref;

OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind  0;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind  1;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind  2;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind  3;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind  4;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind  5;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind  6;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind  7;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind  8;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind  9;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind 10;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind 11;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind 12;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind 13;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind 14;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset gref_umd --split_ind 15;

python ./utils/merge_file.py /hdd/lhxiao/pseudo-q/caption_gen/gref_umd gref_umd;
python ./utils/post_process.py /hdd/lhxiao/pseudo-q/caption_gen/gref_umd/gref_umd gref_umd;

OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind  0;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind  1;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind  2;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind  3;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind  4;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind  5;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind  6;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind  7;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind  8;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind  9;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind 10;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind 11;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind 12;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind 13;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind 14;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset referit --split_ind 15;

python ./utils/merge_file.py /hdd/lhxiao/pseudo-q/caption_gen/referit referit;
python ./utils/post_process.py /hdd/lhxiao/pseudo-q/caption_gen/referit/referit referit;

OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind  0;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind  1;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind  2;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind  3;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind  4;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind  5;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind  6;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind  7;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind  8;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind  9;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind 10;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind 11;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind 12;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind 13;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind 14;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /hdd/lhxiao/pseudo-q/image_data --vg_dataset flickr --split_ind 15;

python ./utils/merge_file.py /hdd/lhxiao/pseudo-q/caption_gen/flickr flickr;
python ./utils/post_process.py /hdd/lhxiao/pseudo-q/caption_gen/flickr/flickr flickr;
