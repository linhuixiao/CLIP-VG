#!/usr/bin/env bash

# $ cd pseudo_sample_generation
# $ bash scripts/generate_pseudo_data_unc.sh
# 原始 --vg_dataset_path ../data/image_data
# 鹏城 229服务器：--vg_dataset_path /hdd/lhxiao/pseudo-q/data  # 主要是使用 data目录下detection_results中物体检测结果和属性检测结果
# 北京 226服务器：--vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data

# python ./utils/merge_file.py ../data/pseudo_samples/unc/top3_query6/ unc;
# python ./utils/post_process.py ../data/pseudo_samples/unc/top3_query6/unc/ unc;

#cd /data_SSD1/lhxiao/pseudo-q/reltr_output/unc/top10
#rm unc/ -rf
#mv unc_train_pseudo_split0.pth  unc_train_pseudo_split_0.pth
#mv unc_train_pseudo_split10.pth unc_train_pseudo_split_10.pth
#mv unc_train_pseudo_split11.pth unc_train_pseudo_split_11.pth
#mv unc_train_pseudo_split12.pth unc_train_pseudo_split_12.pth
#mv unc_train_pseudo_split13.pth unc_train_pseudo_split_13.pth
#mv unc_train_pseudo_split14.pth unc_train_pseudo_split_14.pth
#mv unc_train_pseudo_split15.pth unc_train_pseudo_split_15.pth
#mv unc_train_pseudo_split1.pth  unc_train_pseudo_split_1.pth
#mv unc_train_pseudo_split2.pth  unc_train_pseudo_split_2.pth
#mv unc_train_pseudo_split3.pth  unc_train_pseudo_split_3.pth
#mv unc_train_pseudo_split4.pth  unc_train_pseudo_split_4.pth
#mv unc_train_pseudo_split5.pth  unc_train_pseudo_split_5.pth
#mv unc_train_pseudo_split6.pth  unc_train_pseudo_split_6.pth
#mv unc_train_pseudo_split7.pth  unc_train_pseudo_split_7.pth
#mv unc_train_pseudo_split8.pth  unc_train_pseudo_split_8.pth
#mv unc_train_pseudo_split9.pth  unc_train_pseudo_split_9.pth
#
#cd /data_SSD1/lhxiao/pseudo-q/reltr_output/unc+/top10
#rm unc+/ -rf
#mv unc+_train_pseudo_split0.pth  unc+_train_pseudo_split_0.pth
#mv unc+_train_pseudo_split10.pth unc+_train_pseudo_split_10.pth
#mv unc+_train_pseudo_split11.pth unc+_train_pseudo_split_11.pth
#mv unc+_train_pseudo_split12.pth unc+_train_pseudo_split_12.pth
#mv unc+_train_pseudo_split13.pth unc+_train_pseudo_split_13.pth
#mv unc+_train_pseudo_split14.pth unc+_train_pseudo_split_14.pth
#mv unc+_train_pseudo_split15.pth unc+_train_pseudo_split_15.pth
#mv unc+_train_pseudo_split1.pth  unc+_train_pseudo_split_1.pth
#mv unc+_train_pseudo_split2.pth  unc+_train_pseudo_split_2.pth
#mv unc+_train_pseudo_split3.pth  unc+_train_pseudo_split_3.pth
#mv unc+_train_pseudo_split4.pth  unc+_train_pseudo_split_4.pth
#mv unc+_train_pseudo_split5.pth  unc+_train_pseudo_split_5.pth
#mv unc+_train_pseudo_split6.pth  unc+_train_pseudo_split_6.pth
#mv unc+_train_pseudo_split7.pth  unc+_train_pseudo_split_7.pth
#mv unc+_train_pseudo_split8.pth  unc+_train_pseudo_split_8.pth
#mv unc+_train_pseudo_split9.pth  unc+_train_pseudo_split_9.pth
#
#cd /data_SSD1/lhxiao/pseudo-q/reltr_output/gref/top10
#rm gref/ -rf
#mv gref_train_pseudo_split0.pth  gref_train_pseudo_split_0.pth
#mv gref_train_pseudo_split10.pth gref_train_pseudo_split_10.pth
#mv gref_train_pseudo_split11.pth gref_train_pseudo_split_11.pth
#mv gref_train_pseudo_split12.pth gref_train_pseudo_split_12.pth
#mv gref_train_pseudo_split13.pth gref_train_pseudo_split_13.pth
#mv gref_train_pseudo_split14.pth gref_train_pseudo_split_14.pth
#mv gref_train_pseudo_split15.pth gref_train_pseudo_split_15.pth
#mv gref_train_pseudo_split1.pth  gref_train_pseudo_split_1.pth
#mv gref_train_pseudo_split2.pth  gref_train_pseudo_split_2.pth
#mv gref_train_pseudo_split3.pth  gref_train_pseudo_split_3.pth
#mv gref_train_pseudo_split4.pth  gref_train_pseudo_split_4.pth
#mv gref_train_pseudo_split5.pth  gref_train_pseudo_split_5.pth
#mv gref_train_pseudo_split6.pth  gref_train_pseudo_split_6.pth
#mv gref_train_pseudo_split7.pth  gref_train_pseudo_split_7.pth
#mv gref_train_pseudo_split8.pth  gref_train_pseudo_split_8.pth
#mv gref_train_pseudo_split9.pth  gref_train_pseudo_split_9.pth
#
#cd /data_SSD1/lhxiao/pseudo-q/reltr_output/gref_umd/top10
#rm gref_umd/ -rf
#mv gref_umd_train_pseudo_split0.pth  gref_umd_train_pseudo_split_0.pth
#mv gref_umd_train_pseudo_split10.pth gref_umd_train_pseudo_split_10.pth
#mv gref_umd_train_pseudo_split11.pth gref_umd_train_pseudo_split_11.pth
#mv gref_umd_train_pseudo_split12.pth gref_umd_train_pseudo_split_12.pth
#mv gref_umd_train_pseudo_split13.pth gref_umd_train_pseudo_split_13.pth
#mv gref_umd_train_pseudo_split14.pth gref_umd_train_pseudo_split_14.pth
#mv gref_umd_train_pseudo_split15.pth gref_umd_train_pseudo_split_15.pth
#mv gref_umd_train_pseudo_split1.pth  gref_umd_train_pseudo_split_1.pth
#mv gref_umd_train_pseudo_split2.pth  gref_umd_train_pseudo_split_2.pth
#mv gref_umd_train_pseudo_split3.pth  gref_umd_train_pseudo_split_3.pth
#mv gref_umd_train_pseudo_split4.pth  gref_umd_train_pseudo_split_4.pth
#mv gref_umd_train_pseudo_split5.pth  gref_umd_train_pseudo_split_5.pth
#mv gref_umd_train_pseudo_split6.pth  gref_umd_train_pseudo_split_6.pth
#mv gref_umd_train_pseudo_split7.pth  gref_umd_train_pseudo_split_7.pth
#mv gref_umd_train_pseudo_split8.pth  gref_umd_train_pseudo_split_8.pth
#mv gref_umd_train_pseudo_split9.pth  gref_umd_train_pseudo_split_9.pth
#
#cd /data_SSD1/lhxiao/pseudo-q/reltr_output/referit/top10
#rm referit/ -rf
#mv referit_train_pseudo_split0.pth  referit_train_pseudo_split_0.pth
#mv referit_train_pseudo_split10.pth referit_train_pseudo_split_10.pth
#mv referit_train_pseudo_split11.pth referit_train_pseudo_split_11.pth
#mv referit_train_pseudo_split12.pth referit_train_pseudo_split_12.pth
#mv referit_train_pseudo_split13.pth referit_train_pseudo_split_13.pth
#mv referit_train_pseudo_split14.pth referit_train_pseudo_split_14.pth
#mv referit_train_pseudo_split15.pth referit_train_pseudo_split_15.pth
#mv referit_train_pseudo_split1.pth  referit_train_pseudo_split_1.pth
#mv referit_train_pseudo_split2.pth  referit_train_pseudo_split_2.pth
#mv referit_train_pseudo_split3.pth  referit_train_pseudo_split_3.pth
#mv referit_train_pseudo_split4.pth  referit_train_pseudo_split_4.pth
#mv referit_train_pseudo_split5.pth  referit_train_pseudo_split_5.pth
#mv referit_train_pseudo_split6.pth  referit_train_pseudo_split_6.pth
#mv referit_train_pseudo_split7.pth  referit_train_pseudo_split_7.pth
#mv referit_train_pseudo_split8.pth  referit_train_pseudo_split_8.pth
#mv referit_train_pseudo_split9.pth  referit_train_pseudo_split_9.pth
#
#cd /data_SSD1/lhxiao/pseudo-q/reltr_output/flickr/top10
#rm flickr/ -rf
#mv flickr_train_pseudo_split0.pth  flickr_train_pseudo_split_0.pth
#mv flickr_train_pseudo_split10.pth flickr_train_pseudo_split_10.pth
#mv flickr_train_pseudo_split11.pth flickr_train_pseudo_split_11.pth
#mv flickr_train_pseudo_split12.pth flickr_train_pseudo_split_12.pth
#mv flickr_train_pseudo_split13.pth flickr_train_pseudo_split_13.pth
#mv flickr_train_pseudo_split14.pth flickr_train_pseudo_split_14.pth
#mv flickr_train_pseudo_split15.pth flickr_train_pseudo_split_15.pth
#mv flickr_train_pseudo_split1.pth  flickr_train_pseudo_split_1.pth
#mv flickr_train_pseudo_split2.pth  flickr_train_pseudo_split_2.pth
#mv flickr_train_pseudo_split3.pth  flickr_train_pseudo_split_3.pth
#mv flickr_train_pseudo_split4.pth  flickr_train_pseudo_split_4.pth
#mv flickr_train_pseudo_split5.pth  flickr_train_pseudo_split_5.pth
#mv flickr_train_pseudo_split6.pth  flickr_train_pseudo_split_6.pth
#mv flickr_train_pseudo_split7.pth  flickr_train_pseudo_split_7.pth
#mv flickr_train_pseudo_split8.pth  flickr_train_pseudo_split_8.pth
#mv flickr_train_pseudo_split9.pth  flickr_train_pseudo_split_9.pth


#python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/unc/top10/ unc;
#python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/unc/top10/unc unc;
#
#python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/unc+/top10/ unc+;
#python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/unc+/top10/unc+ unc+;
#
#python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/gref/top10/ gref;
#python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/gref/top10/gref gref;
#
#python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/gref_umd/top10/ gref_umd;
#python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/gref_umd/top10/gref_umd gref_umd;
#
#python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/referit/top10/ referit;
#python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/referit/top10/referit referit;
#
#python ./pseudo_sample_generation/utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/reltr_output/flickr/top10/ flickr;
#python ./pseudo_sample_generation/utils/post_process.py /data_SSD1/lhxiao/pseudo-q/reltr_output/flickr/top10/flickr flickr;


cp /data_SSD1/lhxiao/pseudo-q/reltr_output/unc/top10/unc/unc_train_pseudo.pth /data_SSD1/lhxiao/pseudo-q/relation_expression_ori/unc/unc_train_pseudo.pth
cp /data_SSD1/lhxiao/pseudo-q/reltr_output/unc+/top10/unc+/unc+_train_pseudo.pth /data_SSD1/lhxiao/pseudo-q/relation_expression_ori/unc+/unc+_train_pseudo.pth
cp /data_SSD1/lhxiao/pseudo-q/reltr_output/gref/top10/gref/gref_train_pseudo.pth /data_SSD1/lhxiao/pseudo-q/relation_expression_ori/gref/gref_train_pseudo.pth
cp /data_SSD1/lhxiao/pseudo-q/reltr_output/gref_umd/top10/gref_umd/gref_umd_train_pseudo.pth /data_SSD1/lhxiao/pseudo-q/relation_expression_ori/gref_umd/gref_umd_train_pseudo.pth
cp /data_SSD1/lhxiao/pseudo-q/reltr_output/referit/top10/referit/referit_train_pseudo.pth /data_SSD1/lhxiao/pseudo-q/relation_expression_ori/referit/referit_train_pseudo.pth
cp /data_SSD1/lhxiao/pseudo-q/reltr_output/flickr/top10/flickr/flickr_train_pseudo.pth /data_SSD1/lhxiao/pseudo-q/relation_expression_ori/flickr/flickr_train_pseudo.pth
