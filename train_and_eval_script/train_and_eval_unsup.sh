echo "train_and_eval_unsup.sh"
# train
echo "train unc"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28887 --use_env train_clip_vg.py --num_workers 2 --epochs 110 --batch_size 64 --lr 0.00025  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate      --imsize 224 --max_query_len 77 --dataset unc      --data_root /path_to_image_data --split_root /path_to_split  --output_dir /path_to_output/unc;
echo "train unc+"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28887 --use_env train_clip_vg.py --num_workers 2 --epochs 110 --batch_size 64 --lr 0.00025  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate      --imsize 224 --max_query_len 77 --dataset unc+     --data_root /path_to_image_data --split_root /path_to_split  --output_dir /path_to_output/unc+;
echo "train gref"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28887 --use_env train_clip_vg.py --num_workers 2 --epochs 110 --batch_size 64 --lr 0.00025  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate      --imsize 224 --max_query_len 77 --dataset gref     --data_root /path_to_image_data --split_root /path_to_split  --output_dir /path_to_output/gref;
echo "train gref_umd"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28887 --use_env train_clip_vg.py --num_workers 2 --epochs 110 --batch_size 64 --lr 0.00025  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate      --imsize 224 --max_query_len 77 --dataset gref_umd --data_root /path_to_image_data --split_root /path_to_split  --output_dir /path_to_output/gref_umd;
echo "train referit"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28887 --use_env train_clip_vg.py --num_workers 2 --epochs 110 --batch_size 64 --lr 0.00025  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate      --imsize 224 --max_query_len 77 --dataset referit  --data_root /path_to_image_data --split_root /path_to_split  --output_dir /path_to_output/referit;
echo "train flickr"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28887 --use_env train_clip_vg.py --num_workers 2 --epochs 110 --batch_size 64 --lr 0.00025  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate      --imsize 224 --max_query_len 77 --dataset flickr   --data_root /path_to_image_data --split_root /path_to_split  --output_dir /path_to_output/flickr;

# eval :
# RefCOCO unc
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset unc      --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/unc/best_checkpoint.pth      --eval_set val    --output_dir /path_to_output/unc;
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset unc      --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/unc/best_checkpoint.pth      --eval_set testA  --output_dir /path_to_output/unc;
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset unc      --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/unc/best_checkpoint.pth      --eval_set testB  --output_dir /path_to_output/unc;
# RefCOCO+ unc+
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset unc+     --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/unc+/best_checkpoint.pth     --eval_set val    --output_dir /path_to_output/unc+;
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset unc+     --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/unc+/best_checkpoint.pth     --eval_set testA  --output_dir /path_to_output/unc+;
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset unc+     --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/unc+/best_checkpoint.pth     --eval_set testB  --output_dir /path_to_output/unc+;
# gref
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset gref     --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/gref/best_checkpoint.pth     --eval_set val    --output_dir /path_to_output/gref;
# gref_umd
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset gref_umd --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/gref_umd/best_checkpoint.pth --eval_set val    --output_dir /path_to_output/gref_umd;
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset gref_umd --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/gref_umd/best_checkpoint.pth --eval_set test   --output_dir /path_to_output/gref_umd;
# ReferItGame
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset referit  --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/referit/best_checkpoint.pth --eval_set val    --output_dir /path_to_output/referit;
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset referit  --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/referit/best_checkpoint.pth --eval_set test   --output_dir /path_to_output/referit;
# flickr
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset flickr   --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/flickr/best_checkpoint.pth  --eval_set val    --output_dir /path_to_output/flickr;
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset flickr   --imsize 224 --max_query_len 77 --data_root /path_to_image_data --split_root /path_to_split --eval_model /path_to_output/flickr/best_checkpoint.pth  --eval_set test   --output_dir /path_to_output/flickr;


