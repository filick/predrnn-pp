CUDA_VISIBLE_DEVICES=6,7 python train.py --save_dir ../models/radar2radar/checkpoint --gen_frm_dir ../models/radar2radar/gen \
--input_length 10 --seq_length 20 --img_width 64 --img_channel 1 --patch_size 4 --batch_size 16 --snapshot_interval 2000 \
--dataset_name radar --train_data_paths ../data/radar/list-Z9080 --valid_data_paths ../data/radar/list-Z9080
