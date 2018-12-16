CUDA_VISIBLE_DEVICES=6,7 python train.py --save_dir ../models/radar2radar/checkpoint --gen_frm_dir ../models/radar2radar/gen \
--input_length 1 --seq_length 2 --img_width 502 --img_channel 1 --patch_size 1 --batch_size 2 \
--dataset_name radar --train_data_paths ../data/radar/list-Z9080 --valid_data_paths ../data/radar/list-Z9080
