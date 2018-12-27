CUDA_VISIBLE_DEVICES=6,7 python train.py \
--save_dir ../models/radar2rain-peakyloss/checkpoint \
--gen_frm_dir ../models/radar2rain-peakyloss/gen \
--input_length 10 \
--seq_length 11 \
--img_width 64 \
--img_channel 1 \
--patch_size 4 \
--batch_size 32 \
--lr 0.001 \
--test_interval 1000 \
--snapshot_interval 500 \
--dataset_name rain \
--train_data_paths ../data/rain/list-Z9080 \
--valid_data_paths ../data/radar/list-Z9080 \
--reverse_input False \
--pretrained_model ../models/radar2radar-finetune/checkpoint/model.ckpt-10000
