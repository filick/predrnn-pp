#!/bin/bash
CUDA_VISIBLE_DEVICES=, python inference.py \
--pretrained_model ../models/radar2radar-finetune/checkpoint/model.ckpt-10000
