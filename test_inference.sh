#!/bin/bash
CUDA_VISIBLE_DEVICES=, python inference.py \
--pretrained_model dat/model.ckpt-10000
