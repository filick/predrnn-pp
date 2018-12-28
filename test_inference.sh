#!/bin/bash
CUDA_VISIBLE_DEVICES=, python inference.py \
--pretrained_model ../models/radar2rain-peakyloss/checkpoint/model.ckpt-67000
