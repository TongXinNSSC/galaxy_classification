#!/usr/bin/env sh

python -u MLP.py \
--batch-size 64 \
--test-batch-size 1000 \
--epochs 15 \
--lr 0.01 \
--momentum 0.9 \
--no-cuda \
--save-path 'checkpoint/' \
--load-path 'checkpoint/_15.pth.tar' \
 > log.txt
