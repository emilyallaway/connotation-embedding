#!/usr/bin/env bash

train_data=../../data/stance/IACv2_stance-train.csv
dev_data=../../data/stance/IACv2_stance-dev.csv
test_data=../../data/stance/IACv2_stance-test.csv

python train_model.py -s $1 -i ${train_data} -d ${dev_data}
