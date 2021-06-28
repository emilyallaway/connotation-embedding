#!/usr/bin/env bash

train_data=../../data/stance/IACv2_stance-train.csv
dev_data=../../data/stance/IACv2_stance-dev.csv
test_data=../../data/stance/IACv2_stance-test.csv


if [ $1 == 1 ]
then
    python eval_model.py -s $2 -i ${train_data} -d ${dev_data} -k BEST -m predict -o $3

elif [ $1 == 2 ]
then
    python eval_model.py -s $2 -i ${train_data} -d ${dev_data} -k BEST -m eval


else
    echo "Doing nothing"
fi