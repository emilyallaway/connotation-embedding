#!/usr/bin/env bash

train_data=../../data/embedding/conn_input-train.csv
dev_data=../../data/embedding/conn_input-dev.csv
test_data=../../data/embedding/conn_input-test.csv


if [ $1 == 1 ]
then
    # EXTRACT embeddings
    python eval_model.py -s $2 -i ${train_data} -d ${dev_data} -k BEST -b balanced -m embeds


elif [ $1 == 2 ]
then
    # Evaluate
    python eval_model.py -s $2 -i ${train_data} -d ${dev_data} -b balanced -k BEST


elif [ $1 == 3 ]
then
    # Save predictions
    if [ $3 == "All" ]
    then
        python eval_model.py -s $2 -i ${train_data} -d ${dev_data} -k BEST -b balanced -m predict
    else
        python eval_model.py -s $2 -i ${train_data} -d ${dev_data} -k FINAL -b balanced -m predict -n "$3"
    fi

else
    echo "Doing nothing"
fi
