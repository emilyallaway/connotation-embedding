#!/usr/bin/env bash

train_data=../../data/embedding/conn_input-train.csv
dev_data=../../data/embedding/conn_input-dev.csv
test_data=../../data/embedding/conn_input-test.csv


if [ $1 == 1 ]
then
    echo "Training $2 all dims together WITH related words"
    python train.py -s $2 -i ${train_data} -d ${dev_data} -b balanced


elif [ $1 == 2 ]
then
    echo "Training $2 all dims together WITHOUT related words"
    python train.py -s $2 -i ${train_data} -d ${dev_data} -b balanced -r 0


elif [ $1 == 3 ]
then
    echo "Training $2 all dimensions individually WITH related words"
    for di in "Social Val" "Polite" "Impact" "Fact" "Sent" "Emo" "P(wt)" "P(wa)" "P(at)" "E(t)" "E(a)" "V(t)" "V(a)" "S(t)" "S(a)" "power" "agency"
    do
        echo "DIM $di"
        python train.py -s $2 -i ${train_data} -d ${dev_data} -b balanced -n "$di"
        echo "------------------------------------------"
        echo
    done


else
    echo "Doing nothing"
fi
