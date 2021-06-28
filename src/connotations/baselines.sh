#!/usr/bin/env bash

dim_str="Social_Val.Polite.Fact.Sent.Impact.Emo"

for n in "Maj" "LogReg"
do
    v="numberbatch-en-19.08.txt"
    t="O"
    echo " ... baseline=$n,   pos = $t,   vecs = $v"
    save_str="../../saved_models/$t.$n-$dim_str/"
    echo " ... saving to = ${save_str}"
    python train_test_baselines.py -m 1 -p $t -d All -v $v -n $n -e ../../saved_models/
    echo "------------------------------------"
done