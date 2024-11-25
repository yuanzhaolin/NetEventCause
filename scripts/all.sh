#!/bin/bash

cd ..

model=ODE-RNN
dataset=toy
kind=ggem-1K-5


echo 'Stage 1: Generate the events sequences for training and test.'

python example/6_generate_events_by_ggem.py

echo 'Stage 2: Training events models.'

python example/7_test_event_cause_discovery.py $model --epoch 700 --dataset $dataset --kind $kind --verbose --batch_size 1024 --method dopri5 --cuda

echo 'Stage 3: Root Alarms Analysing.'

python example/8_event_rca.py --dataset $dataset --kind $kind --add_label_cols True --save_all True --manual_rule --model $model

echo 'Stage 4: Evaluating Root alarms identification and causality discovery.'

python example/9_rca_accuracy_eval.py --algorithm nec
