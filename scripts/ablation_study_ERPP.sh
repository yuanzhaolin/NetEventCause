#!/bin/bash

cd ..

model=ERPP
dataset=toy
kind=ggem-1K-5


#echo 'Stage 1: Generate the events sequences for training and test.'
#
#python example/6_generate_events_by_ggem.py

#echo 'Stage 2: Training events models.'
#
#python example/7_test_event_cause_discovery.py $model --epoch 700 --dataset toy --kind ggem-1K-5 --verbose --cuda --batch_size 32

echo 'Stage 3: Root Alarms Analysing.'

python example/8_event_rca.py --model ERPP --dataset toy --kind ggem-1K-5 --add_label_cols True --save_all True --steps 1 --suffix 1
python example/8_event_rca.py --model ERPP --dataset toy --kind ggem-1K-5 --add_label_cols True --save_all True --steps 2 --suffix 2
#python example/8_event_rca.py --model ERPP --dataset toy --kind ggem-1K-5 --add_label_cols True --save_all True --steps 3 --suffix 3
#python example/8_event_rca.py --model ERPP --dataset toy --kind ggem-1K-5 --add_label_cols True --save_all True --steps 5 --suffix 5
#python example/8_event_rca.py --model ERPP --dataset toy --kind ggem-1K-5 --add_label_cols True --save_all True --steps 10 --suffix 10
#python example/8_event_rca.py --model ERPP --dataset toy --kind ggem-1K-5 --add_label_cols True --save_all True --steps 20 --suffix 20
#python example/8_event_rca.py --model ERPP --dataset toy --kind ggem-1K-5 --add_label_cols True --save_all True --steps 40 --suffix 40

echo 'Stage 4: Evaluating Root alarms identification and causality discovery.'

python example/9_rca_accuracy_eval.py
