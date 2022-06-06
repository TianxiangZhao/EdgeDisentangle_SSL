#!/bin/bash

pretrains=(SupEdge DisEdge DifHead)
edges=(1)
pre_edges=(1)
pre_weights=(100 1 0)
model=DISGAT
datas=(cora_full)
seeds=(4)
att=3
gnn_types=(AT)
steps=(5)
con_layers=(0)
epoch=1010

for seed in ${seeds[*]}
  do
    edge=1
    for gnn_type in ${gnn_types[*]}
      do
        for con_layer in ${con_layers[*]}
          do
            for ind1 in ${!pre_weights[*]}
              do
                for ind2 in ${!pre_weights[*]}
                  do
                    for ind3 in ${!pre_weights[*]}
                      do
                        pre_weight=(${pre_weights[$ind1]} ${pre_weights[$ind2]} ${pre_weights[$ind3]})
                        pre_edge=(1 1 1)
                          for step in ${steps[*]}
                            do
                              for data in ${datas[*]}
                                do
                                      
                                  LOGADDRESS="./logs/log_"$data"_preweight"${pre_weights[$ind1]}${pre_weights[$ind2]}${pre_weights[$ind3]}$model$gnn_type
                                  echo $LOGADDRESS
                                  CUDA_VISIBLE_DEVICES=2 python main.py --seed=$seed --model=$model --used_edge=$edge --finetune --downstream=CLS --down_weight=1.0 --step=$step --nhead=4 --dataset=$data --pretrain ${pretrains[*]} --pre_weight ${pre_weight[*]} --pre_edge ${pre_edge[*]} --sparse  --att=$att --constrain_layer=$con_layer --epochs=$epoch --gnn_type=$gnn_type >>$LOGADDRESS
                                      
                                done  
                            done    
                      done
                  done
              done
          done
      done
  done
