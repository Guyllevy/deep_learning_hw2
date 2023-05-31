#!/bin/bash

K_values=(32 64)
L_values=(2 4 8 16)

for K in "${K_values[@]}"; do
  for L in "${L_values[@]}"; do
    exp_name="exp1_1_L${L}_K${K}"
    command="python -m hw2.experiments run-exp -n ${exp_name} -K ${K} -L ${L} -P 4 -H 100"
    echo "Running command: ${command}"
    eval "${command}"
  done
done