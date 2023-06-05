#!/bin/bash

L_values=(2 4 8)
K_values=("32" "64" "128")

for L in "${L_values[@]}"; do
  for K in "${K_values[@]}"; do
    exp_name="exp1_2"
    command="python -m hw2.experiments run-exp -n ${exp_name} -K ${K} -L ${L} -P 4 -H 100"
    echo "Running command: ${command}"
    eval "${command}"
  done
done