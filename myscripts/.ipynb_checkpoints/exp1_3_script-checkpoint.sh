#!/bin/bash

L_values=(2 3 4)
K_values=("64" "128")

for L in "${L_values[@]}"; do
    exp_name="exp1_3"
    command="python -m hw2.experiments run-exp -n ${exp_name} -K ${K_values[0]} ${K_values[1]} -L ${L} -P 4 -H 100"
    echo "Running command: ${command}"
    eval "${command}"
done
