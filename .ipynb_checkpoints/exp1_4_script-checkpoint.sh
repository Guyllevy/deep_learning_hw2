#!/bin/bash

L_values=(8 16 32)

for L in "${L_values[@]}"; do
    exp_name="exp1_4_L${L}_K32"
    command="python -m hw2.experiments run-exp -n ${exp_name} -K 32 -L ${L} -P 4 -H 100 -M resnet"
    echo "Running command: ${command}"
    eval "${command}"
done

L_values_fixed=(2 4 8)

for L in "${L_values_fixed[@]}"; do
    exp_name="exp1_4_L${L}_K64-128-256"
    command="python -m hw2.experiments run-exp -n ${exp_name} -K 64 128 256 -L ${L} -P 4 -H 100 -M resnet"
    echo "Running command: ${command}"
    eval "${command}"
done