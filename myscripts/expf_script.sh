#!/bin/bash

command="python -m hw2.experiments run-exp -n exp1_1 -K 32 -L 16 -P 8 -H 100"
echo "Running command: ${command}"
eval "${command}"

command="python -m hw2.experiments run-exp -n exp1_1 -K 64 -L 16 -P 8 -H 100"
echo "Running command: ${command}"
eval "${command}"

command="python -m hw2.experiments run-exp -n exp1_2 -K 128 -L 8 -P 4 -H 100"
echo "Running command: ${command}"
eval "${command}"



