#!/bin/bash

# Check if the train.py file exists
if [[ ! -f "train_rl_revised.py" ]]; then
  echo "Error: train_rl_revised.py not found!"
  exit 1
fi
export PYTHONPATH=/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/venv/bin/python
# Loop to execute train.py 10 times
for i in {1..10}
do
  echo "Execution $i of train_rl_revised.py"
  /Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/venv/bin/python train_rl_revised.py
done