#!/bin/bash

# Check if the train.py file exists
if [[ ! -f "train.py" ]]; then
  echo "Error: train.py not found!"
  exit 1
fi
export PYTHONPATH=/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api
# Loop to execute train.py 10 times
for i in {1..50}
do
  echo "Execution $i of train.py"
  python3 train.py
done