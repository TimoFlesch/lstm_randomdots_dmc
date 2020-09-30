#!/bin/bash
#
# runs lstm experiment with different bounds.

for (( i = 0; i < 2; i++ )); do
  #statements
  for (( j = 0; j < 4; j++ )); do
    #statements
    python run_batch_lstm.py --run_id $i --bound_idx $j
  done
done
