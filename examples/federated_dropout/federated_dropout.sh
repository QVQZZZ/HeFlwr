#!/bin/bash

# Default parameters
num_rounds=5        # Number of rounds
client_num=8        # Number of clients
dataset="mnist"     # Dataset
partition="iid"     # Partition type
alpha=0.0           # Alpha
batch_size=32       # Batch size

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num_rounds) num_rounds="$2"; shift ;;
        --client_num) client_num="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --partition) partition="$2"; shift ;;
        --alpha) alpha="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Generate output file name based on parameters
output_file="federateddropout_rounds=${num_rounds}_dataset=${dataset}_clients=${client_num}_partition=${partition}_alpha=${alpha}_batch=${batch_size}.txt"

# Start the server process and redirect output to the generated file
python server.py --dataset "$dataset" --num_rounds "$num_rounds" > "$output_file" &

# Server address
server_address="127.0.0.1:8080"

# Define --p parameter values
p_values=("1/4" "1/2" "3/4" "1")

# Start multiple client processes in the background
for (( cid=1; cid<=$client_num; cid++ ))
do
    # Calculate the current client's --p parameter value
    p_value=${p_values[($cid - 1) % ${#p_values[@]}]}
    
    arguments="client.py --server_address $server_address --client_num $client_num --cid $cid --dataset $dataset --partition $partition --alpha $alpha --batch_size $batch_size --p $p_value"
    python $arguments &
done

wait  # Wait for all background processes to finish
