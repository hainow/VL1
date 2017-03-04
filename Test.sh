#!/bin/bash


if [ ! -n "$1" ]
    then
    echo "Usage: " $0 " script_to_run"
    exit
else
    PROCSTRING=$1
    echo "Running: " $PROCSTRING
fi

mkdir -p ./OUTPUT 
OUTPUT_PARENT=./OUTPUT
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
OUTPUT_FILER=$OUTPUT_PARENT.$current_time

echo $$ # current pid 
nice th "${PROCSTRING}" &> $OUTPUT_PARENT/$OUTPUT_FILER

