#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

input_file="$1"
output_file="$2"

cat $input_file | tr -s ' ' '\n' | tr 'A-Z' 'a-z' | sort | uniq -c | sort -nr | awk '{$1=$1;print}' > $output_file

echo "Final word count saved to $output_file"
