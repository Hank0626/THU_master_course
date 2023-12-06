#!/bin/bash

# 测量并记录 generate_matrix.py 的执行时间
echo "Running generate_matrix.py..."
time python3 generate_matrix.py 150 100 50 100 ./dfs/data/input_large.txt output_large.txt

# 测量并记录 client.py 的执行时间
echo "Running client.py..."
time python3 client.py -matrixCal input_large.txt output_large_real.txt

# 比较文件并检查是否相同
if diff output_large.txt output_large_real.txt > /dev/null; then
    echo "Files are identical."
else
    echo "Files differ."
fi