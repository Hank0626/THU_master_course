#!/bin/bash

# 确保input_large.txt output_large.txt已经在所有的node上

# 测量并记录 client.py 的执行时间
echo "Running client.py..."
time python3 client.py -matrixCal input_large.txt output_large_real.txt

# 比较文件并检查是否相同
if diff output_large.txt output_large_real.txt > /dev/null; then
    echo "Files are identical."
else
    echo "Files differ."
fi