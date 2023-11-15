#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

input_file="$1"
output_file="$2"

prefix="split_"

# 分割数据集为四部分
split -n l/4 "$input_file" "$prefix"

# 以下不用for循环是因为想使用非阻塞的方法执行，提高并行效率
scp ${prefix}ab thumm03:~/ &
scp ${prefix}ac thumm04:~/ &
scp ${prefix}ad thumm05:~/
wait

# tr -s ' ' '\n' 将有空格的词组分为多个单词
# tr 'A-Z' 'a-z' 将所有大些字母都换成小写
# uniq -c 分别统计不同单词的出现频率
ssh thumm03 "cat ~/split_ab | tr -s ' ' '\n' | tr 'A-Z' 'a-z' | sort | uniq -c > ~/result03.txt" &
ssh thumm04 "cat ~/split_ac | tr -s ' ' '\n' | tr 'A-Z' 'a-z' | sort | uniq -c > ~/result04.txt" &
ssh thumm05 "cat ~/split_ad | tr -s ' ' '\n' | tr 'A-Z' 'a-z' | sort | uniq -c > ~/result05.txt" &
# 在thumm01本地进行统计
cat ${prefix}aa | tr -s ' ' '\n' | tr 'A-Z' 'a-z' | sort | uniq -c > result01.txt
wait

# 获取远程统计结果
scp thumm03:~/result03.txt . &
scp thumm04:~/result04.txt . &
scp thumm05:~/result05.txt .
wait

# 用awk合并所有四个机器的结果
cat result01.txt result03.txt result04.txt result05.txt | sort | awk '{ count[$2] += $1; } END { for(word in count) print count[word], word }' | sort -nr > "$output_file"

# 清理所有临时文件
rm "$prefix"aa "$prefix"ab "$prefix"ac "$prefix"ad result01.txt result03.txt result04.txt result05.txt

echo "Final word count saved to $output_file"
