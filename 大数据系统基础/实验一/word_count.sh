#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <wc_dataset> <output_word_count>"
    exit 1
fi

wc_dataset=${1%.*}
word_count="$2"

# 创建2GB的数据集
rm -f ${wc_dataset}_2GB.txt
echo "Begin creating a 2GB wc_dataset..."
for i in {1..160}
do
    cat wc_dataset.txt >> ${wc_dataset}_2GB.txt
done

echo "Dataset Statistic:"
ls -lh ${wc_dataset}_2GB.txt| awk '{print $5, $9}'

echo "Doing multi-node word count..."
rm -f $word_count
time bash ./wc_multi.sh ${wc_dataset}_2GB.txt $word_count

echo "Doing single-node word count..."
rm -f ${word_count%.*}_single.txt
time bash ./wc_single.sh ${wc_dataset}_2GB.txt ${word_count%.*}_single.txt


if diff -q $word_count ${word_count%.*}_single.txt > /dev/null
then
    echo "The multi-node output and single-node output are the same."
else
    echo "The files are different."
fi
