#!/bin/bash
FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer', 'cpp_infer', 'serving_infer']
MODE=$2
if [ ${MODE} = "cpp_infer" ]; then
    dataline=$(awk 'NR==52, NR==66{print}'  $FILENAME)
elif [ ${MODE} = "serving_infer" ]; then
    dataline=$(awk 'NR==67, NR==81{print}'  $FILENAME)
else
    dataline=$(awk 'NR==1, NR==51{print}'  $FILENAME)
fi
count=0
for line in ${dataline[*]}; do
   let count++
   echo $count $line
done
