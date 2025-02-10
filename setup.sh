#!/bin/bash

DATASET="equity-post-HCT-survival-predictions"
DATA="data"
COMPRESSED_DATA="${DATA}/compressed"
RAW_DATA="${DATA}/raw"
PROCESSED_DATA="${DATA}/processed"
RAW_FILES="${RAW_DATA}/*"
DATETIME=$(date '+%Y-%m-%d_%H%M%S')

mkdir -p $COMPRESSED_DATA $RAW_DATA $PROCESSED_DATA
kaggle competitions download -c $DATASET
unzip -o "${DATASET}.zip" -d $RAW_DATA
mv "${DATASET}.zip" "${COMPRESSED_DATA}/${DATASET}-${DATETIME}.zip"
for f in $RAW_FILES
do
    echo $DATETIME $(md5sum $f) >> "${DATA}/md5sums.txt"
done