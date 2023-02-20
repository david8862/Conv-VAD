#!/bin/bash
#
# random pick sample files from src path to dst path
#

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 <src_path> <dst_path> <sample_count>"
    exit 1
fi

SRC_PATH=$1
DST_PATH=$2
SAMPLE_COUNT=$3
#SAMPLE_LIST=$(ls $SRC_PATH | sort -R | head -n$SAMPLE_COUNT)
SAMPLE_LIST=$(find $SRC_PATH -name "*" | sort -R | head -n$SAMPLE_COUNT)

# prepare process bar
i=0
ICON_ARRAY=("\\" "|" "/" "-")

# create dest path first
mkdir -p $DST_PATH


for SAMPLE_FILE in $SAMPLE_LIST
do
    cp -drf $SAMPLE_FILE $DST_PATH/
    # update process bar
    let index=i%4
    let percent=i*100/SAMPLE_COUNT
    let num=percent/2
    bar=$(seq -s "#" $num | tr -d "[:digit:]")
    #printf "inference process: %d/%d [%c]\r" "$i" "$SAMPLE_COUNT" "${ICON_ARRAY[$index]}"
    printf "inference process: %d/%d [%-50s] %d%% \r" "$i" "$SAMPLE_COUNT" "$bar" "$percent"
    let i=i+1
done
printf "\nDone\n"
