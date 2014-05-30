#!/bin/bash

for f in $@; do

csv=$(cat $f | sed "s/ /,/g" )
echo "$csv" > "$f.csv"

done
