#!/bin/bash

i=1
while  [ $i -le 100 ]
do 
  sleep 10
  free -m >> ./free.txt
done