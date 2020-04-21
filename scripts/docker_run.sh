#!/usr/bin/env bash

#    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY \
#    -v /home/jiahuei/Documents/1_TF_files:/master/experiments \

#    -p 6006:6006 \

CODE_ROOT="/home/jiahuei/Dropbox/@_PhD/Codes/moses"

docker run -it \
    --gpus all \
    --shm-size 10G \
    -v ${CODE_ROOT}:/master \
    -v /home/jiahuei/Documents/3_Datasets/mol:/mol_data \
    -u "$(id -u)":"$(id -g)" \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY="$DISPLAY" \
    --rm molecularsets/moses

