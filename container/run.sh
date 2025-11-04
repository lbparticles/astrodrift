#!/bin/bash
apptainer run -B /raid/afo59/sft/astrodrift:/data/astrodrift --writable-tmpfs --no-mount home --home /data/home/leo --nv --cleanenv --cwd /data/astrodrift astrodrift.sif
