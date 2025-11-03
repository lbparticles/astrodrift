#!/bin/bash
apptainer run -B /raid/afo59/sft/astrodrift:/data/astrodrift --writable-tmpfs --no-mount home --nv --cleanenv --cwd /data/astrodrift astrodrift.sif
