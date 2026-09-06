#!/bin/bash
apptainer run -B /raid/afo59/sft/astrodrift:/data/drift --writable-tmpfs --no-mount home --home /data/home/leo --nv --cleanenv --cwd /data/drift drift.sif
