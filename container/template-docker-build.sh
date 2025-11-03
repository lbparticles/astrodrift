#!/bin/bash

sudo docker build -t test_defl -f $WORKDIR/container/dockerCUDA/ubuntu24-cuda12/Dockerfile $WORKDIR
