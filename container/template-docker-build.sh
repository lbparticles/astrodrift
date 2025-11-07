#!/bin/bash

docker build -t test_defl -f $WORKDIR/container/ubuntu24-cuda12/Dockerfile $WORKDIR
