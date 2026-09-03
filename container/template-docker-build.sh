#!/bin/bash

docker build -t test_defl -f "$WORKDIR/env/Dockerfile.modern" "$WORKDIR"
