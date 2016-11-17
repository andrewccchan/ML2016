#!/bin/bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python hw3.py $1 $2 
