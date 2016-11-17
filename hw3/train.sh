#!/bin/bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python hw3_m2.py $1 $2 
