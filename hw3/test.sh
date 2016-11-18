#!/bin/bash
KERAS_BACKEND=theano
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python eva.py $1 $2 $3
