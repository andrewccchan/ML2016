#!/bin/bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python eva.py $1 $2 $3
