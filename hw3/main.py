import numpy as np
import pickle
from util import parsePara, getTrainValidSet
from loadData import load_al, load_au
from model import buildCNN
from train import train_basic
from eva import predict
# Load data
para = parsePara();
[al_X, al_y] = load_al(para)
#au_X = load_au(para)

# Build CNN
model = buildCNN(al_X.shape[1:], para)

# Training + Validation
model = train_basic(model, al_X, al_y, "SGD", para)

# Evaluate training on validation sets
# model.evaluate()

# predict
predict(model, para)
