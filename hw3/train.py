import numpy as np
from keras.optimizers import SGD
from util import getTrainValidSet
from eva import eva

def train_basic(model, al_X, al_y, mode, para) :
    print "Training CNN with " + str(mode)
    batch_size = para["batch_size"]
    nb_epoch = para["nb_epoch"]
    if (mode == "SGD") :
        rate = para["sgd_lr"]
        dec = para["sgd_decay"]
        mom = para["sgd_mom"]
        sgd = SGD(lr=rate, decay=dec, momentum=mom, \
            nesterov=True)
        model.compile(loss='categorical_crossentropy', \
            optimizer=sgd, metrics=['accuracy'])
    # Get traning and validation data
    tvRatio = para["tvRatio"]
    al_X = al_X.astype("float32")
    al_X /= 255
    # (X_train, X_valid, y_train, y_valid) = \
    #     getTrainValidSet(al_X, al_y, tvRatio)
    # model.fit(X_train, y_train,
    #         batch_size=batch_size,
    #         nb_epoch=nb_epoch,
    #         validation_data=(X_valid, y_valid),
    #         shuffle=True)
    print al_y
    model.fit(al_X, al_y,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            validation_split=0.1,
            shuffle=True)
    # eva(model, X_valid, y_valid, para)
    return model
