******NOTICE******
One only needs to specify the prefix (trained_model), and then both models (trained_model_cl.h5 and trained_model_en.h5) will be generated/loaded. 
******NOTICE******

In homework assignment, I use the Kaggle library with Theano backend.
To train the model : sh train.sh {path_to_data} {trained_model}
To test to model : sh test.sh {path_to_data} {trained_model} {submit.csv}
Notice that there are two .h5 files under this folder. They are “trained_model_cl.h5” and “trained_model_en.h5”. One only needs to specify the prefix (name before _cl.h5 and _en.h5), then both model will be generated/loaded. 
