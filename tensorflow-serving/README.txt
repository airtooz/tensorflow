[DNN TRAINING USAGE]
    1. launch mnist DNN trainer by "python mnist_DNN_train.py MODEL_STRUC ITER BATCH -d [DIR]", 
       also see "python mnist_DNN_train.py -h"

[SERVER-CLIENT USAGE]
    1. execute ./run_codegen.sh to produce mnist_DNN_pb2.py
    2. execute server by "python mnist_serving.py TRAINED_MODEL", also see "python mnist_serving.py -h"
    3. launch client by "python mnist_client.py IMAGE -l [LABEL]", also see "python mnist_client.py -h"
