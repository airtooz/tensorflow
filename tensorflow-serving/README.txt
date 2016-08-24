Tensorflow Serving MNIST v2.0.2

-----Intro-----
Before trying this example you should install gRPC first. You may refer to the document at http://www.grpc.io/docs/ in order to help you install dependencies.

[Hello gRPC]
Please install grpcio-tools and grpcio using pip.
pip install grpcio-tools
pip install grpcio
    
You may try the following command to make sure if you successfully installed gRPC.
1. cd ~/grpc/examples/python/helloworld/
2. ./run_codegen.sh
***Trouble Shooting***
A. If you get an error: "Can't recognize syntax 'proto3'..." means that your protoc version isn't the latest , please refer to https://github.com/google/protobuf/releases, find "protoc-3.0.0...linux-...", download it, and add it to PATH. You may use: sudo cp "your protoc file absolute directory" "/usr/bin". Once you've done it, go back to the terminal and type: "protoc -version" to see if it shows "libprotoc 3.0.0"
B. If you get an error about protoc-gen-grpc, open the run_codegen.sh and change it into the following format: python –m grpc.tools.protoc –I“protos folder directory” –python_out=. –grpc_python_out=. “the objective proto file directory”
***Trouble Shooting***
3. After running step 2. , type ./run_server.sh
4. Open another terminal, go to the same directory (step 1.), and type ./run_client.sh
5. If the client side received "Hello you" means that you successfully installed gRPC!

-----Intro-----

-----Tensorflow serving-----
Please clone all files in this current directory.

[DNN TRAINING USAGE]
1. launch mnist DNN trainer by "python mnist_DNN_train.py MODEL_STRUC ITER BATCH -d [DIR]", 
    also see "python mnist_DNN_train.py -h"

[SERVER-CLIENT USAGE]
1. execute ./run_codegen.sh to produce mnist_DNN_pb2.py
2. execute server by "python mnist_serving.py TRAINED_MODEL", also see "python mnist_serving.py -h"
3. launch client by "python mnist_client.py IMAGE -l [LABEL]", also see "python mnist_client.py -h"
-----Tensorflow serving-----
