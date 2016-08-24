Tensorflow Serving MNIST v2.0.2

-----Intro-----
Before trying this example you should install gRPC first. You may refer to the document at http://www.grpc.io/docs/ in order to help you install dependencies.

#[Hello gRPC]
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
Do not revise any code in mnist_client.py, mnist_server.py, mnist_DNN_train.py and the "protos" folder
The run_codegen file shouldn't be revised unless error is occured while running it.
The other files (with names ended in _example), can be revised or try your own data.

* test_model_example.json : a file describing how you want to train your model.
* mnist_example: files which are the image data, should be in the format of 28*28, with every element be a floating number betweeen 0 and 1. (This is the format of MNIST data, you may refer to the file containing the data and get the data you want.)
* script_example: a file with all mnist_example file names in it (one for each line), note that you need to have all the image files in the same directory. 

[DNN TRAINING USAGE]
1. You may revise "test_model_example.json" to make your own training model
2. Launch mnist DNN trainer by "python mnist_DNN_train.py test_model_example.json ITER BATCH -d [DIR]", 
    also see "python mnist_DNN_train.py -h". (The ITER and BATCH are parameters during training, you have to specify them, for e.g. 1000 100)
3. After training, you should see a folder called: "model"

[SERVER-CLIENT USAGE]
1. execute ./run_codegen.sh to produce mnist_DNN_pb2.py (do not revise anything in this produced code)
2. execute server by "python mnist_server.py ./model/test_model.struc ./model/test_model", also see "python mnist_server.py -h"
3. launch client by "python mnist_client.py script_example", also see "python mnist_client.py -h"
4. If you want to try serving via different computers, you have to specify the IP addresses and port number while executing mnist_server.py and mnist_client.py. (See usage by typing -h while executing)

-----Tensorflow serving-----
