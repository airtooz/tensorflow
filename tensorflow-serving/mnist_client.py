import tensorflow as tf
import numpy as np
import sys
import time
import argparse
import json
import mnist_DNN_pb2

from tensorflow.examples.tutorials.mnist import input_data
from grpc.beta import implementations
def readscript(filename):
	A = []
	f = file(filename)
	while True:
		line = f.readline().rstrip()
		if len(line)==0:
			break
		A.append(line)
	f.close()
	return A
def readdata(filename):
	data = []
	f = open(filename, 'r')
	for line in f:
		data.append(float(line[0:-2]))
	return data

def readmultidata(filelist):
	for i in xrange(len(filelist)):
		data = []
		f = open(filelist[i], 'r')
		for line in f:
			data.append(float(line[0:-2]))
		yield mnist_DNN_pb2.ImageArray(in_type="test", image=data)
	
	

def run(addr, testimage): # removed testlabel
	addr_list = addr.split(":")
	ip = addr_list[0]
	port = addr_list[1]
	channel = implementations.insecure_channel(ip, int(port))
	stub = mnist_DNN_pb2.beta_create_mnist_Inference_stub(channel)
	print "Sending Request to Server located at", (ip + ':' + port + "...")
	if len(testimage) == 1:
		testdata = readdata(testimage[0])
		reply = stub.GetInput(mnist_DNN_pb2.ImageArray(in_type="test", image=testdata), 30)
	else:
		test_iter = readmultidata(testimage)
		reply = stub.GetMultiInput(test_iter, 180)
	return reply.digits

def input_iter(data_list):
        for i in xrange(len(data_list)):
                d = data_list[i].tolist()
                yield mnist_DNN_pb2.ImageArray(in_type="train", image=d, label=0)

def run_test(addr):
	addr_list = addr.split(":")
	ip = addr_list[0]
	port = addr_list[1]
	channel = implementations.insecure_channel(ip, int(port))
	stub = mnist_DNN_pb2.beta_create_mnist_Inference_stub(channel)
	print "Sending Request to Server located at", (ip + ':' + port + "...")
        Mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
        gen = input_iter(Mnist_data.test.images)
        reply = stub.GetMultiInput(gen, 30)
        answers = np.argmax(Mnist_data.test.labels, 1)
        acc = np.mean(np.equal(reply.digits, answers))
        return acc


if __name__ == '__main__' :
	
    ### Argument Parse ###
	parser = argparse.ArgumentParser(description="mnist DNN Server!")	
	parser.add_argument("imagescript", help="absolute path for test images script, make sure all images included are in the current directory")
#	parser.add_argument("-l", "--label", help="absolute path for test label(if exist)")
	parser.add_argument("-a", "--addr", help="address for the server, default: 127.0.0.1:50051")
    
	args = parser.parse_args()
#	testlabel = None
#	if not args.label == None:
#		l = readdata(args.label)
#        	assert len(l) == 10
#        	testlabel = np.argmax(l)

	if args.addr == None:
        	address = '127.0.0.1:50051'
    	else:
        	address = args.addr
    
        print "----------run model testing----------"
        acc = run_test(address)
        print "accuracy:", acc, '\n' 
    	Script = readscript(args.imagescript)
        predicts = run(address, Script) # removed testlabel
	for i in xrange(len(Script)):
	   	print "From server, predict the testimage \"",Script[i],"\" be: ",predicts[i]

