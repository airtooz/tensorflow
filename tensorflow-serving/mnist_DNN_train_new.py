import tensorflow as tf
import numpy as np
import time
import sys
import os
import argparse
import json
import tensorflow.contrib.learn as skflow
from tensorflow.contrib import layers
from tensorflow.examples.tutorials.mnist import input_data

def readdata(filename):
	A = []
	f = open(filename, 'r')
	for line in f:
		A.append(float(line[0:-2]))
	return A

def modelINFO(model_params, out):
	out.write("----------------MODEL INFO----------------"+'\n')
	out.write(("  Model Name:").ljust(25))
	out.write(str(model_params["name"])+'\n')
	out.write(("  Input Size:").ljust(25))
	out.write(str(model_params["in_size"])+'\n')
	out.write(("  Output Size:").ljust(25))
	out.write(str(model_params["out_size"])+'\n')
	out.write(("  Hidden Layers:").ljust(25))
	out.write(str(model_params["layer_num"])+'\n')
	out.write(("  Number of Nodes:").ljust(25))
	out.write(str(model_params["layer_size"])+'\n')
	out.write(("  Learning Rate:").ljust(25))
	out.write(str(model_params["L_rate"])+'\n')
	out.write(("  Dropout Prob:").ljust(25))
	out.write(str(model_params["prob"])+'\n')
	 

class DNN_classifier:
	def __init__(self, name, in_size=784, out_size=10, layer_num=1, layer_size=[200], L_rate=0.01, drop=0.8):
		self.name = name
		
                ## input ##
		self.in_size = in_size
		self.out_size = out_size
		self.x_ = tf.placeholder(tf.float32, [None, in_size], name="x_")
		self.y_ = tf.placeholder(tf.float32, [None, out_size], name="y_")
		self.prob_ = tf.placeholder(tf.float32, name="prob")
		
                ## result ##
		self.y = None
		self.loss = None
		
                ## model ##
		self.layer_num = layer_num
		self.layer_size = layer_size
		##self.weights = []
		##self.biases = []
		##self.acts_d = []
                self.FCLs = []
                self.FCLs_d = []
		self.drop = drop
		
                ## optimizer initialize ##
		#start_l_rate = L_rate
		self.global_step = tf.Variable(0, trainable=False)
		#self.L_rate = tf.train.exponential_decay(start_l_rate, self.global_step, decay_step, decay_rate, staircase=False)
                self.L_rate = L_rate
		#self.optimizer = tf.train.AdamOptimizer(learning_rate=self.L_rate)
		self.trainer = None

        	## create saver ##
		self.saver = None

        	## create session ##
		self.sess = tf.Session()
        
        def exponential_decay(self, l_rate, global_step):
	        decay_step = 1000
	        decay_rate = 0.5
                staircase = False
                return tf.train.exponential_decay(l_rate, global_step, decay_step, decay_rate, staircase)


	def init_w(self, size, name):
		return tf.Variable(tf.truncated_normal(size, stddev=0.01), name=name)

	def init_b(self, size, name):
		return tf.Variable(tf.zeros(size), name=name)

	def build_model(self):	
		'''w = self.init_w([self.in_size, self.layer_size], 
				name ='W' + str(1))
		b = self.init_b([self.layer_size], 
				name='b' + str(1))
		h = tf.nn.relu(tf.matmul(self.x_, w) + b)
		h_d = tf.nn.dropout(h, self.prob_)
		self.weights.append(w)
		self.biases.append(b)
		self.acts_d.append(h_d)
		for i in xrange(1, self.layer_num - 1):
			w = self.init_w([self.layer_size, self.layer_size], 
					name ='W' + str(i+1))
			b = self.init_b([self.layer_size], 
					name='b' + str(i+1))
			h = tf.nn.relu(tf.matmul(self.acts_d[i-1], w) + b)
			h_d = tf.nn.dropout(h, self.prob_)
			self.weights.append(w)
			self.biases.append(b)
			self.acts_d.append(h_d)
		w = self.init_w([self.layer_size, self.out_size], 
				name ='W' + str(len(self.weights)+1))
		b = self.init_b([self.out_size], 
				name='b' + str(len(self.biases)+1))
		self.y = tf.nn.softmax(tf.matmul(self.acts_d[-1], w) + b)
		self.weights.append(w)
		self.biases.append(b)
		assert len(self.weights) == len(self.biases)
		assert len(self.biases) == len(self.acts_d) + 1
		assert len(self.weights) == self.layer_num'''

        	fully_connected = layers.fully_connected(self.x_, 
                                                 self.layer_size[0], 
                                                 weights_regularizer=layers.l2_regularizer(0.1), 
                                                 biases_regularizer=layers.l2_regularizer(0.1), 
                                                 scope='FCL'+str(1))
		fully_connected_d = layers.dropout(fully_connected, keep_prob=self.prob_)
		self.FCLs.append(fully_connected)
		self.FCLs_d.append(fully_connected_d)
		for layer in xrange(1, self.layer_num):    
			fully_connected = layers.fully_connected(self.FCLs_d[-1], 
                                                     self.layer_size[layer], 
                                                     weights_regularizer=layers.l2_regularizer(0.1), 
                                                     biases_regularizer=layers.l2_regularizer(0.1), 
                                                     scope='FCL'+str(layer+1))
			fully_connected_d = layers.dropout(fully_connected, keep_prob=self.prob_)
			self.FCLs.append(fully_connected)
			self.FCLs_d.append(fully_connected_d)

       		## calc predict and loss ##
		self.y, self.loss = skflow.models.logistic_regression(self.FCLs[-1], self.y_, init_stddev=0.01)

		'''## calc loss ##
                self.loss = -tf.reduce_sum(tf.log(self.y)*self.y_ ) ## cross entropy
                
                ## gradient clip ##
                grad = self.optimizer.compute_gradients(self.loss)
                clipped_grad = [(tf.clip_by_value(g, -1., 1.), var) if g is not None else (tf.zeros_like(var), var) for g, var in grad]
                # -------------- ##
                self.trainer = self.optimizer.apply_gradients(clipped_grad, global_step=self.global_step)
		#self.trainer = self.optimizer.minimize(self.loss, global_step=self.global_step)'''

                ## creat trainer ##
                self.trainer = tf.contrib.layers.optimize_loss(loss=self.loss, 
                                                               global_step=self.global_step, 
                                                               learning_rate=self.L_rate, 
                                                               optimizer='Adam', 
                                                               clip_gradients=1, 
                                                               learning_rate_decay_fn=self.exponential_decay)
		
		## initialize var ##
		self.sess.run(tf.initialize_all_variables())
        	self.saver = tf.train.Saver(max_to_keep=5)

	def training(self, x_batch, y_batch):
		if x_batch.shape[1] != self.in_size:
			raise "Incorrect input size!"
		if y_batch.shape[1] != self.out_size:
			raise "Incorrect output size!"
                self.sess.run(self.trainer, feed_dict={self.x_:x_batch, self.y_:y_batch, self.prob_:self.drop})	

	def testing(self, x_test, y_test):
		if x_test.shape[1] != self.in_size:
			raise "Incorrect input size!"
		if y_test.shape[1] != self.out_size:
			raise "Incorrect output size!"
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return self.sess.run(accuracy, feed_dict={self.x_:x_test, self.y_:y_test, self.prob_:1})

	def validate(self, in_data):
		if in_data.shape[1] != self.in_size:
			raise "Incorrect input size!"
		prediction = tf.argmax(self.y, 1)
		result = self.sess.run(prediction, feed_dict={self.x_:in_data, self.prob_:1})
		return result

	def save_model(self, d, step):
		'''weights = []
		biases = []
		for num in xrange(self.layer_num):
			w = self.sess.run(self.weights[num])
			b = self.sess.run(self.biases[num])
			weights.append(w.tolist())
			biases.append(b.tolist())
		assert len(weights) == self.layer_num
		assert len(biases) == self.layer_num
		f = open(d+self.name+".model", 'w')
		json_to_write = json.dumps({
			"in_size": self.in_size,
			"out_size": self.out_size,
			"layer_num": self.layer_num,
			"layer_size": self.layer_size,
			"L_rate": str(self.sess.run(self.L_rate))[0:9],
			"prob": self.drop,
			"weights": weights,
			"biases": biases
		}, indent=4)
		f.write(json_to_write)
		f.close()'''
        	if not d[-1] == '/':
            		d = d + '/'
        	if not os.path.exists(d):
            		os.makedirs(d)
			f = open(d+self.name+".struc", 'w')
			json_to_write = json.dumps({
				"in_size": self.in_size,
				"out_size": self.out_size,
				"layer_num": self.layer_num,
				"layer_size": self.layer_size,
				"L_rate": self.L_rate,#str(self.sess.run(self.L_rate))[0:9],
				"prob": self.drop,
			}, indent=4)
			f.write(json_to_write)
			f.close()
		self.saver.save(self.sess, d+self.name)


if __name__ == '__main__':
	
	## argument parse ##
	parser = argparse.ArgumentParser(description="validation for single mnist input data!")	
	parser.add_argument("model", help="absolute path for model description file")
	parser.add_argument("iter", help="training step")
	parser.add_argument("batch", help="batch size")
	parser.add_argument("-d", "--dir", help="directory path for model saving, default(./model)")
	args = parser.parse_args()

	model_file = open(args.model, 'r')
	model_params = json.loads("".join(model_file.readlines()))
	model_file.close()
        if not model_params['layer_num'] == len(model_params['layer_size']):
        	raise "layer_num does not match size of layer_size!!"

	model_dir = None
	if args.dir != None:
		model_dir = args.dir
	else:
		model_dir = "./model"
	step = int(args.iter)
	batch = int(args.batch)
	Model = DNN_classifier(name=model_params["name"], 
                        in_size=model_params["in_size"], 
                        out_size=model_params["out_size"], 
                        layer_num=model_params["layer_num"], 
                        layer_size=model_params["layer_size"], 
                        L_rate=model_params["L_rate"], 
                        drop=model_params["prob"])
	Model.build_model()
	
	outfile = sys.stdout
	modelINFO(model_params, outfile)

        outfile.write('\n' + "----------------LOAD  DATA----------------" + '\n')
	Mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

	best_acc = -1
	outfile.write('\n' + "-----------------TRAINING-----------------" + '\n')
    	start = time.time()
	for i in xrange(step):
		batch_x, batch_y = Mnist_data.train.next_batch(batch)
		Model.training(batch_x, batch_y)
		if ((i+1)%50) == 0:
			acc = Model.testing(Mnist_data.test.images, Mnist_data.test.labels)
			outfile.write("  epoch: " + str(i+1).ljust(len(str(step))+4))
            		outfile.write("acc: " + "{:.4f}".format(acc).ljust(6+4) )
            		outfile.write("elapsed time: " + "{:.2f}".format(time.time() - start).rjust(5) + ' secs \n')
			if acc > best_acc:
				best_acc = acc
				Model.save_model(model_dir, i+1)
            	start = time.time()
		
	
	#print np.transpose(Mnist_data.test.images[0]).reshape(1,784)
	'''print Model.test(Mnist_data.test.images, Mnist_data.test.labels)
	print "Predict By Model:", Model.Validate(testdata)[0]
	if isLabel:
		print "Answer:", np.argmax(testlabel)'''
