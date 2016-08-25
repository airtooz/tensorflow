from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from yahoo_finance import Share
import datetime
import matplotlib.finance as finance
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SAVE = False # Very important!! True if you want to renew figure data, false otherwise.
EPISODE = 10000 # Total episodes to run
STEP = 10000  # Step limitation in an episode, this is actually a bug in my code. This parameter should be equal or larger than the training data size, so here I revised into a large number. However, you may still tune this parameter by yourself, it won't occur any error.
TEST = 10 # Test episode

GAMMA = 0.9 # discount factor, may tune it as you like
BUFFER_SIZE = 10000 # the replay buffer size, may tune it as you like
BATCH_SIZE = 32 # minibatch size, may tune it as you like, no larger than BUFFER_SIZE
INITIAL_EPSILON = 0.8 # initial large, in order to explore more, may tune it as you like.
FINAL_EPSILON = 0.1 # explore less, greedy most time, may tune it as you like, smaller than INITIAL_EPSILON.
DAY_LENGTH = 10 # the total days for a training data, also the dim of features.
START = "2011-01-01" # Training data start date
TRAIN_END = "2015-12-31" # Training data end date. Note that the test data will start right after this date.
SAVE_IMAGE_DIR = "/home/airchen/Documents/coding/stock/" # Please type the directory where you want to save your images, note that you shall add /home/USERNAME/ before that directory, and a '/' at the very back!
_ID = 2330 # By default, TSMC (2330)
DPI = 80 # This can be revised, and as I assume this DPI shouldn't be too large or else it will overfit the data, but not too small either, or else the it may underfit the data. I will assume 32~64 will be worth trying, and see which gives better performance.

# Get stock data from yahoo finance
stock = Share(str(_ID)+'.TW')
today = datetime.date.today()
stock_data = stock.get_historical(START, str(today)) # Total data downloaded
print("Historical data since", START,": ", len(stock_data)) # Total length of raw data
stock_data.reverse() # Reverse the data since it starts from the latest data.

#----------Remove the data with 0 volume---------#
i = 0
while( i < len(stock_data)):
    if (int(stock_data[i].get('Volume')) <= 0):
        stock_data.remove(stock_data[i])
        i = -1
    i += 1
#----------Remove the data with 0 volume---------#
print("Remove the datas with zero volume, total data ",len(stock_data))
#----------Count training data length---------#
train_data = stock.get_historical(START, TRAIN_END)
train_data.reverse()

i = 0
while( i < len(train_data)):
    if (int(train_data[i].get('Volume')) <= 0):
        train_data.remove(train_data[i])
        i = -1
    i += 1
#----------Count training data length---------#

# Calculate K and D
BUFFER_DAYS = 9 # In this case, K and D needs a nine day buffer in order to calculate its first data, thus we set this to 9. Please see numerical_features code: dqn_predict.py for furthur info.

K = []
D = []
util = []
for i in xrange(len(stock_data)):
	util.append(float(stock_data[i].get('Close')))
	if i >= (BUFFER_DAYS-1):
		assert len(util) == BUFFER_DAYS

		#----RSV----		
		if max(util) == min(util):
			RSV = 0.0
		else:
			RSV = (util[len(util)-1] - min(util))/(max(util)-min(util))
		#----RSV----

		#----K----
		if i == (BUFFER_DAYS-1):
			temp_K = 0.5*0.6667 + RSV*0.3333
			K.append(temp_K)
		else:
			temp_K = K[-1]*0.6667 + RSV*0.3333
			K.append(temp_K)
		#----K----

		#----D----
		if i == (BUFFER_DAYS-1):
			D.append(0.5*0.6667 + temp_K*0.3333)
		else:
			D.append(D[-1]*0.6667 + temp_K*0.3333)
		#----D----
		util.pop(0)
		assert len(util) == (BUFFER_DAYS-1)

# Feed in images

def save_img(K,D, filename): # This is the function for saving the image of numerical features, since this part is important, I will type a document for explaining how to create other numerical features' images and feed them into the dqn.
	print("Saving images into ",filename,"...")
	for i in xrange(len(K)-DAY_LENGTH+1):
		fig, ax = plt.subplots(nrows=1,ncols=1)
		fig.set_size_inches(1,1)
		ax.plot([i,i+1,i+2,i+3,i+4,i+5,i+6,i+7,i+8,i+9], [K[i], K[i+1], K[i+2], K[i+3], K[i+4], K[i+5], K[i+6], K[i+7], K[i+8], K[i+9]],'r',[i,i+1,i+2,i+3,i+4,i+5,i+6,i+7,i+8,i+9], [D[i],D[i+1],D[i+2],D[i+3],D[i+4],D[i+5],D[i+6],D[i+7],D[i+8],D[i+9]],'b') # Please don't change the color 'r' and 'b', only change it if you understand how to manipulate the colors of matplot, I will intoduce it in the documents. Here, the plots are drawn with 10 days of data (DAY_LENGTH).
		ax.set_ylim(0,1) # K and D are in the range of 0% and 100%, i.e. 0~1
		plt.axis('off') # We close the axis in order to make our image clean with only the K, D curves
		if not os.path.exists(SAVE_IMAGE_DIR+filename+'/'): # Create the folder storing the images, if you already have the folder, it will simply renew the images inside the current folder.
			os.makedirs(SAVE_IMAGE_DIR+filename+'/')
		fig.savefig(SAVE_IMAGE_DIR+filename+'/'+filename+'_'+str(i)+'.png', dpi=DPI) # "/home/airchen/Documents/coding/stock/"+filename+'/'+filename+'_'+str(i)+'.png' , is the directory and the file name for saving the images, the "filename" is a folder name, will be created under the SAVE_IMAGE_DIR.
		fig.clear()
		plt.close(fig)
        print("Finished!!")
        print()
        
def get_img(file_dir): # This is to simply get the images from the folder. The file_dir will be mentioned later.
	img = mpimg.imread(file_dir)
	return img

relative_close = [] # How we will calculate the reward
for i in xrange(DAY_LENGTH,len(stock_data)-4):
	relative_close.append(float(stock_data[i+4].get('Close'))-float(stock_data[i-1].get('Close'))) # The close price of six days later minus the price tomorrow

if SAVE: # renew or create the images if SAVE is True
	save_img(K, D, "KD")

data_length = len(relative_close)-DAY_LENGTH # discard the last image since no label is available

train_image = []
train_label = []

assert len(K) == len(D)
assert len(relative_close) < len(K)

# Here is the image processing area, the processing method will also be in the documents. I will merely explain the usage here.
for i in xrange(len(train_data)):
	if (i+1) % 10 == 0:
		print("Processing training data ",(i+1)," out of ",len(train_data))
		
	file_dir = "/home/airchen/Documents/coding/stock/KD/KD_"+str(i)+".png" # This file_dir is very important since it is the directory you get your images. Sometimes we won't save the images, we will want to use the already existed ones. Please specify the file_dir by following this example.
        temp = np.asarray(get_img(file_dir))
	for pixel_x in xrange(len(temp)):
		for pixel_y in xrange(len(temp[0])):
			for depth in xrange(len(temp[0][0])):
				if temp[pixel_x][pixel_y][0] != 1 and temp[pixel_x][pixel_y][1]!=1 and temp[pixel_x][pixel_y][2]==1: # Red 
					temp[pixel_x][pixel_y][0] = 0.8
				elif temp[pixel_x][pixel_y][0] == 1 and temp[pixel_x][pixel_y][1]!=1 and temp[pixel_x][pixel_y][2]!=1: # Blue
					temp[pixel_x][pixel_y][0] = 0.4
				elif temp[pixel_x][pixel_y][0] != 1 and temp[pixel_x][pixel_y][1]!=1 and temp[pixel_x][pixel_y][2]!=1: # Red and Blue cross
					temp[pixel_x][pixel_y][0] = 0.
				elif temp[pixel_x][pixel_y][0] == 1 and temp[pixel_x][pixel_y][1]==1 and temp[pixel_x][pixel_y][2]==1 and temp[pixel_x][pixel_y][3]==1: # White
					temp[pixel_x][pixel_y][0] = 1.
	buff = temp[:,:,0] # We only take the first layer to feed into dqn.					
	train_image.append(buff)
	train_label.append(relative_close[i])
train_image = np.asarray(train_image,dtype=np.float)

test_image = []
test_label = []
for i in xrange(len(train_data),data_length): # Some of the images won't be used due to the length of relative_close_price is shorter than the data_length.
	print("Processing testing data ",(i+1)-len(train_data)," out of ",data_length-len(train_data))
	file_dir = "/home/airchen/Documents/coding/stock/KD/KD_"+str(i)+".png"
        temp = np.asarray(get_img(file_dir))

	for pixel_x in xrange(len(temp)):
		for pixel_y in xrange(len(temp[0])):
			for depth in xrange(len(temp[0][0])):
				if temp[pixel_x][pixel_y][0] != 1 and temp[pixel_x][pixel_y][1]!=1 and temp[pixel_x][pixel_y][2]==1:
					temp[pixel_x][pixel_y][0] = 0.8
				elif temp[pixel_x][pixel_y][0] == 1 and temp[pixel_x][pixel_y][1]!=1 and temp[pixel_x][pixel_y][2]!=1:
					temp[pixel_x][pixel_y][0] = 0.4
				elif temp[pixel_x][pixel_y][0] != 1 and temp[pixel_x][pixel_y][1]!=1 and temp[pixel_x][pixel_y][2]!=1:
					temp[pixel_x][pixel_y][0] = 0.
				elif temp[pixel_x][pixel_y][0] == 1 and temp[pixel_x][pixel_y][1]==1 and temp[pixel_x][pixel_y][2]==1 and temp[pixel_x][pixel_y][3]==1:
					temp[pixel_x][pixel_y][0] = 1.
	buff = temp[:,:,0]					
	test_image.append(buff)
	test_label.append(relative_close[i])
test_image = np.asarray(test_image,dtype=np.float)
	
class TWStock():
	def __init__(self,train_image,test_image,train_label,test_label):
		self.train_image = train_image
		self.test_image = test_image
		self.train_label = train_label
		self.test_label = test_label
		self.stock_index = 0

		print("Training Data: ",len(train_image))
		print("Testing Data: ",len(test_image))

	def train_reset(self):
		self.stock_index = 0
		return self.train_image[self.stock_index]

	def test_reset(self):
		self.stock_index = 0
		return self.test_image[self.stock_index]
		
	# 0: observe, 1: having stock, 2: no stock
	def train_step(self,action): # for training, feed training data
		self.stock_index+=1
		action_reward = self.train_label[self.stock_index-1] # The last label can be omitted since the last state won't do any more actions
		#action_reward = self.train_data[self.stock_index][0] - self.train_data[self.stock_index-1][0]
		#action_reward = self.train_data[self.stock_index][DAY_LENGTH-1] - self.train_data[self.stock_index][DAY_LENGTH-2]
		if action == 0:
			action_reward = 0
		elif action == 2:
			action_reward = -1*action_reward
		stock_done = False
		if(self.stock_index)>= len(self.train_image)-1:
			stock_done = True
        	else:
           		stock_done = False
		return self.train_image[self.stock_index], action_reward, stock_done, 0

	def test_step(self,action): # for testing, feed testing data
		self.stock_index+=1
		action_reward = self.test_label[self.stock_index-1] # The last label can be omitted since the last state won't do any more actions
		#action_reward = self.train_data[self.stock_index][0] - self.train_data[self.stock_index-1][0]
		#action_reward = self.test_data[self.stock_index][DAY_LENGTH-1] - self.test_data[self.stock_index][DAY_LENGTH-2]
		if action == 0:
			action_reward = 0
		elif action == 2:
			action_reward = -1*action_reward
		stock_done = False
		if(self.stock_index)>= len(self.test_image)-1:
			stock_done = True
        	else:
           		stock_done = False
		return self.test_image[self.stock_index], action_reward, stock_done, 0

class DQN():
	def __init__(self,env):
		# experience replay
		self.replay_buffer = deque()
		# initialize parameters
		self.time_step = 0
		self.epsilon = INITIAL_EPSILON
		#self.state_dim = [1,80,80,1]
		self.action_dim = 3
		self.create_Q_network()
		self.create_training_method()

		# create session
		#g_record = tf.Graph()
		#self.g_session = tf.InteractiveSession(graph=g_record)
		self.t_session = tf.InteractiveSession()

		#with g_record.as_default():
		self.R = tf.placeholder("float", shape = None)
		self.T = tf.placeholder("float", shape = None)
		R_summ = tf.scalar_summary(tags = "testing_reward", values = self.R)
		T_summ = tf.scalar_summary(tags = "training_reward", values = self.T)

		self.merged_summ = tf.merge_all_summaries()
		self.writer = tf.train.SummaryWriter(logdir = "/home/airchen/Documents/coding/stock", graph = self.t_session.graph)

		
		self.t_session.run(tf.initialize_all_variables())
	
	def get_summ(self):
		return self.t_session, self.merged_summ, self.R,self.T, self.writer

	def create_Q_network(self): 
		# Use CNN, we won't use MLP for images...
		# weights and biases
		BIAS_SHAPE1 = 4 # May tune it as you like
		BIAS_SHAPE2 = 8 # May tune it as you like
		W_conv1 = tf.Variable(tf.truncated_normal(shape = [10,10,1,BIAS_SHAPE1],stddev = 0.01)) # The fist and second parameter is the filter height and width...you may tune it but either of them can't exceed DPI !
		b_conv1 = tf.Variable(tf.constant(0.01,shape = [BIAS_SHAPE1]))
		W_conv2 = tf.Variable(tf.truncated_normal(shape = [5,5,BIAS_SHAPE1,BIAS_SHAPE2],stddev = 0.01)) # The fist and second parameter is the filter height and width...you may tune it but either of them can't exceed DPI !
		b_conv2 = tf.Variable(tf.constant(0.01,shape = [BIAS_SHAPE2]))
		W_fc = tf.Variable(tf.truncated_normal(shape = [DPI*DPI*BIAS_SHAPE2,self.action_dim],stddev = 0.01))
		b_fc = tf.Variable(tf.constant(0.01,shape = [self.action_dim]))

		# Layer implementation, dont' change these only if you want to add more layers
		self.state_input = tf.placeholder("float",[None,DPI,DPI])
		x = tf.reshape(self.state_input,[-1,DPI,DPI,1])
		h_conv1 = tf.nn.relu(tf.nn.conv2d(x,W_conv1,strides = [1,1,1,1],padding = 'SAME') + b_conv1)
		h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1,W_conv2,strides = [1,1,1,1],padding = 'SAME') + b_conv2)
		hidden = tf.reshape(h_conv2,[-1,DPI*DPI*BIAS_SHAPE2])
		self.Q_value = tf.matmul(hidden,W_fc) + b_fc
		

	def create_training_method(self):
		self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot key vector
		self.y_input = tf.placeholder("float",[None])
		Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost) # There are many optimizers to use, e.g. RMSPropOptimizer..., please see tensorflow API for furthur info.

	def train_Q_network(self):
		self.time_step+=1
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		next_state_batch = [data[3] for data in minibatch]
		state_batch = np.asarray(state_batch,dtype=np.float)
		next_state_batch = np.asarray(next_state_batch,dtype=np.float)
		
		# step 2: calculate y
		y_batch = []
		Q_value_batch = self.Q_value.eval(feed_dict = {self.state_input:next_state_batch})
		for i in range(0,BATCH_SIZE):
			done = minibatch[i][4]
			if done:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA*np.max(Q_value_batch[i]))
				
		self.optimizer.run(feed_dict = {
			self.y_input:y_batch,
			self.action_input:action_batch,
			self.state_input:state_batch})

	def egreedy_action(self,state): # during training 
		Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]},session = self.t_session)[0] # Unknown
		if random.random() <= self.epsilon:
			return random.randint(0,self.action_dim-1)
		else:
			return np.argmax(Q_value)
		self.epsilon -= (Initial_EPSILON-FINAL_EPSILON)/10000
	
	def action(self,state): # during testing
		return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:[state]})[0])
		
	def perceive(self,state,action,reward,next_state,done):
		# assign the to be made action into a one hot vector
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
		if len(self.replay_buffer) > BUFFER_SIZE:
			self.replay_buffer.popleft()
		if len(self.replay_buffer) > BATCH_SIZE:
			self.train_Q_network()

def main():
	env = TWStock(train_image,test_image,train_label,test_label) # Initialize environment
	agent = DQN(env) # Initialize dqn agent
	sess,merged,R,T,writer = agent.get_summ()
	global_step = 0

	for episode in xrange(EPISODE):
		state = env.train_reset() # reset() returns observation
		# start training
		for step in xrange(STEP):
			global_step +=1
			if global_step %10 ==0:
				print("Running STEP ",global_step)
			action = agent.egreedy_action(state) # e-greedy action for training
			next_state,reward,done,info = env.train_step(action)
			agent.perceive(state,action,reward,next_state,done)
			state = next_state
			if done:
				break
			
		train_reward = 0.
		for i in xrange(TEST):
			state = env.train_reset()
			action0_count = 0
			action1_count = 0
			action2_count = 0
			for j in xrange(STEP):
				action = agent.action(state)
				if action == 0:
					action0_count += 1
				elif action == 1:
					action1_count += 1
				elif action == 2:
					action2_count += 1
			
				state, reward, done, info = env.train_step(action)
				train_reward += reward
				if done:
					break
			if i == 0:
				print("Action 0: ",action0_count,". Action 1: ", action1_count, ". Action 2: ", action2_count)
		print()
		avg_train_reward = train_reward/TEST
		
		total_reward = 0.
		for i in xrange(TEST):
			state = env.test_reset()
			action0_count = 0
			action1_count = 0
			action2_count = 0
			for j in xrange(STEP):
				action = agent.action(state) # direct action for test
				if action == 0:
					action0_count += 1
				elif action == 1:
					action1_count += 1
				elif action == 2:
					action2_count += 1
				
				state, reward, done, info = env.test_step(action)
				total_reward += reward
				if done:
					break
			if i == 0:
				print("Action 0: ",action0_count,". Action 1: ", action1_count, ". Action 2: ", action2_count)
		avg_reward = total_reward/TEST
		print ("Episode: ", episode,"Training Average Reward: ",avg_train_reward, " Evaluation Average Reward: ",avg_reward)
		record = sess.run(merged, feed_dict={R:avg_reward,T:avg_train_reward})
		writer.add_summary(record, global_step = global_step)
		writer.flush()

if __name__ == '__main__':
	main()
