from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
from collections import deque
from yahoo_finance import Share
import datetime

EPISODE = 10000 # Total episodes to run, feel free to tune it.
STEP = 10000 # Step limitation in an episode, this is actually a bug in my code. This parameter should be equal or larger than the training data size, so here I revised into a large number. However, you may still tune this parameter by yourself, it won't occur any error.
TEST = 10 # Test episode

GAMMA = 0.9 # discount factor, may tune it as you like
BUFFER_SIZE = 10000 # the replay buffer size, may tune it as you like
BATCH_SIZE = 32 # minibatch size, may tune it as you like, no larger than BUFFER_SIZE
INITIAL_EPSILON = 0.8 # initial large, in order to explore more, may tune it as you like.
FINAL_EPSILON = 0.1 # explore less, greedy most time, may tune it as you like, smaller than INITIAL_EPSILON.
DAY_LENGTH = 10 # the total days for a training data, also the dim of features.
FEATURE_NUM = 8 # Currently: close price, volume, K, D, RSI9, MA5, MA20, MA5-MA20. Remember to change this parameter if you add any new feature or remove any of them.
START = "2011-01-01" # Training data start date
TRAIN_END = "2015-12-31" # Training data end date. Note that the test data will start right after this date.
_ID = 2330 # By default, TSMC (2330)

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

# Available features: Close price, Volume, K, D, RSI9, MA5, MA20, MA5-MA20
# If you add or reduce any features here, you should remember to go change the FEATURE_NUM constant

BUFFER_DAYS = 20 # This parameter means that the feature that requires most days to calculate, e.g. MA20 requires 20 days (Start to have data at the 20th day). If you add something like MA30, then this parameter should be 30.

data = np.zeros((len(stock_data)-(BUFFER_DAYS-1),FEATURE_NUM), dtype = np.float)
util = []
for i in xrange(len(stock_data)):
	util.append(float(stock_data[i].get('Close')))
	rise = 0.
	fall = 0.
	if i >= (BUFFER_DAYS-1):
		assert len(util) == BUFFER_DAYS
		data[i-(BUFFER_DAYS-1)][0] = float(stock_data[i].get('Close')) # Get close price
		data[i-(BUFFER_DAYS-1)][1] = float(float(stock_data[i].get('Volume'))/1000000.) # Get volume and count in millions

		#----RSI9----
		for j in range(len(util)-8,len(util)):
			if util[j] >= util[j-1]:
				rise += (util[j]-util[j-1])
			else:
				fall += (util[j-1]-util[j])
		if rise == 0 and fall == 0:
			data[i-(BUFFER_DAYS-1)][2] = 0.5
		else:
			data[i-(BUFFER_DAYS-1)][2] = rise/(rise+fall)
		#----RSI9----
		u9 = util[len(util)-9:len(util)]
		#----RSV----		
		if max(u9) == min(u9):
			RSV = 0.0
		else:
			RSV = (u9[len(u9)-1] - min(u9))/(max(u9)-min(u9))
		#----RSV----

		#----K----
		if i == (BUFFER_DAYS-1):
			K = 0.5*0.6667 + RSV*0.3333
			data[i-(BUFFER_DAYS-1)][3] = K
		else:
			K = data[i-BUFFER_DAYS][3]*0.6667 + RSV*0.3333
			data[i-(BUFFER_DAYS-1)][3] = K
		#----K----

		#----D----
		if i == (BUFFER_DAYS-1):
			data[i-(BUFFER_DAYS-1)][4] = 0.5*0.6667 + K*0.3333
		else:
			data[i-(BUFFER_DAYS-1)][4] = data[i-BUFFER_DAYS][4]*0.6667 + K*0.3333
		#----D----
		
		#----MA5----
		data[i-(BUFFER_DAYS-1)][5] = sum(util[len(util)-5:len(util)])/5.0
		#----MA5----

		#----MA20----
		data[i-(BUFFER_DAYS-1)][6] = sum(util)/20.0
		#----MA20----

		#---(MA5-MA20)---
		data[i-(BUFFER_DAYS-1)][7] = data[i-(BUFFER_DAYS-1)][5]-data[i-(BUFFER_DAYS-1)][6]
		#---(MA5-MA20)---

		util.pop(0)
		assert len(util) == (BUFFER_DAYS-1)

box = np.zeros((len(data)-(BUFFER_DAYS-1),DAY_LENGTH*FEATURE_NUM), dtype = np.float) # The size of the data

# Assigning the data to the box, dor each row, there includes FEATURE_NUM*DAY_LENGTH data, there are FEATURE_NUM features and DAY_LENGTH data for a batch, we will reshape this when we feed into the placeholder.
for m in xrange(len(data)-(BUFFER_DAYS-1)):
	for i in xrange(FEATURE_NUM):
		for j in xrange(DAY_LENGTH):
			box[m][i*DAY_LENGTH+j] = data[m+j][i]
			
# Define TWStock class	
class TWStock():
	def __init__(self, data, train_length):
		self.stock_data = data
		self.train_data = self.stock_data[0:train_length] # Training Data
		self.test_data = self.stock_data[train_length:len(stock_data)] # Testing Data
		self.stock_index = 0 # This parameter is important to understand, this index will add 1 for every step, and supposedly, it should add until the last data, instead of merely running STEP times.
		print("Training Data: ",len(self.train_data))
		print("Testing Data: ",len(self.test_data))

	def train_reset(self): # For each episode, reset the index into 0, i.e. start runnning from the fist batch of data again
		self.stock_index = 0
		return self.train_data[self.stock_index] # Return initial state, in this code, the state is the data batch

	def test_reset(self): # For each episode, reset the index into 0, i.e. start runnning from the fist batch of data again
		self.stock_index = 0
		return self.test_data[self.stock_index] # Return initial state, in this code, the state is the data batch
		
	# 0: Observe, 1: Buy, 2: Sell
	def train_step(self,action): # For training, feed training data
		self.stock_index+=1 # Just as I mentioned before, this means to go on to the next batch(FEATURE_NUM*DAY_LENGTH) of data
		action_reward = self.train_data[self.stock_index+1][DAY_LENGTH-1] - self.train_data[self.stock_index][DAY_LENGTH-1] # The reward is the close price for tomorrow minus today's
		if action == 0:
			action_reward = 0 # Do nothing, 0 reward
		elif action == 2:
			action_reward = -1*action_reward
		stock_done = False
		if(self.stock_index)>= len(self.train_data)-2:
			stock_done = True # Already at the final state(last batch of data)
        	else:
           		stock_done = False
		return self.train_data[self.stock_index], action_reward, stock_done, 0

	def test_step(self,action): # for testing, feed testing data
		self.stock_index+=1
		action_reward = self.test_data[self.stock_index+1][DAY_LENGTH-1] - self.test_data[self.stock_index][DAY_LENGTH-1]
		if action == 0:
			action_reward = 0
		elif action == 2:
			action_reward = -1*action_reward
		stock_done = False
		if(self.stock_index)>= len(self.test_data)-2:
			stock_done = True
        	else:
           		stock_done = False
		return self.test_data[self.stock_index], action_reward, stock_done, 0

class DQN():
	def __init__(self,env):
		# experience replay
		self.replay_buffer = deque()
		# initialize parameters
		self.epsilon = INITIAL_EPSILON
		self.action_dim = 3 # Totally three actions
		self.create_Q_network()
		self.create_training_method()

		# create session, used for launching tensorflow and tensorboard
		self.t_session = tf.InteractiveSession()
		self.R = tf.placeholder("float", shape = None)
		self.T = tf.placeholder("float", shape = None)
		R_summ = tf.scalar_summary(tags = "testing_reward", values = self.R)
		T_summ = tf.scalar_summary(tags = "training_reward", values = self.T)

		self.merged_summ = tf.merge_all_summaries()
		self.writer = tf.train.SummaryWriter(logdir = "/home/airchen/Documents/coding/stock", graph = self.t_session.graph) # The logdir is the directory you want to log your tensorboard event files, please feel free to change it, and remember you want to always add: /home/USERNAME/ before the directory.
		
		self.t_session.run(tf.initialize_all_variables())
	
	def get_summ(self): # For writing events to tensorboard.
		return self.t_session, self.merged_summ, self.R,self.T, self.writer

	def create_Q_network(self): # You may switch between CNN and MLP, currently CNN.
		'''
		# Use MLP
		# weights and biase
		W1 = tf.Variable(tf.truncated_normal([self.state_dim,20]))
		b1 = tf.Variable(tf.constant(0.01, shape = [20]))
		W2 = tf.Variable(tf.truncated_normal([20,self.action_dim]))
		b2 = tf.Variable(tf.constant(0.01, shape = [self.action_dim]))

		# Layer implementation
		self.state_input = tf.placeholder("float",[None,self.state_dim])
		hidden = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
		self.Q_value = tf.matmul(hidden,W2) + b2
		'''
		# Use CNN
		# weights and biases
		BIAS_SHAPE1 = 8 # May tune it as you like
		BIAS_SHAPE2 = 4 # May tune it as you like
		W_conv1 = tf.Variable(tf.truncated_normal(shape = [FEATURE_NUM,5,1,BIAS_SHAPE1],stddev = 0.01)) # The second parameter is the filter width...you may tune it but can't exceed DAY_LENGTH !
		b_conv1 = tf.Variable(tf.constant(0.01,shape = [BIAS_SHAPE1]))
		W_conv2 = tf.Variable(tf.truncated_normal(shape = [FEATURE_NUM,2,BIAS_SHAPE1,BIAS_SHAPE2],stddev = 0.01)) # The second parameter is the filter width...you may tune it but can't exceed DAY_LENGTH !
		b_conv2 = tf.Variable(tf.constant(0.01,shape = [BIAS_SHAPE2]))
		W_fc = tf.Variable(tf.truncated_normal(shape = [FEATURE_NUM*DAY_LENGTH*BIAS_SHAPE2,self.action_dim],stddev = 0.01))
		b_fc = tf.Variable(tf.constant(0.01,shape = [self.action_dim]))

		# Layer implementation
		self.state_input = tf.placeholder("float",[None,FEATURE_NUM*DAY_LENGTH])
		x = tf.reshape(self.state_input,[-1,FEATURE_NUM,DAY_LENGTH,1])
		h_conv1 = tf.nn.relu(tf.nn.conv2d(x,W_conv1,strides = [1,1,1,1],padding = 'SAME') + b_conv1)
		h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1,W_conv2,strides = [1,1,1,1],padding = 'SAME') + b_conv2)
		hidden = tf.reshape(h_conv2,[-1,FEATURE_NUM*DAY_LENGTH*BIAS_SHAPE2])
		self.Q_value = tf.matmul(hidden,W_fc) + b_fc
		

	def create_training_method(self):
		self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot key vector
		self.y_input = tf.placeholder("float",[None])
		Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action)) # Our cost.
		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost) # There are many optimizers to use, e.g. RMSPropOptimizer..., please see tensorflow API for furthur info.

	def train_Q_network(self):
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
	env = TWStock(box, len(train_data)) # Initialize environment
	agent = DQN(env) # Initialize dqn agent
	sess,merged,R,T,writer = agent.get_summ() # Get summary for tensorboard

	for episode in xrange(EPISODE):
		state = env.train_reset() # reset() returns observation
		# start training
		for step in xrange(STEP):
			action = agent.egreedy_action(state) # e-greedy action for training
			next_state,reward,done,info = env.train_step(action)
			agent.perceive(state,action,reward,next_state,done)
			state = next_state
			if (step+1) % 200 == 0: # Print out current progress
                                print("Running episode:",episode, "Step:", step+1)
			if done: # break only if it run to the end of the training data, or when stock_index is equal to STEP, but it should be better satisfying the former condition.
				break
		if episode % 10 == 0: # for every 10 episodes, print out the current training reward and the test reward.
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
			writer.add_summary(record, global_step = episode)
			writer.flush() # Remember to add this or else you will see nothing on the tensorboard

if __name__ == '__main__':
	main()
