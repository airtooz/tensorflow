from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from yahoo_finance import Share
import datetime
import matplotlib.finance as finance
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SAVE = False # True if you want to renew figure data, false otherwise
EPISODE = 10000 # Total episodes to run
STEP = 300 # Step limitation in an episode
TEST = 10 # Test episode

GAMMA = 0.9 # discount factor
BUFFER_SIZE = 10000 # the replay buffer size
BATCH_SIZE = 32 # minibatch size
INITIAL_EPSILON = 0.8 # initial large, in order to explore more
FINAL_EPSILON = 0.1 # explore less, greedy most time
DAY_LENGTH = 10 # the total days for a training data, also the dim of features
FEATURE_NUM = 8 # Currently: close price, volume, K, D, RSI9, MA5, MA20, MA5-MA20
UPDATE_FREQUENCY = 100 # target freezing, weights update frequency
START = "2011-01-01" # Data start date
TRAIN_END = "2015-12-31" # Training data end date
_ID = 2330 # By default, TSMC (2330)

stock = Share(str(_ID)+'.TW')
today = datetime.date.today()
stock_data = stock.get_historical(START, str(today))
print("Historical data since", START,": ", len(stock_data))
stock_data.reverse()


i = 0
while( i < len(stock_data)):
    if (int(stock_data[i].get('Volume')) <= 0):
        stock_data.remove(stock_data[i])
        i = -1
    i += 1

train_data = stock.get_historical(START, TRAIN_END)
train_data.reverse()

i = 0
while( i < len(train_data)):
    if (int(train_data[i].get('Volume')) <= 0):
        train_data.remove(train_data[i])
        i = -1
    i += 1

print("Remove the datas with zero volume, total data ",len(stock_data))

'''
close = []
for i in xrange(len(stock_data)):
	close.append(float(stock_data[i].get('Close')))
np.array(close)
sigma = np.std(close)
avg = np.mean(close)

normalize_close = []
for i in xrange(len(stock_data)):
	normalize_close.append((close[i]-avg)/sigma)

y_minlim = min(normalize_close)
y_maxlim = max(normalize_close)
'''




# Feed in close price
'''
close = np.zeros((len(stock_data)-DAY_LENGTH, DAY_LENGTH), dtype=np.float)
for i in range(0, len(close)):
    for j in range(0, DAY_LENGTH):
        close[i,j] = float(stock_data[i+j].get('Close'))
print (close)
'''

# Eight features: Close price, Volume, K, D, RSI9, MA5, MA20, MA5-MA20

data = np.zeros((len(stock_data)-19,FEATURE_NUM), dtype = np.float)
util = []
for i in xrange(len(stock_data)):
	util.append(float(stock_data[i].get('Close')))
	rise = 0.
	fall = 0.
	if i >= 19:
		assert len(util) == 20
		data[i-19][0] = float(stock_data[i].get('Close'))
		data[i-19][1] = float(float(stock_data[i].get('Volume'))/1000000.)
		#----RSI9----
		for j in range(len(util)-8,len(util)):
			if util[j] >= util[j-1]:
				rise += (util[j]-util[j-1])
			else:
				fall += (util[j-1]-util[j])
		if rise == 0 and fall == 0:
			data[i-19][2] = 0.5
		else:
			data[i-19][2] = rise/(rise+fall)
		#----RSI9----
		u9 = util[len(util)-9:len(util)]
		#----RSV----		
		if max(u9) == min(u9):
			RSV = 0.0
		else:
			RSV = (u9[len(u9)-1] - min(u9))/(max(u9)-min(u9))
		#----RSV----

		#----K----
		if i == 19:
			K = 0.5*0.6667 + RSV*0.3333
			data[i-19][3] = K
		else:
			K = data[i-20][3]*0.6667 + RSV*0.3333
			data[i-19][3] = K
		#----K----

		#----D----
		if i == 19:
			data[i-19][4] = 0.5*0.6667 + K*0.3333
		else:
			data[i-19][4] = data[i-20][4]*0.6667 + K*0.3333
		#----D----
		
		#----MA5----
		data[i-19][5] = sum(util[len(util)-5:len(util)])/5.0
		#----MA5----

		#----MA20----
		data[i-19][6] = sum(util)/20.0
		#----MA20----

		#---(MA5-MA20)---
		data[i-19][7] = data[i-19][5]-data[i-19][6]
		#---(MA5-MA20)---

		util.pop(0)
		assert len(util) == 19

box = np.zeros((len(data)-19,DAY_LENGTH*FEATURE_NUM), dtype = np.float)
for m in xrange(len(data)-19):
	for i in xrange(FEATURE_NUM):
		for j in xrange(DAY_LENGTH):
			box[m][i*10+j] = data[m+j][i]
print(box)

'''
# Feed in images

def save_img(data, filename):
	for i in xrange(len(data)-DAY_LENGTH+1):
		fig, ax = plt.subplots(nrows=1,ncols=1)
		fig.set_size_inches(1,1)
		ax.plot([i,i+1,i+2,i+3,i+4,i+5,i+6,i+7,i+8,i+9], [data[i], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5], data[i+6], data[i+7], data[i+8], data[i+9]])
		ax.set_ylim([y_minlim,y_maxlim])
		plt.axis('off')
		fig.savefig("/home/airchen/Documents/coding/stock/"+filename+'/'+filename+'_'+str(i)+'.png', dpi=80)
		fig.clear()
		plt.close(fig)

def get_img(file_dir):
	img = mpimg.imread(file_dir)
	return img


relative_close = []
for i in xrange(1,len(stock_data)):
	relative_close.append(float(stock_data[i].get('Close'))-float(stock_data[i-1].get('Close')))
y_minlim = min(relative_close)
y_maxlim = max(relative_close)


if SAVE:
	save_img(relative_close, "Close")

data_length = len(relative_close)-DAY_LENGTH+1
train_length = (data_length*4)//5

train_image = []
for i in xrange(train_length):
	file_dir = "/home/airchen/Documents/coding/stock/Close/Close_"+str(i)+".png"
        #temp = np.asarray(get_img(file_dir))[:,:,0]
	train_image.append(get_img(file_dir))
train_image = np.asarray(train_image,dtype=np.float)

test_image = []
for i in xrange(train_length:data_length):
	file_dir = "/home/airchen/Documents/coding/stock/Close/Close_"+str(i)+".png"
        #temp = np.asarray(get_img(file_dir))[:,:,0]
	test_image.append(get_img(file_dir))
test_image = np.asarray(test_image,dtype=np.float)

assert len(train_image) + len(test_image) == data_length
'''

	
class TWStock():
	def __init__(self, data, train_length):
		self.stock_data = data
		self.train_data = self.stock_data[0:train_length]
		self.test_data = self.stock_data[train_length:len(stock_data)]
		self.stock_index = 0
		print("Training Data: ",len(self.train_data))
		print("Testing Data: ",len(self.test_data))

	def render(self):
		return

	def train_reset(self):
		self.stock_index = 0
		return self.train_data[self.stock_index]

	def test_reset(self):
		self.stock_index = 0
		return self.test_data[self.stock_index]
		
	# 0: observe, 1: having stock, 2: no stock
	def train_step(self,action): # for training, feed training data
		self.stock_index+=1
		action_reward = self.train_data[self.stock_index][0] - self.train_data[self.stock_index-1][0]
		#action_reward = self.train_data[self.stock_index][DAY_LENGTH-1] - self.train_data[self.stock_index][DAY_LENGTH-2]
		if action == 0:
			action_reward = 0
		elif action == 2:
			action_reward = -1*action_reward
		stock_done = False
		if(self.stock_index)>= len(self.train_data)-1:
			stock_done = True
        	else:
           		stock_done = False
		return self.train_data[self.stock_index], action_reward, stock_done, 0

	def test_step(self,action): # for testing, feed testing data
		self.stock_index+=1
		action_reward = self.train_data[self.stock_index][0] - self.train_data[self.stock_index-1][0]
		#action_reward = self.test_data[self.stock_index][DAY_LENGTH-1] - self.test_data[self.stock_index][DAY_LENGTH-2]
		if action == 0:
			action_reward = 0
		elif action == 2:
			action_reward = -1*action_reward
		stock_done = False
		if(self.stock_index)>= len(self.test_data)-1:
			stock_done = True
        	else:
           		stock_done = False
		return self.test_data[self.stock_index], action_reward, stock_done, 0

class DQN():
	def __init__(self,env):
		# experience replay
		self.replay_buffer = deque()
		# initialize parameters
		self.time_step = 0
		self.epsilon = INITIAL_EPSILON
		#self.state_dim = [1,80,80,1]
		self.action_dim = 3
		'''
		self.total_updates = 0
		self.update_target = []
		self.last_target_layer = None
		self.last_policy_layer = None
		self.target_update_frequency = UPDATE_FREQUENCY
		'''
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
		'''
		# Target freezing parameters
		policy_input = None
		target_input = None
		'''
		# Use CNN
		# weights and biases
		W_conv1 = tf.Variable(tf.truncated_normal(shape = [8,5,1,8],stddev = 0.01))
		b_conv1 = tf.Variable(tf.constant(0.01,shape = [8]))
		W_conv2 = tf.Variable(tf.truncated_normal(shape = [8,2,8,4],stddev = 0.01))
		b_conv2 = tf.Variable(tf.constant(0.01,shape = [4]))
		W_fc = tf.Variable(tf.truncated_normal(shape = [320,self.action_dim],stddev = 0.01))
		#W_fc = tf.Variable(tf.truncated_normal(shape = [400,self.action_dim],stddev = 0.01))
		b_fc = tf.Variable(tf.constant(0.01,shape = [self.action_dim]))

		# Layer implementation
		self.state_input = tf.placeholder("float",[None,80])
		x = tf.reshape(self.state_input,[-1,8,10,1])
		#x = tf.reshape(self.state_input,[-1,10,1,1])
		h_conv1 = tf.nn.relu(tf.nn.conv2d(x,W_conv1,strides = [1,1,1,1],padding = 'SAME') + b_conv1)
		h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1,W_conv2,strides = [1,1,1,1],padding = 'SAME') + b_conv2)
		hidden = tf.reshape(h_conv2,[-1,320])
		#hidden = tf.reshape(h_conv2,[-1,400])
		self.Q_value = tf.matmul(hidden,W_fc) + b_fc
		

	def create_training_method(self):
		self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot key vector
		self.y_input = tf.placeholder("float",[None])
		Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

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
		'''
		for i in xrange(BATCH_SIZE):
			for j in xrange(80):
				print(state_batch[i][j])
		'''
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
	sess,merged,R,T,writer = agent.get_summ()

	for episode in xrange(EPISODE):
		state = env.train_reset() # reset() returns observation
		# start training
		for step in xrange(STEP):
			action = agent.egreedy_action(state) # e-greedy action for training
			next_state,reward,done,info = env.train_step(action)
			agent.perceive(state,action,reward,next_state,done)
			state = next_state
			if done:
				break
		if episode % 20 == 0:
			train_reward = 0.
			for i in xrange(TEST):
				state = env.train_reset()
				action0_count = 0
				action1_count = 0
				action2_count = 0
				for j in xrange(STEP):
					env.render()
					action = agent.action(state)
					if action == 0:
						action0_count += 1
					elif action == 1:
						action1_count += 1
					elif action == 2:
						action2_count += 1
					else:
						print("Never come here!!")
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
					env.render()
					action = agent.action(state) # direct action for test
					if action == 0:
						action0_count += 1
					elif action == 1:
						action1_count += 1
					elif action == 2:
						action2_count += 1
					else:
						print("Never come here!!")
					
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
			writer.flush()
			if avg_reward >= 200:
				break

if __name__ == '__main__':
	main()
