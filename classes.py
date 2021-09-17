import gym

from tensorflow.keras.models import Model,save_model,load_model
from tensorflow.keras.layers import Input,Dense,Concatenate

import numpy as np

import os


class Game: # wrapper for gym env
    def __init__(self,config):
        self.env = gym.make(config['env']['env_name'])
        self.env.seed( int( np.random.choice( range(int(1e5)) ) ) ) # since we set a seed for numpy, we can get reproducible results by setting a seed for the gym env, where the seed number is generated from numpy
        self.action_size = config['env']['action_size']
        
        self.current_state = self.env.reset()
        self.at_terminal_state = False

        self.state_history = [self.current_state.reshape(1,-1)] # starts at t = 0
        self.action_history = [] # starts at t = 0
        self.reward_history = [] # starts at t = 1 (the transition reward for reaching state 1)

        self.value_history = [] # starts at t = 0 (from MCTS)
        self.policy_history = [] # starts at t = 0 (from MCTS)
    def apply_action(self,action_index): # applies action corresponding to action_index to self.env, and records the action, resulting state, and resulting transition reward
        # this method should never be called when self.at_terminal_state is True
        
        obs,reward,done,info = self.env.step(action_index)

        self.current_state = obs
        self.state_history.append(self.current_state.reshape(1,-1))
        self.action_history.append( np.array([1 if i==action_index else 0 for i in range(self.action_size)]).reshape(1,-1) )
        self.reward_history.append(reward)
        self.at_terminal_state = done

        if self.at_terminal_state: self.env.close()

class NetworkModel: # neural network model
    def __init__(self,config):

        # building representation function layers
        obs_input_layer = Input(config['env']['state_shape'])
        hidden_layer = Dense(config['model']['representation_function']['num_neurons'],activation=config['model']['representation_function']['activation_function'],bias_initializer='glorot_uniform',
                             kernel_regularizer=config['model']['representation_function']['regularizer'],bias_regularizer=config['model']['representation_function']['regularizer'])(obs_input_layer)
        for _ in range(config['model']['representation_function']['num_layers']):
            hidden_layer = Dense(config['model']['representation_function']['num_neurons'],activation=config['model']['representation_function']['activation_function'],bias_initializer='glorot_uniform',
                                 kernel_regularizer=config['model']['representation_function']['regularizer'],bias_regularizer=config['model']['representation_function']['regularizer'])(hidden_layer)
        hidden_state_output_layer = Dense(config['model']['hidden_state_size'],activation=config['model']['representation_function']['activation_function'],bias_initializer='glorot_uniform',
                                          kernel_regularizer=config['model']['representation_function']['regularizer'],bias_regularizer=config['model']['representation_function']['regularizer'])(hidden_layer)
        
        self.representation_function = Model(obs_input_layer,hidden_state_output_layer)

        # building dynamics function layers
        hidden_state_input_layer = Input(config['model']['hidden_state_size'])
        action_input_layer = Input(config['env']['action_size'])
        concat_layer = Concatenate()([hidden_state_input_layer,action_input_layer])
        hidden_layer = Dense(config['model']['dynamics_function']['num_neurons'],activation=config['model']['dynamics_function']['activation_function'],bias_initializer='glorot_uniform',
                             kernel_regularizer=config['model']['dynamics_function']['regularizer'],bias_regularizer=config['model']['dynamics_function']['regularizer'])(concat_layer)
        for _ in range(config['model']['dynamics_function']['num_layers']):
            hidden_layer = Dense(config['model']['dynamics_function']['num_neurons'],activation=config['model']['dynamics_function']['activation_function'],bias_initializer='glorot_uniform',
                                 kernel_regularizer=config['model']['dynamics_function']['regularizer'],bias_regularizer=config['model']['dynamics_function']['regularizer'])(hidden_layer)
        hidden_state_output_layer = Dense(config['model']['hidden_state_size'],activation=config['model']['dynamics_function']['activation_function'],bias_initializer='glorot_uniform',
                                          kernel_regularizer=config['model']['dynamics_function']['regularizer'],bias_regularizer=config['model']['dynamics_function']['regularizer'])(hidden_layer)
        transition_reward_output_layer = Dense(1,activation='linear',bias_initializer='glorot_uniform',
                                               kernel_regularizer=config['model']['dynamics_function']['regularizer'],bias_regularizer=config['model']['dynamics_function']['regularizer'])(hidden_layer)
        
        self.dynamics_function = Model([hidden_state_input_layer,action_input_layer],[hidden_state_output_layer,transition_reward_output_layer])

        # building prediction function layers
        hidden_state_input_layer = Input(config['model']['hidden_state_size'])
        hidden_layer = Dense(config['model']['prediction_function']['num_neurons'],activation=config['model']['prediction_function']['activation_function'],bias_initializer='glorot_uniform',
                             kernel_regularizer=config['model']['prediction_function']['regularizer'],bias_regularizer=config['model']['prediction_function']['regularizer'])(hidden_state_input_layer)
        for _ in range(config['model']['prediction_function']['num_layers']):
            hidden_layer = Dense(config['model']['prediction_function']['num_neurons'],activation=config['model']['prediction_function']['activation_function'],bias_initializer='glorot_uniform',
                                 kernel_regularizer=config['model']['prediction_function']['regularizer'],bias_regularizer=config['model']['prediction_function']['regularizer'])(hidden_layer)
        policy_output_layer = Dense(config['env']['action_size'],activation='softmax',bias_initializer='glorot_uniform',
                                    kernel_regularizer=config['model']['prediction_function']['regularizer'],bias_regularizer=config['model']['prediction_function']['regularizer'])(hidden_layer)
        value_output_layer = Dense(1,activation='linear',bias_initializer='glorot_uniform',
                                   kernel_regularizer=config['model']['prediction_function']['regularizer'],bias_regularizer=config['model']['prediction_function']['regularizer'])(hidden_layer)
        
        self.prediction_function = Model(hidden_state_input_layer,[policy_output_layer,value_output_layer])

        self.action_size = config['env']['action_size']
    def save(self,model_name): # save the model weights
        os.mkdir(f'models/{model_name}')
        self.representation_function.save_weights(f'models/{model_name}/representation_function_weights.h5')
        self.dynamics_function.save_weights(f'models/{model_name}/dynamics_function_weights.h5')
        self.prediction_function.save_weights(f'models/{model_name}/prediction_function_weights.h5')
    def load(self,model_name): # load the model weights
        self.representation_function.load_weights(f'models/{model_name}/representation_function_weights.h5')
        self.dynamics_function.load_weights(f'models/{model_name}/dynamics_function_weights.h5')
        self.prediction_function.load_weights(f'models/{model_name}/prediction_function_weights.h5')

class Node: # MCTS node
    def __init__(self,prior):
        self.prior = prior # prior probability given by the output of the prediction function of the parent node
        
        self.hidden_state = None # from dynamics function
        self.transition_reward = 0 # from dynamics function
        self.policy = None # from prediction function
        self.value = None # from prediction function

        self.is_expanded = False
        self.children = []

        self.cumulative_value = 0 # divide this by self.num_visits to get the mean action-value of this node
        self.num_visits = 0
    def expand_node(self,parent_hidden_state,parent_action,network_model):
        # use the dynamics function to get this node's hidden state representation and transition reward
        # use the prediction function to get this node's value and the child node's prior probability
        
        hidden_state,transition_reward = network_model.dynamics_function( [ parent_hidden_state , parent_action ] )
        self.hidden_state = hidden_state
        self.transition_reward = transition_reward.numpy()[0][0] # convert to scalar

        policy,value = network_model.prediction_function( self.hidden_state )
        self.policy = policy
        self.value = value.numpy()[0][0] # convert to scalar
        
        for action in range(network_model.action_size):
            self.children.append( Node(self.policy.numpy()[0][action]) )
        self.is_expanded = True
    def expand_root_node(self,current_state,network_model): # to be called only on the root node
        # same as self.expand_node() method, except representation function is used to get this node's hidden state
        # therefore there's no corresponding predicted transition reward for the root node
        
        hidden_state = network_model.representation_function(current_state.reshape(1,-1))
        self.hidden_state = hidden_state
        self.transition_reward = 0 # no transition reward for the root node

        policy,value = network_model.prediction_function( self.hidden_state )
        self.policy = policy
        self.value = value.numpy()[0][0] # convert to scalar

        for action in range(network_model.action_size):
            self.children.append( Node(self.policy.numpy()[0][action]) )
        self.is_expanded = True
    def get_ucb_score(self,visit_sum,min_q_value,max_q_value,config): # visit_sum is the sum of all visits across all children nodes (for this node's parent)
        normalized_q_value = self.transition_reward + config['self_play']['discount_factor'] * self.cumulative_value / max(self.num_visits,1)
        if min_q_value != max_q_value: normalized_q_value = (normalized_q_value - min_q_value) / (max_q_value - min_q_value) # min-max normalize q-value, to make sure q-value is in the interval [0,1]
        # if min and max value are equal, we would end up dividing by 0

        return normalized_q_value + \
               self.prior * np.sqrt(visit_sum) / (1 + self.num_visits) * \
               ( config['mcts']['c1'] + np.log( (visit_sum + config['mcts']['c2'] + 1) / config['mcts']['c2'] ) )
        
class ReplayBuffer: # stores recently played games
    def __init__(self,config):
        self.buffer = [] # list of Game objects, that contain the state, action, reward, MCTS policy, and MCTS value history
        self.buffer_size = int(config['replay_buffer']['buffer_size'])
        self.sample_size = int(config['replay_buffer']['sample_size'])
    def add(self,game): # add a new entry to self.buffer, and remove the oldest entry if the buffer size is at its max, specified by the config parameter
        if len(self.buffer) >= self.buffer_size: self.buffer.pop(0)
        self.buffer.append(game)
    def sample(self): # sample a number of games from self.buffer, specified by the config parameter
        if len(self.buffer) <= self.sample_size: return self.buffer.copy()
        return np.random.choice(self.buffer,size=self.sample_size,replace=False).tolist()

