import gym

from tensorflow.keras.models import Model,save_model,load_model
from tensorflow.keras.layers import Input,Dense,Concatenate

import numpy as np

import os


class Game: # wrapper for gym env
    """
    A wrapper class for gym environments. Keeps track of the history of the game.
    """
    
    def __init__(self,config):
        """
        Constructor method for the Game class.
        
        Args:
            config (dict): A dictionary specifying parameter configurations
            
        Attributes:
            env (gym.wrappers.time_limit.TimeLimit): The gym environment
            action_size (int): The number of possible actions in env's action space

            current_state (numpy.ndarray): The numpy array representation of env's current state
            at_terminal_state (bool): True if env is at a terminal state, False otherwise

            state_history (list[numpy.ndarray]): Contains a list of past states visited during this game
            action_history (list[numpy.ndarray]): Contains a list of past actions applied to the env (actions are one-hot encoded into a vector)
            reward_history (list[float]): Contains a list of past transition rewards received from the env

            value_history (list[float]): Contains a list of predicted values for each corresponding past state, outputted by MCTS
            policy_history (list[numpy.ndarray]): Contains a list of action distributions for each corresponding past state, outputted by MCTS
        """
        
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
    def apply_action(self,action_index):
        """
        Apply the action_index to the game env. Record the resulting state, action and transition reward.
        Update whether the resulting state is terminal or not.
        
        Args:
            action_index (int): Represents an action in the game's action space

        Returns: None
        """
        
        # this method should never be called when self.at_terminal_state is True
        
        obs,reward,done,info = self.env.step(action_index)

        self.current_state = obs
        self.state_history.append(self.current_state.reshape(1,-1))
        self.action_history.append( np.array([1 if i==action_index else 0 for i in range(self.action_size)]).reshape(1,-1) )
        self.reward_history.append(reward)
        self.at_terminal_state = done

        if self.at_terminal_state: self.env.close()

class NetworkModel: # neural network model
    """
    A class that contains the representation, dynamics and prediction network.
    These networks are trained during agent self-play.
    """
    
    def __init__(self,config):
        """
        Constructor method for the NetworkModel class.
        
        Args:
            config (dict): A dictionary specifying parameter configurations
            
        Attributes:
            representation_function (tensorflow.python.keras.engine.functional.Functional): A neural network that takes in the game state as input, and outputs a hidden state representation
            dynamics_function (tensorflow.python.keras.engine.functional.Functional): A neural network that takes in a hidden state representation and action vector as input, and outputs the resulting hidden state representation and predicted transition reward
            prediction_function (tensorflow.python.keras.engine.functional.Functional): A neural network that takes in a hidden state representation as input, and outputs a predicted value and policy

            action_size (int): The number of possible actions in the game env's action space
        """

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
    def save(self,model_name):
        """
        Save the weights of the representation_function, dynamics_function and prediction_function.
        
        Args:
            model_name (str): The filename to use when creating weight files.

        Returns: None
        """
        
        os.mkdir(f'models/{model_name}')
        self.representation_function.save_weights(f'models/{model_name}/representation_function_weights.h5')
        self.dynamics_function.save_weights(f'models/{model_name}/dynamics_function_weights.h5')
        self.prediction_function.save_weights(f'models/{model_name}/prediction_function_weights.h5')
    def load(self,model_name):
        """
        Load the weights of the representation_function, dynamics_function and prediction_function into this NetworkModel.
        
        Args:
            model_name (str): The filename of the weight files to load into this NetworkModel.

        Returns: None
        """
        self.representation_function.load_weights(f'models/{model_name}/representation_function_weights.h5')
        self.dynamics_function.load_weights(f'models/{model_name}/dynamics_function_weights.h5')
        self.prediction_function.load_weights(f'models/{model_name}/prediction_function_weights.h5')

class Node:
    """
    A class that represents the nodes used in Monte Carlo Tree Search.
    """
    
    def __init__(self,prior):
        """
        Constructor method for the Node class.
        
        Args:
            prior (float): The prior probability assigned to this Node upon instantiation, obtained from the prediction function of the NetworkModel (this value is supposed to represent how promising this Node is, before exploration)
            
        Attributes:
            prior (float): The prior probability assigned to this Node upon instantiation, obtained from the argument

            hidden_state (numpy.ndarray): The hidden state representation of this Node, obtained from either the representation or dynamics function of the NetworkModel
            transition_reward (float): The predicted transition reward as a result of transitioning to this Node from the parent state, obtained from the dynamics function of the NetworkModel
            policy (numpy.ndarray): The predicted action distribution, obtained from applying the prediction function to this Node's hidden_state (the values will serve as priors for the children of this Node)
            value (float): The predicted value of this Node, obtained from applying the prediction function to this Node's hidden_state

            is_expanded (bool): True if this Node is expanded, False otherwise
            children (list[Node]): Contains a list of this Node's children

            cumulative_value (float): Every simulation of MCTS that contains this Node in its search path, the leaf node's predicted value is backpropagated up and accumulated in this cumulative_value
            num_visits (int): The number of simulations of MCTS that contained this Node in its search path (this value is used to divide the cumulative_value to get the mean Q-value)
        """
        
        self.prior = prior # prior probability given by the output of the prediction function of the parent node
        
        self.hidden_state = None # from dynamics function
        self.transition_reward = 0 # from dynamics function
        self.policy = None # from prediction function
        self.value = None # from prediction function

        self.is_expanded = False
        self.children = []

        self.cumulative_value = 0 # divide this by self.num_visits to get the mean Q-value of this node
        self.num_visits = 0
    def expand_node(self,parent_hidden_state,parent_action,network_model):
        """
        Expand this Node. Use the dynamics function on parent_hidden_state and parent_action to get this Node's hidden state representation and transition reward.
        Then use the prediction function on this Node's hidden state representation to get this Node's value and the prior probabilities of this Node's children.
        
        Args:
            parent_hidden_state (numpy.ndarray): The hidden state representation of this Node's parent
            parent_action (numpy.ndarray): The corresponding action vector taken from this Node's parent to get to this Node
            network_model (NetworkModel): The NetworkModel's dynamics function and prediction function will be used to expand this Node

        Returns: None
        """

        # get hidden state representation and transition reward
        hidden_state,transition_reward = network_model.dynamics_function( [ parent_hidden_state , parent_action ] )
        self.hidden_state = hidden_state
        self.transition_reward = transition_reward.numpy()[0][0] # convert to scalar

        # get predicted policy and value
        policy,value = network_model.prediction_function( self.hidden_state )
        self.policy = policy
        self.value = value.numpy()[0][0] # convert to scalar

        # instantiate child Node's with prior values, obtained from the predicted policy
        for action in range(network_model.action_size):
            self.children.append( Node(self.policy.numpy()[0][action]) )
            
        self.is_expanded = True
    def expand_root_node(self,current_state,network_model):
        """
        Expand this Node. This function should only be called on root nodes, and assumes that this Node is a root node.
        Use the representation function on the game env's current_state to get this Node's hidden state representation.
        Then use the prediction function on this Node's hidden state representation to get this Node's value and the prior probabilities of this Node's children.
        
        Args:
            current_state (numpy.ndarray): The numpy array representation of the game env's current state
            network_model (NetworkModel): The NetworkModel's representation function and prediction function will be used to expand this Node

        Returns: None
        """
        
        # same as self.expand_node() method, except representation function is used to get this node's hidden state
        # therefore there's no corresponding predicted transition reward for the root node

        # get hidden state representation
        hidden_state = network_model.representation_function(current_state.reshape(1,-1))
        self.hidden_state = hidden_state
        self.transition_reward = 0 # no transition reward for the root node

        # get predicted policy and value
        policy,value = network_model.prediction_function( self.hidden_state )
        self.policy = policy
        self.value = value.numpy()[0][0] # convert to scalar

        # instantiate child Node's with prior values, obtained from the predicted policy
        for action in range(network_model.action_size):
            self.children.append( Node(self.policy.numpy()[0][action]) )
            
        self.is_expanded = True
    def get_ucb_score(self,visit_sum,min_q_value,max_q_value,config):
        """
        Calculate the modified UCB score of this Node. This value will be used when selecting Nodes during MCTS simulations.
        The UCB score balances between exploiting Nodes with known promising values, and exploring Nodes that haven't been searched much throughout the MCTS simulations.
        
        Args:
            visit_sum (int): The total number of visits across all child Node's of this Node's parent
            min_q_value (float): The minimum Q-value found across all Nodes selected by MCTS across all simulations (used for min-max normalization)
            max_q_value (float): The maximum Q-value found across all Nodes selected by MCTS across all simulations (used for min-max normalization)
            config (dict): A dictionary specifying parameter configurations

        Returns:
            ucb_score (float): This value is calculated across all child Nodes and MCTS selects the child Node with the highest UCB score to add to the search path
        """
        
        normalized_q_value = self.transition_reward + config['self_play']['discount_factor'] * self.cumulative_value / max(self.num_visits,1)
        if min_q_value != max_q_value: normalized_q_value = (normalized_q_value - min_q_value) / (max_q_value - min_q_value) # min-max normalize q-value, to make sure q-value is in the interval [0,1]
        # if min and max value are equal, we would end up dividing by 0

        return normalized_q_value + \
               self.prior * np.sqrt(visit_sum) / (1 + self.num_visits) * \
               ( config['mcts']['c1'] + np.log( (visit_sum + config['mcts']['c2'] + 1) / config['mcts']['c2'] ) )
        
class ReplayBuffer:
    """
    A class that stores the history of the most recent games of self-play.
    """
    
    def __init__(self,config):
        """
        Constructor method for the ReplayBuffer class.
        
        Args:
            config (dict): A dictionary specifying parameter configurations

        Attributes:
            buffer (list[Game]): Buffer that stores Game objects.
            buffer_size (int): Indicates the maximum size of the buffer
            sample_size (int): Indicates how many Games to sample from the buffer when we call the sample() method
        """
        
        self.buffer = [] # list of Game objects, that contain the state, action, reward, MCTS policy, and MCTS value history
        self.buffer_size = int(config['replay_buffer']['buffer_size'])
        self.sample_size = int(config['replay_buffer']['sample_size'])
    def add(self,game):
        """
        Add the game to the ReplayBuffer. Remove the oldest Game entry if the size of the buffer exceeds buffer_size (which is set by the config parameter upon instantiation)
        
        Args:
            game (Game): The Game to add to the ReplayBuffer

        Returns: None
        """
        
        if len(self.buffer) >= self.buffer_size: self.buffer.pop(0)
        self.buffer.append(game)
    def sample(self): # sample a number of games from self.buffer, specified by the config parameter
        """
        Sample a number of Games from the buffer equal to sample_size (which is set by the config parameter upon instantiation).
        
        Args: None

        Returns:
            game_samples (list[Game]): A list of sampled Games to be used to train the NetworkModel weights
        """
        
        if len(self.buffer) <= self.sample_size: return self.buffer.copy()
        return np.random.choice(self.buffer,size=self.sample_size,replace=False).tolist()

