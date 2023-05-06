from gym.wrappers import Monitor

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import binary_crossentropy,mean_squared_error

import numpy as np

import pickle
import time

from classes import *


def self_play(network_model,config):
    """
    This is the main function to call.
    Iteratively perform self-play via Monte Carlo Tree Search (MCTS), and then train the network_model.
    Every config['self_play']['save_interval'], test the network_model and record the game rendering.

    Args:
        network_model (NetworkModel): The network model to be trained
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """

    # test initial network_model
    test(network_model,config)
    
    optimizer = Adam(learning_rate=config['train']['learning_rate'],beta_1=config['train']['beta_1'],beta_2=config['train']['beta_2'])

    replay_buffer = ReplayBuffer(config)
    start_time = time.time()
    for num_iter in range( 1 , int(config['self_play']['num_games'])+1 ):
        game = Game(config)

        # self-play
        while not game.at_terminal_state:
            action_index = mcts(game,network_model,get_temperature(num_iter),config)
            game.apply_action(action_index)
        print(f'Iteration: {num_iter}\tTotal reward: {sum(game.reward_history)}\tTime elapsed: {(time.time()-start_time)/60} minutes')

        # training
        replay_buffer.add(game)
        train(network_model,replay_buffer,optimizer,config)

        # save progress
        if (num_iter % config['self_play']['save_interval']) == 0:
            timestamp = str(time.time()).replace('.','_')
            with open(f"replay_buffers/{config['env']['env_name']}_{timestamp}.pkl",'wb') as file: pickle.dump(replay_buffer,file)
            network_model.save(f"{config['env']['env_name']}_{timestamp}")

            # test current network_model
            test(network_model,config)

def mcts(game,network_model,temperature,config): 
    """
    Perform Monte Carlo Tree Search (MCTS) on the current game state, and return an action index that indicates what action to take.

    Args:
        game (Game): The game object, containing the current state of the game
        network_model (NetworkModel): The network model will be used for inference to conduct MCTS
        temperature (float): Controls the level of exploration of MCTS (the lower the number, the greedier the action selection)
        config (dict): A dictionary specifying parameter configurations

    Returns:
        action_index (int): Represents an action in the game's action space
    """
    
    root_node = Node(0)
    root_node.expand_root_node(game.current_state,network_model)
    
    min_q_value,max_q_value = root_node.value,root_node.value # keep track of min and max mean-Q values to normalize them during selection phase
    # this is for environments that have unbounded Q-values, otherwise the prior could potentially have very little influence over selection, if Q-values are large
    for _ in range(int(config['mcts']['num_simulations'])):
        current_node = root_node

        # SELECT a leaf node
        search_path = [root_node] # node0, ... (includes the final leaf node)
        action_history = [] # action0, ...
        while current_node.is_expanded:
            # total_num_visits need to be at least 1
            # otherwise when selecting for child nodes that haven't been visited, their priors won't be taken into account, because it'll be multiplied by total_num_visits in the UCB score, which is zero
            total_num_visits = max( 1 , sum([ current_node.children[i].num_visits for i in range(len(current_node.children)) ]) )

            action_index = np.argmax([ current_node.children[i].get_ucb_score(total_num_visits,min_q_value,max_q_value,config) for i in range(len(current_node.children)) ])
            current_node = current_node.children[action_index]

            search_path.append(current_node)
            action_history.append( np.array([1 if i==action_index else 0 for i in range(config['env']['action_size'])]).reshape(1,-1) )

        # EXPAND selected leaf node
        current_node.expand_node( search_path[-2].hidden_state, action_history[-1] , network_model )

        # BACKPROPAGATE the bootstrapped value (approximated by the network_model.prediction_function) to all nodes in the search_path
        value = current_node.value
        for node in reversed(search_path):
            node.cumulative_value += value
            node.num_visits += 1

            node_q_value = node.cumulative_value / node.num_visits
            min_q_value , max_q_value = min(min_q_value,node_q_value) , max(max_q_value,node_q_value) # update min and max values

            value = node.transition_reward + config['self_play']['discount_factor'] * value # updated for parent node in next iteration of the loop

    # SAMPLE an action proportional to the visit count of the child nodes of the root node
    total_num_visits = sum([ root_node.children[i].num_visits for i in range(len(root_node.children)) ])
    policy = np.array( [ root_node.children[i].num_visits/total_num_visits for i in range(len(root_node.children)) ] )

    if temperature == None: # take the greedy action (to be used during test time)
        action_index = np.argmax(policy)
    else: # otherwise sample (to be used during training)
        policy = (policy**(1/temperature)) / (policy**(1/temperature)).sum()
        action_index = np.random.choice( range(network_model.action_size) , p=policy )

    # update Game search statistics
    game.value_history.append( root_node.cumulative_value/root_node.num_visits ) # use the root node's MCTS value as the ground truth value when training
    game.policy_history.append(policy.reshape(1,-1)) # use the MCTS policy as the ground truth value when training

    return action_index

def train(network_model,replay_buffer,optimizer,config):
    """
    Train the network_model by sampling games from the replay_buffer.

    Args:
        network_model (NetworkModel): The network model will be used for inference to conduct MCTS
        replay_buffer (ReplayBuffer): Contains a history of the most recent games of self-play
        optimizer (tensorflow.python.keras.optimizer_v2.adam.Adam): The optimizer used to update the network_model weights
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """
    
    # for every game in sample batch, unroll and update network_model weights for config['train']['num_unroll_steps'] time steps
    with tf.GradientTape() as tape:

        loss = 0
        for game in replay_buffer.sample():

            game_length = len(game.reward_history)
            sampled_index = np.random.choice( range(game_length) ) # sample an index position from the length of reward_history

            hidden_state = network_model.representation_function(game.state_history[sampled_index])
            # first we get the hidden state representation using the representation function
            # then we iteratively feed the hidden state into the dynamics function with the corresponding action, as well as feed the hidden state into the prediction function
            # we then match these predicted values to the true values
            # note we don't call the prediction function on the initial hidden state representation given by the representation function, since there's no associating predicted transition reward to match the true transition reward
            # this is because we don't / shouldn't have access to the previous action that lead to the initial state

            if (sampled_index+config['train']['num_unroll_steps']) < game_length: num_unroll_steps = int(config['train']['num_unroll_steps'])
            else: num_unroll_steps = game_length-1-sampled_index
            
            for start_index in range( sampled_index , sampled_index+num_unroll_steps ):
                # can only be unrolled up to the second-last time step, since every time step (start_index), we are predicting and matching values that are one time step into the future (start_index+1)

                ### get predictions ###
                hidden_state,pred_reward = network_model.dynamics_function([ hidden_state , game.action_history[start_index] ])
                pred_policy,pred_value = network_model.prediction_function(hidden_state)
                # the new hidden_state outputted by the dynamics function is at time step (start_index+1)
                # pred_reward is the transition reward outputted by the dynamics function by taking action_(start_index) at state_(start_index)
                # pred_policy and pred_value are the predicted values of state_(start_index+1), using the prediction function
                # therefore, pred_reward, pred_policy and pred_value are all at time step (start_index+1)

                ### make targets ###
                if (game_length - start_index - 1) >= config['train']['num_bootstrap_timesteps']: # bootstrap using transition rewards and mcts value for final bootstrapped time step
                    true_value = sum([ game.reward_history[i] * ( config['self_play']['discount_factor']**(i-start_index) ) for i in range( start_index, int( start_index+config['train']['num_bootstrap_timesteps'] ) ) ]) + \
                                 game.value_history[ start_index + int(config['train']['num_bootstrap_timesteps']) ] * ( config['self_play']['discount_factor']**(config['train']['num_bootstrap_timesteps']) )
                    # using game.reward_history[start_index] actually refers to reward_(start_index+1), since game.reward_history is shifted by 1 time step forward
                    # if the last reward we use is at game.reward_history[end_index], then the value we use to bootstrap is game.value_history[end_index+1]
                    # but since game.reward_history is shifted, we end up actually using reward_(end_index+1) and value_(end_index+1)
                    # this means we get the transition reward going into state_(end_index+1) and the bootstrapped value at state_(end_index+1)
                    # therefore the variable true_value is at time step (start_index+1)
                else: # don't bootstrap; use only transition rewards until termination
                    true_value = sum([ game.reward_history[i] * ( config['self_play']['discount_factor']**(i-start_index) ) for i in range(start_index,game_length) ])

                true_reward = game.reward_history[start_index] # since game.reward_history is shifted, this transition reward is actually at time step (start_index+1)
                true_policy = game.policy_history[start_index+1] # we need to match the pred_policy at time step (start_index+1) so we need to actually index game.policy_history at (start_index+1)

                ### calculate loss ###
                loss += (1/num_unroll_steps) * ( mean_squared_error(true_reward,pred_reward) + mean_squared_error(true_value,pred_value) + binary_crossentropy(true_policy,pred_policy) ) # take the average loss among all unroll steps
        loss += tf.reduce_sum(network_model.representation_function.losses) + tf.reduce_sum(network_model.dynamics_function.losses) + tf.reduce_sum(network_model.prediction_function.losses) # regularization loss

    ### update network_model weights ###
    grads = tape.gradient( loss, [ network_model.representation_function.trainable_variables, network_model.dynamics_function.trainable_variables, network_model.prediction_function.trainable_variables ] )
    optimizer.apply_gradients( zip( grads[0], network_model.representation_function.trainable_variables ) )
    optimizer.apply_gradients( zip( grads[1], network_model.dynamics_function.trainable_variables ) )
    optimizer.apply_gradients( zip( grads[2], network_model.prediction_function.trainable_variables ) )

def get_temperature(num_iter):
    """
    This function regulates exploration vs exploitation when selecting actions during self-play.
    Given the current interation number of the learning algorithm, return the temperature value to be used by MCTS. 

    Args:
        num_iter (int): The number of iterations that have passed for the learning algorithm

    Returns:
        temperature (float): Controls the level of exploration of MCTS (the lower the number, the greedier the action selection)
    """
    
    # as num_iter increases, temperature decreases, and actions become greedier
    if num_iter < 100: return 3
    elif num_iter < 200: return 2
    elif num_iter < 300: return 1
    elif num_iter < 400: return .5
    elif num_iter < 500: return .25
    elif num_iter < 600: return .125
    else: return .0625

def test(network_model,config):
    """
    Using a trained network_model, greedily play config['test']['num_test_games'] games, and return a list of the game histories.
    If config['test']['record'] is True, record the game renderings.

    Args:
        network_model (NetworkModel): The network model will be used for inference to conduct MCTS
        config (dict): A dictionary specifying parameter configurations

    Returns:
        game_list (list[Game]): The list of games that were played by the network_model
    """

    print('\n=========== TESTING ===========')

    game_list = []    
    for _ in range( int(config['test']['num_test_games']) ):
        game = Game(config)

        if config['test']['record']: # we need to wrap the game.env in a Monitor, so reset the seed and initial current_state after
            game.env = Monitor( gym.make(config['env']['env_name']) , f"recordings/{config['env']['env_name']}_{str(time.time()).replace('.','_')}" )
            game.env.seed( int( np.random.choice( range(int(1e5)) ) ) )
            game.current_state = game.env.reset()

        while not game.at_terminal_state:
            if config['test']['record']: game.env.render()
            
            action_index = mcts(game,network_model,None,config) # set temperature value to None, so MCTS always returns the greedy action
            game.apply_action(action_index)

        print(f'Total reward: {sum(game.reward_history)}')

        game_list.append(game)
    print()
    return game_list

if __name__ == '__main__':
    # dictionary defining gym environment attributes
    env_attributes = { 'cartpole': { 'env_name': 'CartPole-v1',
                                     'state_shape': (4,),
                                     'action_size': 2 }
                       }

    env_key_name = 'cartpole'
    config = { 'env': { 'env_name': env_attributes[env_key_name]['env_name'], # this string gets passed on to the gym.make() function to make the gym environment
                        'state_shape': env_attributes[env_key_name]['state_shape'], # used to define input shape for representation function
                        'action_size': env_attributes[env_key_name]['action_size'] }, # used to define output size for prediction function
               'model': { 'representation_function': { 'num_layers': 2, # number of hidden layers
                                                       'num_neurons': 256, # number of hidden units per layer
                                                       'activation_function': 'relu', # activation function for every hidden layer
                                                       'regularizer': L2(1e-3) }, # regularizer for every layer
                          'dynamics_function': { 'num_layers': 2,
                                                 'num_neurons': 256,
                                                 'activation_function': 'relu',
                                                 'regularizer': L2(1e-3) },
                          'prediction_function': { 'num_layers': 2,
                                                   'num_neurons': 256,
                                                   'activation_function': 'relu',
                                                   'regularizer': L2(1e-3) },
                          'hidden_state_size': 256 }, # size of hidden state representation
               'mcts': { 'num_simulations': 1e2, # number of simulations to conduct, every time we call MCTS
                         'c1': 1.25, # for regulating MCTS search exploration (higher value = more emphasis on prior value and visit count)
                         'c2': 19625 }, # for regulating MCTS search exploration (higher value = lower emphasis on prior value and visit count)
               'self_play': { 'num_games': 700, # number of games the agent plays to train on
                              'discount_factor': 1.0, # used when backpropagating values up mcts, and when calculating bootstrapped value during training
                              'save_interval': 100 }, # how often to save network_model weights and replay_buffer
               'replay_buffer': { 'buffer_size': 1e3, # size of the buffer
                                  'sample_size': 1e2 }, # how many games we sample from the buffer when training the agent
               'train': { 'num_bootstrap_timesteps': 500, # number of timesteps in the future to bootstrap true value
                          'num_unroll_steps': 1e1, # number of timesteps to unroll to match action trajectories for each game sample
                          'learning_rate': 1e-3, # learning rate for Adam optimizer
                          'beta_1': 0.9, # parameter for Adam optimizer
                          'beta_2': 0.999 }, # parameter for Adam optimizer
               'test': { 'num_test_games': 10, # number of times to test the agent using greedy actions
                         'record': False }, # True if you want to record the game renders, False otherwise
               'seed': 0
               }

    tf.random.set_seed(config['seed'])
    np.random.seed(config['seed'])

    with tf.device('/CPU:0'):
        network_model = NetworkModel(config)
        self_play(network_model,config)

    game_list = test(network_model,config)
    
