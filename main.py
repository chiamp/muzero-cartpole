from gym.wrappers import Monitor

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import binary_crossentropy,mean_squared_error

import numpy as np

import pickle
import time

from classes import *


def self_play(network_model,config): # main loop
    # facilitates self-play via MCTS, and then trains network_model
    
    optimizer = Adam(learning_rate=config['train']['learning_rate'],beta_1=config['train']['beta_1'],beta_2=config['train']['beta_2'])

    replay_buffer = ReplayBuffer(config)
##    with open('replay_buffers/Acrobot-v1_1631744464_0856915.pkl','rb') as file: replay_buffer = pickle.load(file)
    start_time = time.time()
    for num_iter in range( 1 , int(config['self_play']['num_games'])+1 ):
        game = Game(config)

        while not game.at_terminal_state:
            action_index = mcts(game,network_model,get_temperature(num_iter),config)
            game.apply_action(action_index)
        print(f'Iteration: {num_iter}\tTotal reward: {sum(game.reward_history)}\tTime elapsed: {(time.time()-start_time)/60} minutes')

        replay_buffer.add(game)
        train(network_model,replay_buffer,optimizer,config)

        if (num_iter % config['self_play']['save_interval']) == 0:
            timestamp = str(time.time()).replace('.','_')
            with open(f"replay_buffers/{config['env']['env_name']}_{timestamp}.pkl",'wb') as file: pickle.dump(replay_buffer,file)
            network_model.save(f"{config['env']['env_name']}_{timestamp}")
        
    return replay_buffer

def mcts(game,network_model,temperature,config): # Monte Carlo Tree Search
    root_node = Node(0)
    root_node.expand_root_node(game.current_state,network_model)
##    root_node.cumulative_value += root_node.value # backpropagate value to mean Q value
##    root_node.num_visits += 1

    min_q_value,max_q_value = root_node.value,root_node.value # keep track of min and max mean-Q values to normalize them during selection phase
    # this is for environments that have unbounded Q values, otherwise the prior could potentially have very little influence over selection, if Q values are large
    for _ in range(int(config['mcts']['num_simulations'])):
        current_node = root_node

        # SELECT a leaf node
        search_path = [root_node] # node0, ... (includes the final leaf node)
        action_history = [] # action0, ...
        while current_node.is_expanded:
            total_num_visits = sum([ current_node.children[i].num_visits for i in range(len(current_node.children)) ])

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
    game.value_history.append(root_node.value)
    game.policy_history.append(policy)

    return action_index

def train(network_model,replay_buffer,optimizer,config): # training loop
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
            if (sampled_index+config['train']['num_unroll_steps']) < (game_length-1): num_unroll_steps = int(config['train']['num_unroll_steps'])
            else: num_unroll_steps = game_length-1-sampled_index
##            for start_index in range( sampled_index , int( min( game_length-1 , sampled_index+config['train']['num_unroll_steps'] ) ) ):
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

def get_temperature(num_iter): # temperature function used to regulate exploration vs exploitation when selecting actions during self-play
    # as num_iter increases, temperature decreases, an actions become greedier
##    return .75
    if num_iter < 400*2: return 3
    elif num_iter < 800*2: return 2
    elif num_iter < 1200*2: return 1
    else: return .5
##    elif num_iter < 1600: return .5
##    else: return .25

def test(network_model,config,n=1): # play n games, and return a list of the game histories
    game_list = []
    for _ in range(n):
        game = Game(config)

        while not game.at_terminal_state:
            game.env.render()
            action_index = mcts(game,network_model,None,config)
            game.apply_action(action_index)

        print(f'Total reward: {sum(game.reward_history)}')

        game_list.append(game)
    return game_list

def record(network_model,config): # play one game and record a video file, saved in the assets/ directory
    game = Game(config)
    
    game.env = Monitor( gym.make(config['env']['env_name']) , f"assets/{config['env']['env_name']}_{str(time.time()).replace('.','_')}" )
    game.env.seed( int( np.random.choice( range(int(1e5)) ) ) )
    game.current_state = game.env.reset()

    while not game.at_terminal_state:
        game.env.render()
        action_index = mcts(game,network_model,None,config)
        game.apply_action(action_index)

    print(f'Total reward: {sum(game.reward_history)}')

if __name__ == '__main__':
    env_attributes = { 'cartpole': { 'env_name': 'CartPole-v1',
                                     'state_shape': (4,),
                                     'action_size': 2 },
                       'mountaincar': { 'env_name': 'MountainCar-v0',
                                        'state_shape': (2,),
                                        'action_size': 3 },
                       'acrobot': { 'env_name': 'Acrobot-v1',
                                    'state_shape': (6,),
                                    'action_size': 3 },
                       'lunarlander': { 'env_name': 'LunarLander-v2',
                                        'state_shape': (8,),
                                        'action_size': 4 }
                       }

    env_key_name = 'lunarlander' # change this value ('cartpole','mountaincar','acrobot') to train different environments
    config = { 'env': { 'env_name': env_attributes[env_key_name]['env_name'],
                        'state_shape': env_attributes[env_key_name]['state_shape'], # used to define input shape for representation function
                        'action_size': env_attributes[env_key_name]['action_size'] }, # used to define output size for prediction function
               'model': { 'representation_function': { 'num_layers': 4,#1,#4,
                                                       'num_neurons': 64,#16,#64,
                                                       'activation_function': 'relu',
                                                       'regularizer': L2(1e-3) },
                          'dynamics_function': { 'num_layers': 4,#1,#4,
                                                 'num_neurons': 64,#16,#64,
                                                 'activation_function': 'relu',
                                                 'regularizer': L2(1e-3) },
                          'prediction_function': { 'num_layers': 4,#1,#4,
                                                   'num_neurons': 64,#16,#64,
                                                   'activation_function': 'relu',
                                                   'regularizer': L2(1e-3) },
                          'hidden_state_size': 32 }, #8 # size of hidden state representation
               'mcts': { 'num_simulations': 1e2,
                         'c1': 1.25, #1e-1 # for regulating mcts search exploration (higher value = more emphasis on prior value and visit count)
                         'c2': 19625 }, #1e6 # for regulating mcts search exploration (higher value = lower emphasis on prior value and visit count)
               'self_play': { 'num_games': 1e10,
                              'save_interval': 100,
                              'discount_factor': 1.0 }, # used when backpropagating values up mcts, and when calculating bootstrapped value during training
               'replay_buffer': { 'buffer_size': 1e3,
                                  'sample_size': 1e2 }, #1e1
               'train': { 'num_bootstrap_timesteps': 1000, # number of timesteps in the future to bootstrap true value
                          'num_unroll_steps': 1e1, #1 # number of timesteps to unroll to match action trajectories for each game sample
                          'learning_rate': 1e-3, #1e-2
                          'beta_1': 0.9,
                          'beta_2': 0.999 },
               'seed': 1
               }

    tf.random.set_seed(config['seed'])
    np.random.seed(config['seed'])

    with tf.device('/CPU:0'):
        network_model = NetworkModel(config)
####        network_model.load('Acrobot-v1_1631581594_7407389')
##        network_model.load('Acrobot-v1_1631744464_0856915')
##        network_model.load('Acrobot-v1_1632058191_2013526')
        replay_buffer = self_play(network_model,config)

##    network_model = NetworkModel(config)
####    network_model.load('CartPole-v1_1631394556_759742')
##    network_model.load('CartPole-v1_1632015579_2017398')
####    network_model.load('MountainCar-v0_1631738788_459358')
####    g = Game(config)
##    
##    a,r = mcts(g,network_model,3,config)
##    print(r.value,*[r.children[i].num_visits for i in range(config['env']['action_size'])])
##    print(*[r.children[i].children[j].num_visits for i in range(config['env']['action_size']) \
##            for j in range(config['env']['action_size'])])

##    game_list = test(network_model,config,n=1)
##    record(network_model,config)
    
    
    
