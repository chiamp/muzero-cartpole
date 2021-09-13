
# MuZero and gym
This is a repo where I apply the MuZero reinforcement learning algorithm on the [gym environments](https://gym.openai.com/envs/) provided by OpenAI.

## Table of Contents
* [What is MuZero?](#what-is-muzero?)
* [Thoughts](#thoughts)
* [MuZero Technical Details](#muzero-technical-details)
* [Motivation](#motivation)
* [Neural Network Structure](#neural-network-structure)
* [Training Details](#training-details)
* [Extraction Algorithm](#extraction-algorithm)
* [Requirements](#requirements)
* [Commentary](#commentary)
* [Additional Reading](#additional-reading)

## What is MuZero?
Before we talk about MuZero, we have to mention and give context to its predecessors: AlphaGo, AlphaGo Zero and AlphaZero.

#### AlphaGo (2016)
[AlphaGo](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) is a reinforcement learning algorithm created by DeepMind that managed to beat the world champion at Go in 2016 ([link to documentary](https://www.youtube.com/watch?v=WXuK6gekU1Y)). It was considered a huge feat in artificial intelligence at the time, as although AI algorithms were able to beat the best chess players (see [Deep Blue vs Kasparov](https://en.wikipedia.org/wiki/Deep_Blue_versus_Garry_Kasparov)), there did not exist an AI algorithm that was capable of beating professional Go players. This is due to the fact that Go has a significantly larger amount of possible board positions (10<sup>170</sup>) compared to chess (10<sup>43</sup>), and so the same algorithms that were used to create superhuman AI algorithms for chess, could not be used for Go. 

In addition, Deep Blue benefitted from game-specific knowledge; i.e. specific board positions and pattern structures were hard-coded into its evaluation function, which it uses to evaluate which board positions are favorable for itself (as seen on page 75 of the [Deep Blue paper](https://www.sciencedirect.com/science/article/pii/S0004370201001291)):
![Alt text](assets/deep_blue_evaluation_features.png)
This knowledge specifically about the game of chess, was provided by chess grandmasters, who worked together with IBM to create Deep Blue.

With Go, however, such a hand-crafted, hard-coded evaluation function was difficult to create due to the nature of the game, and coupled with the fact that the number of possible board positions was much greater than chess, made it much harder to develop an AI algorithm for.

AlphaGo was able to solve this problem via a combination of reinforcement learning and deep neural networks. AlphaGo employs a search algorithm called Monte Carlo Tree Search ([MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)) to find the optimal move to play. It uses a policy neural network to limit the search breadth (the amount of possible moves to consider) and uses a value neural network to limit the search depth (the amount of turns into the future to consider), so that using MCTS in Go becomes tractable.

The neural networks in AlphaGo were first trained on pre-existing human games, and then later through self-play. Through this method, AlphaGo was able to make history and beat the world champion in Go.

#### AlphaGo Zero (2017)
[AlphaGo Zero](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ) employs a similar algorithm as AlphaGo, but rather than first training on human games, it learns entirely from scratch, through self-play. That is to say, AlphaGo Zero never sees human Go games, and so its knowledge of Go is based entirely on the games it plays against itself.

Through this method of training, AlphaGo Zero was able to defeat the previous AlphaGo 100 games to 0.

#### AlphaZero (2018)
[AlphaZero](https://arxiv.org/pdf/1712.01815.pdf) is a more generalized version of the AlphaGo Zero algorithm; not only can it be applied to Go, but also to chess and shogi. It was able to outperform the chess computer program Stockfish and the shogi computer program Elmo, both of which have won world championship computer tournaments in the past.

#### MuZero (2020)
[MuZero](https://arxiv.org/pdf/1911.08265.pdf) is the latest version of the algorithm, that eliminates the pre-requisite of requiring a transition model of the environment. Search-based methods like MCTS can only be used if the algorithm has access to a transition model; essentially it needs to know the "rules" of the game (e.g. if I play a certain move, then the rules of the game will tell me what the new board state would look like), so that it can search for hypothetical moves in the future, to determine what is the optimal action to take now. Therefore the game rules were provided to AlphaGo, AlphaGo Zero and AlphaZero, so that it could facilitate this search.

In MuZero, however, the transition model is not provided. Instead, MuZero learns a transition model entirely through self-play. Using this method, MuZero was able to match and exceed AlphaZero's performance, but without the pre-requisite of requiring a transition model of the environment be provided.

## Thoughts
In our effort to make machines more intelligent, we try to encode our knowledge and expertise of the domain we're working with into the machine. Deep Blue is an example of that; it is a monumental feat of engineering and collaboration between computer scientists and chess grandmasters, working together and pooling their knowledge and experience to create a machine capable of superhuman performance in chess. 

Yet, this approach has its limits. For one, the time and resources required to figure out what domain-specific knowledge would be useful for the machine to have, and then figure out how to quantify and hard-code this knowledge into a computer system is huge. Secondly, the knowledge we encode into the machine is domain-specific, meaning that encoding a computer system with chess-specific knowledge, will make the computer only good at playing chess. If we wanted to make a machine capable of playing Go at a high level, we would need to find professional Go players and redo the whole process of figuring out what sort of Go knowledge we think is useful to put into the computer system. This process has to be repeated for every unique domain we'd like to create an AI system for, which takes a lot of time and resources. Not to mention the fact that there may be domains that such experts don't exist currently, or the nature of the domain itself makes it hard to quantify a hard-coded evaluation function.

With the advent of machine learning, we can now take a different approach to making machines more intelligent. Rather than telling a machine explicitly what to look out for, and what we consider to be good or bad, we can use neural networks to have the machine learn these values for itself through its own experiences of self-play. AlphaGo was the first step in this direction; because of the nature of the game, coming up with a hard-coded evaluation function for Go was much more difficult compared to chess. So instead, DeepMind decided to let the machine learn for itself, after studying many human games, and playing against itself over and over again, it was able to figure out what board positions in Go were considered good or bad, via its own experience. Rather than providing a hard-coded evaluation function, AlphaGo learned its own evaluation function based on its own experience, and using that, was able to beat the world champion in Go.

AlphaGo Zero removed the human element; it did not see any human Go games when training, and only learned through self-play. The fact that it was able to defeat AlphaGo 100 games to 0, perhaps hints at the idea that humans are not the pinnacle of expertise in any domain. Human knowledge of Go is deep; the game has been around for thousands of years, and the theory and strategy has been well developed and studied. Yet the computer program that was devoid of any human knowledge was able to beat the computer program that was partially trained on human games 100 games to 0. It seems that the knowledge that we humans have gathered, may not be the most optimal. And this idea is what makes me excited as to what other novel things AI algorithms can show us...what other things have we accepted as "optimal", that can actually be improved upon, based on what AI algorithms can discover?

And it's not just in Go. AlphaZero shows us that the same algorithm used to achieve superhuman performance in Go, can also be applied to different domains in chess and shogi. No longer do we have to spend extra time and resources to hard-code domain-specific knowledge into the AI's evaluation functions, when we can use a general reinforcement learning algorithm to learn the evaluation function via self-play. This is extremely exciting, especially for problems were domain-experts may not even exist, like self-driving cars. The fact that AlphaZero was able to exceed Stockfish's performance in chess, considering the fact that Stockfish is built quite similarly to Deep Blue in terms of hard-coded evaluation functions, gives further credence to the value of this type of reinforcement learning algorithm. 

The elimination of the need to provide a transition model, is another step toward the generalization of this reinforcement learning algorithm. It may seem trivial in the space of board games, where the rules and consequences of actions are well-defined. But for real-world applications, we would need a transition model that mimics the real world, in order to allow the machine to accurately learn and plan future actions accordingly. Unfortunately, the real world isn't as discrete and deterministic as the world of board games; it can be continuous and stochastic. Of course we can try to make approximations of the real-world, but they'll never be as good as the real thing. Fortunately with MuZero, the transition model itself can also be learned via self-play and experience! And with this algorithm, we're one step closer towards artificial intelligence applicable in the real-world.

Artificial intelligence has the potential to solve a lot of problems for us. The fact that it could potentially discover novel things that humans either prematurely dismissed as non-optimal, or never even considered, is exciting to me, especially in the fields of medicine, energy and finance. I think we're at an exciting period of time where AI technology is expanding at an exponential rate, and I can't wait to see what the future has in store for us!

## MuZero Technical Details

### Neural Network Structure
![Alt text](neural_network_structure.png)
A neural network with a unique structure is used to fit to the data:
* First the inputs (denoted X1 and X2 in the above diagram) are passed through a set of activation functions (the identity, square and sine function in the above diagram)
* The activation function outputs are then passed through a batch normalization layer
* The batch-normalized activation function outputs are used as inputs for the next layer (denoted as H1 to H6 in the above diagram)
* The batch-normalized activation function outputs are also passed to an addition operator and a multiplication operator, each weighted with individual weights
* The output of the addition and multplication operators are then batch normalized and used as inputs for the next layer (denoted as H7 and H8 respectively in the above diagram)
* The outputs of this layer (denoted as H1 to H8 in the above diagram) are used as inputs for the next layer (where they will be passed onto activation functions and then the addition and multiplication operator)
* On the final layer, the outputs (denoted as K1 to KN in the above diagram) are passed to an addition operator, each weighted with individual weights, and the sum is taken as the final output of the neural network

### Training Details
The data is z-normalized and then the neural network is used to train on this data. Learning rate is decayed every time the loss value of either the training or validation set is higher than the average of the loss values of the past 10 iterations (this number can be changed as a hyperparameter). Training stops when either the loss reaches a certain threshold, or when the learning rate decays to a certain threshold.

### Extraction Algorithm
Once the neural network is done training, the structure of the network is extracted into a mathematical equation. The extraction algorithm starts with the final output neuron of the neural network, and replaces it with a mathematical expression, denoting the summation of the previous layer's outputs. It then recursively replaces each neuron output with a mathematical expression equivalent to the operator's function (i.e. addition, multiplication, activation functions, and batch normalization), until the remaining neurons in the mathematical expression are the input neurons of the neural network.

The equation is then expanded and simplified, and then the equation weights and coefficients are iteratively rounded to the nearest 0 to 12 digits (this number can be changed as a hyperparameter). It then performs testing with these rounded equations and keeps the one that performs the best.

### Requirements
* Python 3.7.4
* Tensorflow 2.3.0
* Scikit-learn 0.21.3
* Numpy 1.16.4
* Sympy 1.7.1

### Commentary
The algorithm indeed converges for simple equations, but has difficulty for more complex ones. Stacking additional layers, and adding additional activation functions has two effects:
* It exponentially increases the time it takes for the equation to be extracted due to the increase in time expanding and simplifying the equation
* It also runs the risk of blowing up the outputs due to the multiply operator receiving more and more inputs

Normalizing the data and batch normalizing layer outputs were used to counteract this and while it alleviates the issue a little bit, it doesn't eliminate the problem.

I also tried counteracting this by forcing the weights of the inputs going into the multiplication operator to be either 0 or 1 (i.e. a selection layer of some sort). I experimented with using binary weights on a toy example, but had trouble with the network training stalling when the weights reached 0, as the gradient for the previous layer weights would also be 0. There's also the edge case where if the gradient only updates the weights to values beyond the 1 or 0 threshold, the weight values would not change and thus the network training would stall. As such, I haven't tried binary weights on the neural network structure I described above for symbolic regression.

L1 regularization was tried to force more weights to be 0, but it didn't seem to work for varying values I tried. Experimentally, the algorithm performed better without any regularization at all. In theory, a penalty term that penalizes based on the number of non-zero weights, instead of the magnitude of the weights, is what this neural network needs for what it's doing. We should only care about prioritizing simplicity (i.e. fewer number of non-zero weights in the network, which would also help speed up the extraction of the mathematical equation from the network), rather than how big the mangitude is for some of the non-zero weights. However, I couldn't think of a differentiable penalty term (the derivative of an indicator variable outputting a constant in either the zero or non-zero condition would be 0) that could be added to the objective function.

### Additional Reading
* [AI Feynman](https://arxiv.org/pdf/1905.11481.pdf)
* [AI Feynman 2.0](https://arxiv.org/pdf/2006.10782.pdf)
* [BinaryConnect](https://arxiv.org/pdf/1511.00363.pdf)