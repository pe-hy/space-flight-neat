# space-flight-neat

This was an assignment created during the course Basics of Softcomputing during my Bc. studies at University of Ostrava.

## Instructions

Run the main.py file in the same folder as the asset folder to begin simulation.

Or run the SaveLoadDisplay.py file to load an exported .pkl file.

## SOFCO - Project 2 (EN)

### Defining the idea behind my implementation

In this project, I've decided to implement a neuroevolutionary algorithm for my own game. It is a simple 2D game set in space and the goal is to fly the spaceship as far as possible and avoid the approaching asteroids. There are currently 3 observations on the input layer of the neural network - the position of the artificial intelligence-controlled rocket on the y-axis, the changing distance of the nearest (relative to the rocket) moving asteroid on the x-axis and the position of the asteroid on the y-axis. Originally, the intention was to fully configure the neural network using the NEAT library. 
This is still possible, however, I further apply the softmax activation function on the output layer of the neural network from NEAT, in order to normalize the output values and to determine the probability of selected action - flight up, down, and stay in place. The NEAT neural network otherwise uses the ReLu activation function for each layer, as specified in the configuration file.

The genetic structure is managed by the NEAT library, where individual parameters can be changed in the configuration file.

### Evaluation of individuals

Fitness is initialized with a value of 0. For each timestep, while the individual is alive (i.e. does not get out of the window or hit the asteroid), a value of 0.1 is added. If it hits or gets out of the window, 5 is deducted. If it passes the asteroid, 0.5 is added.

`fitness_criterion = max` - evaluation of fitness based on the highest achieved value of the individual.

`fitness_threshold = 2000` - fitness value, when we end the simulation (learning process).


### Crossover operator

This is not trivial, as the NEAT algorithm changes the network topology. In one of the papers on NEAT [(1)](http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf), they describe a property where crossing takes place on the basis of historical markings of nodes and their incremental identification number according to their age. In the implementation of this library, the description is as follows: Those derived from a common ancestor (that are homologous) are matched up for crossover, and connections are matched if the nodes they connect have common ancestry. [(2)](https://neat-python.readthedocs.io/en/latest/neat_overview.html)

### Mutation operator

Again, this is not trivial. The network topology can also mutate and create new connections and nodes. New generations are created on the basis of fitness value and mutation of the best individuals. Since structural mutations and eg. mutations leading to updated weight values can cause problems, and while the result may appear positive in the long run, it may show negative results in the short term. The NEAT algorithm solves this by dividing individuals into species that have the shortest genetic distance. These groups of species then compete with each other in a simulation.

### The parameters

Population size - `pop_size = 200`

Probability (and strength) of mutation - different for bias, weights, activation functions, nodes.

`bias_mutate_power = 0.5` - bias mutation strength

`bias_mutate_rate = 0.6` - probability of bias mutation

`Activation_mutate_rate = 0.0` - probability of activation function mutation

`weight_mutate_power = 0.5` - weight mutation strength

`weight_mutate_rate = 0.8` - weight mutation probability

End of simulation condition - Fitness value exceeds 2000.

#### Sources:

https://neat-python.readthedocs.io/en/latest/neat_overview.html - NEAT documentation
http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf - Efficient Evolution of Neural Network Topologies

#### Inspired by:

https://github.com/techwithtim/NEAT-Flappy-Bird/ - NEAT-Flappy-Bird
