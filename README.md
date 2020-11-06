# space-flight-neat

## Instructions

Run the main.py file in the same folder as the asset folder to begin simulation.

Or run the SaveLoadDisplay.py file to load an exported .pkl file.

## SOFCO - Project 2 (EN)

### Defining the idea behind my implementation

In this project, I've decided to implement neuroevolutionary algorithms for my own game. It is a simple 2D game set in space and the goal is to fly the spaceship as far as possible and avoid the approaching asteroids. There are currently 3 observations on the input layer of the neural network - the position of the artificial intelligence-controlled rocket on the y-axis, the changing distance of the nearest (relative to the rocket) moving asteroid on the x-axis and the position of the asteroid on the y-axis. Originally, the intention was to fully configure the neural network using the NEAT library. 
This is still possible, however, I further apply the softmax activation function on the output layer of the neural network from NEAT, in order to normalize the output values and to determine the probability of selected action - flight up, down, and stay in place. The NEAT neural network otherwise uses the ReLu activation function for each layer, as specified in the configuration file.

The genetic structure is managed by the NEAT library, where individual parameters can be changed in the configuration file.

I will continue to work on the project and try to generalize the algorithm in the future so that the artificial intelligence can surpass an environment with more asteroids, environment with randomly changing speeds of asteroids, etc.

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


# space-flight-neat
## SOFCO - Projekt 2 (CZ)

### Vymezení problému řešení a navržení genetické struktury

V tomto projektu jsem se rozhodl na implementaci neuroevolučních algoritmů pro vlastní hru. Jde o jednoduchou 2D hru zasazenou do vesmíru a cílem je vesmírnou lodí proletět co nejdále a vyhnout se přibližujícím se asteroidům. Na vstupní vrstvě neuronové sítě jsou aktuálně 3 pozorování - pozice umělou inteligencí ovládané rakety na ose y, měnící se vzdálenost nejbližšího (vůči raketě) pohybujícího se asteroidu na ose x a pozice asteroidu na ose y. Původně bylo záměrem konfigurovat neuronovou síť čistě pomocí knihovny NEAT. 
To je stále možné, avšak na výstupní vrstvu neuronové sítě z NEATu, která má všechny vrstvy propojeny aktivační funkci ReLu, dále aplikuji softmax aktivační funkci, za účelem normalizovat výstupní hodnoty a určit pravděpodobnost zvolené akce - let nahoru, dolů, a zůstat na místě.

O genetickou strukturu se stará knihovna NEAT, kde v konfiguraci lze nastavovat jednotlivé parametry.

Na projektu budu dále pracovat a pokusím se v budoucnu algoritmus generalizovat tak, aby umělá inteligence zvládla překonat prostředí s více asteroidy, náhodně měnící se rychlostí asteroidů atp.

### Ohodnocení jedinců

Fitness se inicializuje s hodnotou 0. Za každý krok (timestep), dokud je jedinec naživu (tzn. nedostane se mimo okno nebo nenarazí do asteroidu), se přičítá hodnota 0.1. Pokud narazí nebo se dostane mimo okno, odečte se 5. Pokud se dostane za asteroid, přičte se 0.5. 

`fitness_criterion = max` - vyhodnocení fitness na základě nejvyšší dosažené hodnoty jedince.

`fitness_threshold = 2000` - hodnota fitness, kdy ukončujeme simulaci (proces učení).

### Operátor křížení

Není triviální, jelikož algoritmus NEAT mění topologii sítě. V jednom z vědeckých článků[(1)](http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf) o NEAT popisují vlastnost, kdy křížení probíhá na základě historických značení uzlů a jejich inkrementovaného identifikačního čísla dle stáří. V implementaci této knihovny je popis následovný: 
Uzly, které jsou odvozeny od společného předka (které jsou homologní), jsou porovnány pro křížení a jejich propojení budou shodná, pokud mají uzly, které spojují, společný původ (společného předka).  [(2)](https://neat-python.readthedocs.io/en/latest/neat_overview.html)

### Operátor mutace

Opět není triviální, topologie sítě taktéž může mutovat a vytvářet nová spojení a uzly. Nové generace vznikají na základě fitness hodnoty a mutace nejlepších jedinců. Jelikož strukturní mutace a např. mutace vedoucí k aktualizovaným váhovým hodnotám může způsobovat problémy, kdy výsledek se dlouhodobě může jevit jako kladný, tak krátkodobě může vykazovat záporné výsledky. Algoritmus NEAT toto řeší tak, že rozděluje jedince na druhy, které mají nejkratší genetickou vzdálenost. Tyto skupinky druhů spolu poté soupeří v simulaci.

### Nastavení parametrů

Velikost populace - `pop_size = 200`

Pravděpodobnost (a síla) mutace - různá u biasu, váh, aktivačních funkcí, uzlů.

`bias_mutate_power      = 0.5` - síla mutace biasu

`bias_mutate_rate        = 0.6` - pravděpodobnost mutace biasu

`activation_mutate_rate  = 0.0` - pravděpodobnost mutace aktivační funkce

`weight_mutate_power     = 0.5` - síla mutace váh

`weight_mutate_rate      = 0.8` - pravděpodobnost mutace váh

Podmínka ukončení výpočtu - Fitness hodnota přesáhne 2000.


#### Sources:

https://neat-python.readthedocs.io/en/latest/neat_overview.html - NEAT documentation
http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf - Efficient Evolution of Neural Network Topologies

#### Inspired by:

https://github.com/techwithtim/NEAT-Flappy-Bird/ - NEAT-Flappy-Bird
