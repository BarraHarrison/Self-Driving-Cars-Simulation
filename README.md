# Self-Driving Car Simulation in Python
The goal was to create a system where cars learn to follow the racetrack (map1.png)
without human intervention. This is done byevolving neural networks over successive
generations. I used Python and Pygame libraries as well as NEAT (NeuroEvolution of Augmenting Topologies)
to create this self-driving simulation.

## Aims of the Program
- Cars self-drive and navigate the race track using sensors and neural networks
- NEAT is used to evolve the car's decision-making abilities
- Collision detection, fitness evaluation and boundary avoidence are implemented

## What is NEAT?
NEAT (NeuroEvolution of Augmenting Topologies) is a genetic algorithm designed to evolve
neural networks. It starts with simple networks and adds complexity. It allows networks to
develop better strategies over time.

In the self-driving simulation, NEAT processes inputs from the sensors and outputs decisions (turning and accelerating).
Fitness scores encourages the cars to pass their traits to the next generation, making sure the car improves with each generation.

## Challenges Faced
- Stagnation in Learning: Early generations failed to improve significantly
- Boundary Avoidance: Cars would crash into boundaries or stop moving completely
- Specifice Scenarios: Cars struggled with sharp turns.
