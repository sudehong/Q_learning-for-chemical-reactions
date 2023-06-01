# Q_learning-for-chemical-reactions
Using Q-learning to predict chemical reactions (bonding and cleavage).


First, create a graph with a total of 5 molecules as reactants and convert it into an adjacency matrix. 
Establish a state transition matrix to determine whether conversion between molecules is possible. 
Create a transition reward matrix where the value is 1 only if the final goal is reached, and 0 otherwise. 
Utilize Q-learning to generate the values of all states.
