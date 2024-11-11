1. Defining the envrionment 
the grid - randomly generated
the obstacles - randomly posiitoned: to not be positioned in the same space with the goal and not out of the board
visualizatio of the grid before implementing the states

2. Defining the states
 - Each cell in the grid represents a unique state.
 - the states are represented as pairs of coordinates  it depends on the grid size chosen previously
This satisfied the requirement to the specifications of states.

3. Define the actions
- the agent moves up - decreasing the row index by 1
- the agent moves down - increasing the row index by 1
- the agent moves left -  decreasing the column index by 1
- the agent moves right - increasing the column index by 1

In the case of moving off the grid - is_valid_state function:
- If the agent moves  off the grid  - the action fails and it remains in the same cell
- If the agent moves into a cell which is an obstacle - the action fails and it remains in the same cell
This satisfied the requirement to the specifications of states.

The reward_grid - captures the rewards for each cell, and stores the values based on the value_iteration functions. 
At the initialization of the reward_grid:
- for the goal the reward_grid is set at 1
- for the obstacles the reward_grid is set to -1
- for the other states the reward_grid is set to 0

4. The transition probabilities
We assume a deterministic environment, where the agent’s action has the expected outcome unless blocked by obstacles or grid edges.

The rules:
We have the state with the coordinates (i,j)
- if the agent is moved into a valid  cell -> the probability is 1.
- if the agent is moved off the grid/into an obstacle -> the probability of staying in the same cell is 1

The specification of transition probabilities is satisfied - each action deterministically moves the agent to a new state unless blocked, in which case the agent stays in the same place.

5. The reward function
It assigns rewards based on the outcomes of actions in different states.

The rules:
 - if the agent reaches the goal state, it receives the reward +1
 - if the agent moves into an obstacle cell, it receives the reward -1 and remains in the same cell
 - if te agent moves to a regular cell, which is not an obstacle and not a goal cell, it receives the reward 0
This values are correlated with the reward_grid variable. 

The value_function is used to represent the value of each state in the grid.  

6. The value iteration functionality
The value function variable for the specific grid is initialized with 0 for all the states.
A variable delta is initalized by 0 to track the maximum change of the value function across all states and is compared at the end of each loop with a static variable set before the starting of the function, to see if it converges. 
Moving among the grid, starting with the upper left corer and moving to each column on the same row, the algorithm checks if the specific state is a goal one or an obstacle. In the case of being neither of them, it proceeds to the value computational function


To do
1. add a starting point
