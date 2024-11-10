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

The reward_grid - captures the rewards for each cell, and stores the valkues based on the value_iteration functions. 
At the initialization of the reward_grid:
- for the goal the reward_grid is set at 1
- for the obstacles the reward_grid is set to -1
- for the other states the reward_grid is set to 0

4. The transition probabilities
We assume a deterministic environment, where the agent’s action has the expected outcome unless blocked by obstacles or grid edges.


the value_function is used to represent the value of each state in the grid.  