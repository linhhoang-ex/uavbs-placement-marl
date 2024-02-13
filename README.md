# # SAGINEnv_V1.0

Observation: 

- normalized location of all users
- normalized location of the macro BS
- normalized location of all drone BSs

Actions: 

- 0: move north
- 1: move west
- 2: move south
- 3: move east

Reward: 

If the distance to the mBS > the remaining distance that the UAV can travel

- +1 if the coordinated actions either
  - increase the # of satisfied users
  - increase the sum distance from the drone BSs to the macro BS
- -1 if the coordinated actions either
  - decrease the # of satisfied users
  - violate the flying time limit
    (i.e., distance to the mBS > the remaining distance that the UAV can travel)
- 0 if the # of satisfied users is unchanged


Some points to be considered:

- each agent receives a reward from the environment
- all agents have the same observation
