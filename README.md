### SAGINEnv_V0

Implement using the parallel environment template.

### SAGINEnv_V1.0

Implement using the AEC environment template -> convert to a parallel env using PettingZoo's wrapper: parallel_wrapper_fn().

### SAGINEnv_V1.1

Same as the SAGINEnv_V1.0, except that the observation of the agent is flattened.

Observation:

- a heatmap of the user distribution being flattened
- normalized location of the macro BS
- normalized location of all drone BSs

### Points to be considered

If the distance to the mBS > the remaining distance that the UAV can travel

- +1 if the coordinated actions either
  - increase the # of satisfied users
  - increase the sum distance from the drone BSs to the macro BS
- -1 if the coordinated actions either
  - decrease the # of satisfied users
  - violate the flying time limit
    (i.e., distance to the mBS > the remaining distance that the UAV can travel)
- 0 if the # of satisfied users is unchanged

Otherwise, use action masking to force the UAV to return to the initial location.
