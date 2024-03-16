### SAGINEnv_V0

Implement using the parallel environment template.

### SAGINEnv_V1.0

Implement using the AEC environment template -> convert to a parallel env using PettingZoo's wrapper: parallel_wrapper_fn().

### SAGINEnv_V1.1

Same as the SAGINEnv_V1.0, except that the observation of the agent is flattened.

##### Observation:

- a heatmap of the user distribution being flattened
- normalized location of the macro BS
- normalized location of all drone BSs
- remove the distance limit in the observation of agents. The agetn can see all users in its clustering, plus all users not satisfied with the rates provided by other base stations.

Note: in V1.0, the agent must read the heatmap (i.e., an image) to figure out the location of itself and other base stations.

##### Reward model:
- global (w=0.8): +1/-1/0 based on the # of satisfied users
- local (w=0.2, newly added): +1/-1/0 based on the # of users served by the coresponding drone BS

##### Other changes:
- add K-means clustering, can be used for initial placement of drone BSs or for benchmarking (Kmeans (clustering) + Greedy (movement))
- increase the bandwidth of the mBS:
  mBS -> within 1.6km (20 Mbps)
  droneBS -> within ~700m (20 Mbps)
- Initilize hotspot areas: explicitly initiate 4 hotspots in the four sub-areas (upper left, lower left, lower right, and upper right)
- Initial placement of UAVs: random in the target area. Tested with K-means-based clustering, but the initial placement with K-means might not facilitate the training (too close to the optimal placement).
- Initial placement of users: does not force the user coordinates to be within the target area (using np.clip()). Instead, consider those users as outliers and disregard them.
- User association: assign users to the base staions with the strongest signal, removing the priority in selecting the base station.

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
