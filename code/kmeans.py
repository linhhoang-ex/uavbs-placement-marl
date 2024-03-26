from pettingzoo import ParallelEnv
from ppo_sagin_v1_1 import *
from envs.sagin_v1 import *
from envs.utils import *


class Kmeans_Agent:
    def __init__(self, env: ParallelEnv) -> None:
        self.env = env
        self.n_users = env.unwrapped.n_users
        self.n_mbss = env.unwrapped.n_mbss
        self.n_uavs = env.unwrapped.n_uavs
        self.n_actions = env.action_space(env.possible_agents[0]).n
        self._action_to_direction = {
            0: np.array([1, 0]),        # east
            1: np.array([0, 1]),        # north
            2: np.array([-1, 0]),       # west
            3: np.array([0, -1]),       # south
            4: np.array([0, 0]),        # no movement
        }
        self.agents = ["uav_" + str(r) for r in range(self.n_uavs)]
        self.uav_velocity = env.unwrapped.uav_velocity
        self.bound = env.unwrapped.bound

    def get_h_dist(self, current, target):
        """
        Params:
            current: shape=(2, 1)
            target: shape=(2, 1)
        """
        delta = current - target

        return np.sqrt(np.sum(delta**2))

    def get_actions(
        self,
        user_locs: np.ndarray,
        mbs_locs: np.ndarray,
        uav_locs: np.ndarray
    ):
        """
        Params:
            user_locs: shape=(2, n_users)
            mbs_locs: shape=(2, n_mbss)
            uav_locs: shape=(2, n_uavs)
        """
        # reclustering -> get new cluster centroids
        bs_locs = np.concatenate((mbs_locs, uav_locs), axis=1)
        kmeans = kmeans_clustering(
            user_locs=user_locs,
            init_bs_locs=bs_locs,
            n_clusters=self.n_uavs + self.n_mbss
        )
        centroids = kmeans.cluster_centers_.transpose()  # shape=(2, n_clusters)

        # get movement decisions toward the centroid
        actions = {}
        for i in range(self.n_uavs):
            # get all actions making the UAV stationary/closer to the new cluster centroid
            target = centroids[:, self.n_mbss + i].reshape(2, -1)
            current_loc = uav_locs[:, i].reshape(2, -1)
            possible_actions = []
            old_dist = self.get_h_dist(current_loc, target)
            for action in self._action_to_direction.keys():
                direction = self._action_to_direction[action].reshape(2, -1)
                next_loc = current_loc + direction * self.uav_velocity
                next_loc = np.clip(next_loc, -self.bound, self.bound)
                if self.get_h_dist(next_loc.reshape(2, -1), target) <= old_dist:
                    possible_actions.append(action)

            # select the best movement (greedily)
            actions["uav_" + str(i)] = np.random.choice(possible_actions)

        return actions
