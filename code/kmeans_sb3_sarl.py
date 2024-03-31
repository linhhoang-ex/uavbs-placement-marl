from pettingzoo import ParallelEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from ppo_sagin_v1_1 import *
from envs.sagin_v1 import *
from envs.utils import *


class Kmeans_SB3_Agent:
    def __init__(self, env: ParallelEnv, model: BaseAlgorithm) -> None:
        """
        Args:
            env (ParallelEnv): an instance of the parallel_env (multi-agent)
            model (BaseAlgorithm): a trained model (SB3) in the SingeDroneEnv
        """
        self.env = env
        self.model = model

        self.n_mbss = env.unwrapped.n_mbss
        self.n_uavs = env.unwrapped.n_uavs
        self.agents = ["uav_" + str(r) for r in range(self.n_uavs)]

    def get_actions(
        self,
        user_locs: np.ndarray,
        mbs_locs: np.ndarray,
        uav_locs: np.ndarray
    ):
        # re-cluster users to base stations
        bs_locs = np.concatenate((mbs_locs, uav_locs), axis=1)
        kmeans = kmeans_clustering(
            user_locs=user_locs,
            init_bs_locs=bs_locs,
            n_clusters=self.n_uavs + self.n_mbss
        )
        labels = kmeans.labels_

        # get a movement decision for each UAV
        actions = {}
        for i, agent in enumerate(self.agents):
            cluster_id = self.n_mbss + i
            locs = {
                'self': uav_locs[:, i].reshape(2, -1),
                'user': user_locs[:, labels == cluster_id]
            }
            observation = get_obs_flattened(locs=locs, mode="sarl")
            action, _ = self.model.predict(observation)
            actions[agent] = action.item()

        return actions
