import numpy as np
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# import functools
import gymnasium
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from envs.utils import gen_hist2d, get_obs_flattened


def get_horizontal_dist(
    bs_loc: np.ndarray,     # x- and y-coordinates of the BS, shape=(2,)
    users_loc: np.ndarray   # x- and y-coordinates of users, shape=(2 n_users)
) -> np.ndarray:            # horizontal distance to the BS for each user, shape=(n_users,)
    '''Calculate the horizontal distance (in meters) to the base station for each user.'''
    delta = bs_loc.reshape(2, 1) - users_loc

    return np.sqrt(np.sum(delta**2, axis=0))


def kmeans_clustering(
    user_locs: np.ndarray,
    init_bs_locs: Any,
    n_clusters: int,
) -> Tuple[Any]:
    '''Cluster users in to clusters using KMeans Clustering.

    Parameters
    ----------
        user_locs: positions of all users, shape=(2, n_users)
        init_bs_locs: init positions of all BSs, shape=(2, n_bss)

    Returns
    -------
        kmeans: an object of sklearn.cluster.KMeans class, with attributes:
            cluster_centers_: ndarray of shape (2, n_clusters)
                Coordinates of cluster centers
            labels_: ndarray of shape (n_samples,)
                Labels of each point
            inertia_: float
                Sum of squared distances of samples to their closest cluster center.

    References
    ---------
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    '''
    kmeans = KMeans(n_clusters=n_clusters,
                    n_init=1,
                    init=init_bs_locs.transpose(),
                    verbose=0
                    ).fit(user_locs.transpose())

    # return kmeans.cluster_centers_.transpose(), kmeans.labels_, kmeans.inertia_
    return kmeans


def get_snr_macrobs_db(
    rng,                        # an instant of np.random.default_rng()
    h_dist: np.ndarray,         # horizontal distance (m) to the mBS, shape=(n_users,)
    logF_db: np.float_ = 2,     # std var for the log-normally distributed shadowing
    noise_db: np.float_ = -90,  # total noise power in dBm
    pTx_dBm: np.float_ = 46,    # transmit power in dBm of the macro BS
    fading: bool = True,        # whether or not consider fading effects
    n_samples: np.float_ = 250  # no. of samples/s for SNR w/ coherent time ~40ms @fc=2GHz, v=15m/s
) -> Tuple[Any]:     # snr [dB] for each user, shape=(n_users)
    '''Calculate the path loss [dB] and SNR [dB] for each user via the macro BS's link.

    References:
    - https://www.arib.or.jp/english/html/overview/doc/STD-T63v9_20/5_Appendix/Rel5/25/25942-530.pdf
    - https://en.wikipedia.org/wiki/Coherence_time_(communications_systems) (coherent time)
    '''
    h_dist = np.maximum(h_dist, 1)          # at leat 1 m (the reference dist), avoid log(0)
    p_loss_mean_db = 128.1 + 37.6 * np.log10(h_dist / 1e3)
    if fading is True:
        fading_db = rng.normal(
            loc=0, scale=logF_db, size=(len(h_dist), n_samples)
        ).mean(axis=-1)                 # considering coherent time
        p_loss_db = p_loss_mean_db + fading_db
    else:
        p_loss_db = p_loss_mean_db
    snr_mean_db = pTx_dBm - p_loss_mean_db - noise_db
    snr_db = pTx_dBm - p_loss_db - noise_db
    return (snr_db, p_loss_db, snr_mean_db)


def get_snr_uavbs_db(
    rng,                            # an instant of np.random.default_rng()
    h_dist_m: np.ndarray,           # horizontal distance to the drone BS, shape=(n_users,)
    flying_alt: np.float_ = 120,    # flying altitude of the drone BS
    ref_pw_db: np.float_ = -47,     # reference signal power [dB] at d0=1m and fc=5.8GHz
    noise_db: np.float_ = -90,      # total noise power in dBm
    pTx_dBm: np.float_ = 30,        # transmit power in dBm of the drone BS
    kappa: np.float_ = 50,          # coefficient for the Rician channel effect
    p_loss_coeff: np.float_ = 2.7,  # path loss's coefficient
    fading: bool = True,            # whether or not to consider fading effects
    n_samples: np.float_ = 667      # no. of samples/s for SNR w/ coherent time ~15ms @fc=5.8GHz, v=15m/s
) -> Tuple[Any]:
    '''Calculate the path loss [dB] and SNR [dB] for each user via the drone BS's link.

    References:
    - https://doi.org/10.1109/TWC.2019.2926279 (for the comm. model)
    - https://doi.org/10.1109/LWC.2017.2710045 (path loss coefficients)
    - https://en.wikipedia.org/wiki/Coherence_time_(communications_systems) (coherent time)
    '''
    dist_m = np.sqrt(h_dist_m**2 + flying_alt**2)
    p_loss_db = 10 * p_loss_coeff * np.log10(dist_m)
    if fading is True:
        psi_rician = np.sqrt(kappa / (1 + kappa)) \
            + np.sqrt(1 / (1 + kappa)) * rng.normal(size=(len(h_dist_m), n_samples))
        psi_rician = np.mean(psi_rician, axis=-1)   # considering coherent time
    else:
        psi_rician = np.sqrt(kappa / (1 + kappa))
    snr_db = pTx_dBm + ref_pw_db + to_dB(psi_rician**2) - p_loss_db - noise_db
    snr_mean_db = pTx_dBm + ref_pw_db - p_loss_db - noise_db

    return (snr_db, p_loss_db, snr_mean_db)


def get_drate_bps(
    snr_db: np.ndarray,             # SNR [dB] of all users, shape=(n_users,)
    type: str,                      # in ['uav', 'mbs']
    mbs_bw_mhz: np.ndarray = 20,    # bandwidth in MHz for the macro BS's link
    uav_bw_mhz: np.ndarray = 20,    # bandwidth in MHz for the drone BS's link
) -> np.ndarray:                # data rate [bps] for each user, shape = n_users
    '''Calculate the data rate [bps] via the SNR [dB] and channel bandwidth [MHz]'''
    if type == 'uav':
        return uav_bw_mhz * 1e6 * np.log2(1 + convert_from_dB(snr_db))
    elif type == 'mbs':
        return mbs_bw_mhz * 1e6 * np.log2(1 + convert_from_dB(snr_db))
    else:
        raise ValueError("type can only be 'mbs' or 'uav'")


def to_dB(val):
    '''Convert real values to dB.'''
    return 10 * np.log10(val)


def convert_from_dB(val_dB):
    '''Convert dB to real values.'''
    return 10 ** (val_dB / 10)


def env(**kwargs):
    env = raw_env(**kwargs)
    if env.continuous:
        env = wrappers.ClipOutOfBoundsWrapper(env)
    else:
        env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv):
    metadata = {
        "name": "sagin_v1.1",
        "description": "AEC version (turn-based games)",
        "render_modes": ["rgb_array"],
        "is_parallelizable": True,
        "action_masking": False
    }

    def __init__(self, bound=1000, n_uavs=3, n_mbss=1, uav_altitude=120,
                 uav_velocity=25, continuous=False, render_mode=None,
                 hotspots=None, max_cycles=512, drate_threshold=20e6,
                 local_reward_ratio=0.2, drate_reward_ratio=0, seed=None,
                 uav_init_mode='random', uav_init_locs=None,
                 link_assignment='greedy (drate)', fading=True,
                 user_mode='stationary', user_velocity=0):
        '''
        Params
        ------
        uav_init_mode: in ['random', 'kmeans', 'specific']
            If uav_init_mode == 'specific', uav_init_locs must be defined.

        link_assignment: in ["greedy (drate)", "kmeans"]
            "greedy (drate)": assign users to the BS with the strongest signal
            "kmeans": using k-means clustering to assign users to BSs

        user_mode (str, optional): the mobility model of users, in ["stationary", "random walk"]
            If user_mode is not "stationary", user_velocity (>0) must be defined.
            "stationary": no movements
            "random walk": random walk model with 9 degrees of freedom \n

        fading (bool, optional): whether fading is considered in communications.
        '''
        self.bound = bound      # boundary [m] of the area, x, y in range [-bound, bound]
        self.n_uavs = n_uavs    # Number of drone base stations
        self.n_mbss = n_mbss    # Number of macro base stations
        self.uav_altitude = uav_altitude    # The UAV's flying altitude
        self.uav_init_mode = uav_init_mode
        self.uav_init_locs = uav_init_locs  # init positions of UAVs
        self.uav_velocity = uav_velocity    # The UAV's velocity (in m/s)
        self.hotspots = hotspots            # Infos about hotspot areas
        self.max_cycles = max_cycles        # Max no. of steps in an episode
        self.link_assignment = link_assignment
        self.user_mode = user_mode
        self.user_velocity = user_velocity
        self.fading = fading
        self.local_rw_ratio = local_reward_ratio
        self.drate_rw_ratio = drate_reward_ratio
        self.drate_threshold = drate_threshold  # satisfactory data rate level in bps

        self.agents = ["uav_" + str(r) for r in range(self.n_uavs)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_uavs))))
        self._agent_selector = agent_selector(self.agents)

        self.obs_shape = get_obs_flattened(
            locs={},
            n_uavs=self.n_uavs,
            n_mbss=self.n_mbss
        ).shape         # observation shape of one agent
        self.continuous = continuous    # action space: continuous/discrete
        self.render_mode = render_mode

        assert self.n_uavs > 1, "n_uavs must be greater than 1 (multi-agent)"

        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    gymnasium.spaces.Box(
                        low=0,
                        high=1,
                        shape=self.obs_shape,
                        dtype=np.float64,
                    )
                ] * self.n_uavs,
            )
        )

        assert self.continuous is False, "Currently support discrete actions only"
        if self.continuous:
            pass    # To be considered
        else:
            # 5 actions: move to the east (right), north (up), west (left),
            # south (down), and remain stationary (i.e., no movement)
            self.action_spaces = dict(
                zip(self.agents, [gymnasium.spaces.Discrete(5)] * self.n_uavs)
            )

        """
        The following dictionary maps abstract actions from `self.action_space`
        to the direction we will walk in if that action is taken.
        Example: 0 corresponds to "east", 1 to "north" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),        # east
            1: np.array([0, 1]),        # north
            2: np.array([-1, 0]),       # west
            3: np.array([0, -1]),       # south
            4: np.array([0, 0]),        # no movement
        }

        self.state_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=self.obs_shape,
            dtype=np.uint8,
        )

        self.terminate = False
        self.truncate = False

        self._seed(seed)

    # @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        '''Takes in agent and returns the observation space for that agent.'''
        return self.observation_spaces[agent]

    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        '''Takes in agent and returns the action space for that agent.'''
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        # self.np_random is an instance of np.random.default_rng().
        self.np_random, seed = seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        '''Reset the environment to a starting point and set up the environment
        so that render(), step(), and observe() can be called without issues.
        '''
        if seed is not None:
            self._seed(seed)

        # Init hotspot areas
        rng = self.np_random
        if self.hotspots is None:
            self.hotspots = [
                (0, 0, 1000, 100),               # (x0, y0, stddev, n_users)
                (1000, 1000, 1000, 200),
                (rng.normal(-500, 200), rng.normal(500, 200), 300, 50),
                (rng.normal(-500, 200), rng.normal(-500, 200), 300, 50),
                (rng.normal(500, 200), rng.normal(-500, 200), 300, 50),
                (rng.normal(500, 200), rng.normal(500, 200), 300, 50)
            ]
            # for _ in range(3):
            #     self.hotspots.append((
            #         self.np_random.uniform(-800, 800),      # x0
            #         self.np_random.uniform(-800, 800),      # y0
            #         200,                                    # stddev
            #         50                                      # n_users
            #     ))

        # Init locations of all network entities (users, macroBSs, and droneBSs)
        # Shape of location arrays: (2, n_users), (2, n_mbss), and (2, n_uavs)
        self.locs = {
            'user': self.gen_user_init_locs(self.hotspots),
            'mbs': self.gen_mbs_loc()
        }
        if self.uav_init_mode == 'specific':
            self.locs['uav'] = self.uav_init_locs.copy()
        else:
            self.locs['uav'] = self.gen_uav_init_locs(
                mode=self.uav_init_mode,
                user_locs=self.locs['user'],
                mbs_locs=self.locs['mbs']
            )
            self.uav_init_locs = self.locs['uav'].copy()
        self.n_users = self.locs['user'].shape[-1]
        self.check_locations_in_bound()

        # Initialization for step() method
        self.agents = self.possible_agents[:]

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.terminate = False
        self.truncate = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        if self.link_assignment == "greedy (drate)":
            kpis = self.get_drates_and_link_assignment_greedy()
        elif self.link_assignment == "kmeans":
            kpis = self.get_drates_and_link_assignment_kmeans()
        self.infos['global'] = kpis

        # No. of steps elapsed in the episode, for truncation condition
        self.n_steps = 0

    def step(self, action):
        '''Accepts and executes the action of the current agent_selection in the environment.
        Automatically switches control to the next agent.
        '''
        if (
            self.terminations[self.agent_selection] or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        if self.continuous:
            raise Exception("Currently support discrete actions only.")
        else:
            self.move_uav(agent, action)

        # Update rewards, terminations, and truncations if the current agent is the last
        if self._agent_selector.is_last():
            self.check_locations_in_bound()
            if self.link_assignment == "greedy (drate)":
                new_kpis = self.get_drates_and_link_assignment_greedy()
            elif self.link_assignment == "kmeans":
                new_kpis = self.get_drates_and_link_assignment_kmeans()

            # Calculate rewards for each agent
            self.rewards = self.get_rewards(new_kpis)

            # Must be updated after calculating rewards: update .infos['global']
            # self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
            self.infos['global'] = new_kpis

            # Update locations of users
            self.move_users()

            # Check truncation conditions (overwrites termination conditions)
            self.n_steps += 1
            self.truncate = self.n_steps >= self.max_cycles - 1
            self.truncations = {agent: self.truncate for agent in self.agents}

            # Check termination conditions
            self.terminations = {agent: self.terminate for agent in self.agents}

        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()

    def render(self):
        pass

    def close(self):
        '''Close any resources that should be released.'''
        plt.close()

    def observe(self, agent):
        '''Return the observation an agent currently can make.'''
        i = self.agent_name_mapping[agent]
        locs_ = self.locs.copy()
        locs_['self'] = locs_['uav'][:, i].reshape(2, -1)
        locs_['uav'] = np.delete(locs_['uav'], i, 1)

        # Option 1: UAV-BS see only users it is assigned to
        # mapping = self.infos['global']['bs_mapping'].copy()
        # locs_['user'] = locs_['user'][:, (mapping - self.n_mbss) == i]

        # Option 2: UAV-BS see users within a specific range (e.g., 400m)
        # mapping = self.infos['global']['bs_mapping'].copy()
        # h_dist = get_horizontal_dist(locs_['self'], locs_['user'])
        # drates = self.infos['global']['drates'].copy()
        # mask = ((mapping - self.n_mbss) == i) | ((h_dist <= 700) & (drates < self.drate_threshold))

        # Option 3: UAV-BS see users with unsatisfied data rates from the mBS
        # drates_mbs = self.infos['global']['drates_map'][0]
        # mask = drates_mbs < self.drate_threshold

        # Option 4: UAV-BS see users with unsatisfied data rates from the mBS and
        # other drone BSs (optional: within 700m)
        assert self.n_mbss == 1, "Currently support one macro BS only"
        mapping = self.infos['global']['bs_mapping'].copy()
        drates_mbs = self.infos['global']['drates_map'][0]
        drates_uav_alt = self.infos['global']['drates_map'][1:, :].copy()
        drates_uav_alt = np.delete(drates_uav_alt, i, 0)
        drates_uav_alt = np.max(drates_uav_alt, axis=0)
        # h_dist = get_horizontal_dist(locs_['self'], locs_['user'])
        mask = drates_mbs < self.drate_threshold        # not satisfied with the mBS
        mask &= drates_uav_alt < self.drate_threshold   # not satisfied with other droneBSs
        # mask &= h_dist <= 700
        mask |= (mapping - self.n_mbss) == i

        # Option 5: UAV-BS see its users and other users (optional: within 1000m) with
        # unsatisfied data rates from the mBS and all other drone BSs in higher priorities
        # assert self.n_mbss == 1, "Currently support one macro BS only"
        # mapping = self.infos['global']['bs_mapping'].copy()
        # drates_alt = self.infos['global']['drates_map'][:i + 1, :].copy()
        # drates_alt = np.max(drates_alt, axis=0)
        # # h_dist = get_horizontal_dist(locs_['self'], locs_['user'])
        # mask = (drates_alt < self.drate_threshold)
        # # mask &= (h_dist <= 1000)
        # mask |= (mapping - self.n_mbss) == i

        locs_['user'] = locs_['user'][:, mask]

        # return gen_hist2d(locs_)
        return get_obs_flattened(locs_)

    def state(self):
        '''Return a global view of the environment.'''
        return gen_hist2d(self.locs)

    def gen_mbs_loc(self) -> np.ndarray:
        assert self.n_mbss == 1, "Currently support one MBS only"
        return np.array([self.bound - 1, self.bound - 1]).reshape(2, -1)

    def gen_uav_init_locs(
        self,
        mode: str = 'random',
        **kwargs
    ) -> np.ndarray:
        '''Generate initial locations for all drone BSs.

            Params
            ------
            type: ['random', 'kmeans']
                If type == 'random': UAVs are initially located randomly.

                If type == 'kmeans': UAVs are initially located at the cluster \
                    centroids determined by K-means clustering.
                    In support of clustering, two additional args are required:
                        user_locs: np.ndarray,      # shape=(2, n_users)
                        mbs_locs: np.ndarray,       # shape=(2, n_users)

            Returns
            -------
            shape = (2, n_uavs).
        '''
        if mode == 'random':
            # Version 1: random initial locations
            xlocs = self.np_random.uniform(-self.bound, self.bound, self.n_uavs)
            ylocs = self.np_random.uniform(-self.bound, self.bound, self.n_uavs)

            return np.array([xlocs, ylocs])

        elif mode == 'kmeans':
            # Version 2: init UAVs at the cluster centroid
            user_locs = kwargs['user_locs']
            mbs_locs = kwargs['mbs_locs']
            if self.n_uavs <= 8:
                uav_init_locs = np.array([
                    [-1, 1],        # top left
                    [-1, -1],       # bottom left
                    [1, -1],        # bottom right
                    [0, 0],         # center
                    [-1, 0],
                    [0, -1],
                    [1, 0]
                ]) * self.bound
                uav_init_locs = uav_init_locs.transpose()
                init_locs = np.concatenate((mbs_locs, uav_init_locs[:, :self.n_uavs]), axis=1)
            else:
                xlocs = self.np_random.uniform(-self.bound, self.bound, self.n_uavs)
                ylocs = self.np_random.uniform(-self.bound, self.bound, self.n_uavs)
                uav_init_locs = np.array([xlocs, ylocs])
                init_locs = np.concatenate((mbs_locs, uav_init_locs), axis=1)

            kmeans = kmeans_clustering(
                user_locs=user_locs,
                init_bs_locs=init_locs,
                n_clusters=self.n_mbss + self.n_uavs
            )
            cluster_centers = kmeans.cluster_centers_.transpose()

            return cluster_centers[:, self.n_mbss:]

        else:
            raise KeyError("type shoule be in ['random', 'kmeans']")

    def gen_user_init_locs(
        self,
        hotspot_dict: List[Tuple[float, float, float, float]],
    ) -> np.ndarray:
        '''Generate user locations around some hotspots, each hot spot is represented
        as a tuple of (x0, y0, stddev, nusers). Output shape = (2, n_users).'''
        xlocs = np.array([])
        ylocs = np.array([])
        for x0, y0, stddev, nusers in hotspot_dict:
            xlocs = np.concatenate((xlocs, self.np_random.normal(x0, stddev, nusers)))
            ylocs = np.concatenate((ylocs, self.np_random.normal(y0, stddev, nusers)))

        locs = np.asarray([xlocs, ylocs])
        outliers = (locs[0] < -self.bound) | (locs[0] > self.bound)
        outliers |= (locs[1] < -self.bound) | (locs[1] > self.bound)
        locs = np.delete(locs, outliers, axis=1)

        return locs

    def move_uav(self, uav: str, action: np.ndarray):
        '''Update the location of the corresponding uav given an action'''
        agent_id = self.agent_name_mapping[uav]
        direction = self._action_to_direction[action]       # shape=(2,)
        next_loc = self.locs['uav'][:, agent_id] + direction * self.uav_velocity    # shape=(2,)
        next_loc = np.clip(next_loc, -self.bound, self.bound).flatten()
        self.locs['uav'][:, agent_id] = next_loc

    def move_users(self):
        """Update the locations of all users in self.locs['user']."""
        if self.user_mode == "stationary":
            pass

        if self.user_mode != "stationary":
            assert self.user_velocity > 0, \
                "if users are not stationary, user_velocity must be greater than 0"

        if self.user_mode == "random walk":
            curr_locs = self.locs['user'].copy()
            directions = self.np_random.choice([-1, 0, 1], size=(2, self.n_users))
            new_locs = curr_locs + self.user_velocity * directions
            new_locs = np.clip(new_locs, -self.bound, self.bound)
            self.locs['user'] = new_locs

    def get_drates_map(self, user_locs, bs_locs):
        """ Get the data rate on each link between a user and a base station.

        Returns:
            drates_map: shape=(n_bss, n_users)"""
        drates_map = np.zeros(shape=(self.n_mbss + self.n_uavs, self.n_users))
        for i in range(self.n_mbss + self.n_uavs):
            h_dist_ = get_horizontal_dist(bs_locs[:, i], user_locs)
            if i < self.n_mbss:
                snr_ = get_snr_macrobs_db(self.np_random, h_dist_, fading=self.fading)[0]
                drates_map[i, :] = get_drate_bps(snr_, 'mbs')
            else:
                snr_ = get_snr_uavbs_db(self.np_random, h_dist_, fading=self.fading)[0]
                drates_map[i, :] = get_drate_bps(snr_, 'uav')

        return drates_map

    def get_stats_on_uavbs(self, drates, bs_mapping):
        avg_drates_by_uavbs = []
        n_users_by_uavbs = []
        for i in range(self.n_uavs):
            rs = drates[bs_mapping == i + self.n_mbss]
            n_users_by_uavbs.append(rs.size)
            if rs.size == 0:        # no users are assigned to this drone BS
                avg_drates_by_uavbs.append(0)
            else:
                avg_drates_by_uavbs.append(rs.mean())

        return np.array(avg_drates_by_uavbs), np.array(n_users_by_uavbs)

    def get_drates_and_link_assignment_greedy(self):
        '''Associate users to macro and drone BSs: greedily based on downlink rates.
        The BS that offers the highest data rate is assigned to each user.

        Returns:
        - 'drates': data rates for each user, shape=(n_users,)
        - 'bs_mapping': indexes of the assigned BS for each user, shape=(n_users,)
        - 'drate_avg': average data rate of all users
        - 'n_satisfied': number of users with satisfied data rates
        - 'avg_drates_by_uavbs': average data rates provided by each drone BS, shape=(n_uavs,)
        '''
        user_locs = self.locs['user']
        bs_locs = np.concatenate((self.locs['mbs'], self.locs['uav']), axis=-1)

        # Calculate the data rate on each links for all users
        drates_map = self.get_drates_map(user_locs, bs_locs)

        # V1.0 (User association): assign users to mBS/droneBS with the strongest signal
        drates = np.max(drates_map, axis=0)
        bs_mapping = np.argmax(drates_map, axis=0)

        # V1.1 (User association): only assign to droneBS if the mBS's signal is not stronog enough
        # drates_mbs = drates_map[:self.n_mbss, :]
        # drates = np.max(drates_mbs, axis=0)
        # bs_mapping = np.argmax(drates_mbs, axis=0)
        # drates_uav = drates_map[self.n_mbss:, :]
        # mask1 = drates < self.drate_threshold
        # mask2 = drates < np.max(drates_uav, axis=0)
        # mask = mask1 & mask2
        # drates[mask] = np.max(drates_uav[:, mask], axis=0)
        # bs_mapping[mask] = self.n_mbss + np.argmax(drates_uav[:, mask], axis=0)

        # V1.2 (User association): only assign to a droneBS if the signal
        # from the mBS and all previous droneBSs is not stronog enough
        # drates = np.max(drates_map, axis=0)
        # bs_mapping = np.argmax(drates_map, axis=0)
        # for i in range(self.n_users):
        #     if sum(drates_map[:, i] >= self.drate_threshold) > 1:
        #         bs_mapping[i] = np.argmax(drates_map[:, i] >= self.drate_threshold)
        #         drates[i] = drates_map[bs_mapping[i], i]

        # For tracking KPIs
        drate_avg = drates.mean()
        n_satisfied = np.sum(drates >= self.drate_threshold)
        avg_drates_by_uavbs, n_users_by_uavbs = self.get_stats_on_uavbs(
            drates=drates, bs_mapping=bs_mapping
        )
        kpis = {
            'drates': drates,           # shape=(n_users,)
            'drates_map': drates_map,   # shape=(n_bss, n_users)
            'bs_mapping': bs_mapping,   # shape=(n_users,)
            'drate_avg': drate_avg,
            'n_satisfied': n_satisfied,
            'avg_drates_by_uavbs': avg_drates_by_uavbs,
            'n_users_by_uavbs': n_users_by_uavbs
        }

        return kpis

    def get_drates_and_link_assignment_kmeans(self):
        """Assign users to base stations using K-means clustering."""
        user_locs = self.locs['user']
        uav_locs = self.locs['uav']
        mbs_locs = self.locs['mbs']
        bs_locs = np.concatenate((mbs_locs, uav_locs), axis=1)

        kmeans = kmeans_clustering(
            user_locs=user_locs,
            init_bs_locs=bs_locs,
            n_clusters=self.n_uavs + self.n_mbss
        )

        bs_mapping = kmeans.labels_
        drates_map = self.get_drates_map(user_locs, bs_locs)
        drates = np.array([drates_map[bs_mapping[i], i] for i in range(self.n_users)])
        drate_avg = drates.mean()
        n_satisfied = np.sum(drates >= self.drate_threshold)
        avg_drates_by_uavbs, n_users_by_uavbs = self.get_stats_on_uavbs(
            drates=drates, bs_mapping=bs_mapping
        )
        kpis = {
            'drates': drates,           # shape=(n_users,)
            'drates_map': drates_map,   # shape=(n_bss, n_users)
            'bs_mapping': bs_mapping,   # shape=(n_users,)
            'drate_avg': drate_avg,
            'n_satisfied': n_satisfied,
            'avg_drates_by_uavbs': avg_drates_by_uavbs,
            'n_users_by_uavbs': n_users_by_uavbs
        }

        return kpis

    def get_global_reward(self, new_kpis: Dict[str, np.ndarray]) -> float:
        old_kpis = self.infos['global']

        if new_kpis['n_satisfied'] > old_kpis['n_satisfied']:
            n_satisfied_score = 1
        elif new_kpis['n_satisfied'] < old_kpis['n_satisfied']:
            n_satisfied_score = -1
        else:
            n_satisfied_score = 0

        # if new_kpis['drate_avg'] > old_kpis['drate_avg']:
        #     drate_score = 1
        # elif new_kpis['drate_avg'] < old_kpis['drate_avg']:
        #     drate_score = -1
        # else:
        #     drate_score = 0

        # return (1 - self.drate_rw_ratio) * n_satisfied_score\
        #     + self.drate_rw_ratio * drate_score

        # return new_kpis['n_satisfied'] - old_kpis['n_satisfied']

        return n_satisfied_score

    def get_local_rewards(self, new_kpis: Dict[str, np.ndarray]) -> np.ndarray:
        old_kpis = self.infos['global']

        n_users_scores = np.zeros(self.n_uavs)
        for i in range(self.n_uavs):
            if new_kpis['n_users_by_uavbs'][i] > old_kpis['n_users_by_uavbs'][i]:
                n_users_scores[i] = 1
            elif new_kpis['n_users_by_uavbs'][i] < old_kpis['n_users_by_uavbs'][i]:
                n_users_scores[i] = -1
            else:
                n_users_scores[i] = 0

        drate_scores = np.zeros(self.n_uavs)
        if new_kpis['avg_drates_by_uavbs'][i] > old_kpis['avg_drates_by_uavbs'][i]:
            drate_scores[i] = 1
        elif new_kpis['avg_drates_by_uavbs'][i] < old_kpis['avg_drates_by_uavbs'][i]:
            drate_scores[i] = -1
        else:
            drate_scores[i] = 0

        return (1 - self.drate_rw_ratio) * n_users_scores\
            + self.drate_rw_ratio * drate_scores

        # return n_users_scores

    def get_rewards(self, new_kpis: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        global_rewards = self.get_global_reward(new_kpis)
        local_rewards = self.get_local_rewards(new_kpis)
        rewards = (1 - self.local_rw_ratio) * global_rewards\
            + self.local_rw_ratio * local_rewards

        return dict(zip(self.agents, rewards))

    def check_locations_in_bound(self):
        for val in self.locs.values():
            assert np.all(val <= self.bound), "all locations must in range [-bound, bound]"
            assert np.all(val >= -self.bound), "all locations must in range [-bound, bound]"


class SingleDroneEnv(gymnasium.Env):
    def __init__(
        self, bound=1000, n_uavs=1, n_mbss=1, uav_altitude=120,
        uav_velocity=25, render_mode=None,
        hotspots=None, max_cycles=512, drate_threshold=20e6,
        drate_reward_ratio=1, step_penalty=0.05, seed=None,
        uav_init_mode='random', uav_init_locs=None,
        d_terminate=15
    ):
        """
        Params
        ------
        uav_init_mode: ['random', 'specific']
            If uav_init_mode == 'specific', uav_init_locs must be defined.
        """
        self.bound = bound      # boundary [m] of the area, x, y in range [-bound, bound]
        self.n_uavs = n_uavs    # Number of drone base stations
        self.n_mbss = n_mbss    # Number of macro base stations
        self.uav_altitude = uav_altitude    # The UAV's flying altitude
        self.uav_init_mode = uav_init_mode
        self.uav_init_locs = uav_init_locs  # init positions of UAVs
        self.uav_velocity = uav_velocity    # The UAV's velocity (in m/s)
        self.hotspots = hotspots            # Infos about hotspot areas
        self.max_cycles = max_cycles        # Max no. of steps in an episode

        self.drate_rw_ratio = drate_reward_ratio
        self.step_penalty = step_penalty
        self.drate_threshold = drate_threshold  # satisfactory data rate level in bps
        self.d_terminate = d_terminate  # Terminate condition, distance(drone, hotspot)

        self.obs_shape = get_obs_flattened(
            locs={},
            n_uavs=self.n_uavs,
            n_mbss=self.n_mbss,
            mode="sarl"
        ).shape         # observation shape of one agent
        self.render_mode = render_mode

        assert self.n_uavs == 1, "n_uavs must be one (single-agent scenario)"

        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=self.obs_shape,
            dtype=np.float64,
        )

        # 5 actions: move to the east (right), north (up), west (left),
        # south (down), and remain stationary (i.e., no movement)
        self.action_space = gymnasium.spaces.Discrete(5)
        self._action_to_direction = {
            0: np.array([1, 0]),        # east
            1: np.array([0, 1]),        # north
            2: np.array([-1, 0]),       # west
            3: np.array([0, -1]),       # south
            4: np.array([0, 0]),        # no movement
        }

        self.terminate = False
        self.truncate = False

        self._seed(seed)

    def _seed(self, seed=None):
        # self.np_random is an instance of np.random.default_rng().
        self.np_random, seed = seeding.np_random(seed)

    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)

        self.locs = {'user': self.gen_user_init_locs()}
        if self.uav_init_mode == "random":
            self.locs['self'] = self.gen_uav_init_loc()
            self.uav_init_loc = self.locs['self'].copy()
        elif self.uav_init_mode == "specific":
            self.locs['self'] = self.uav_init_locs.copy()
        self.locs['mbs'] = np.array([self.bound, self.bound]).reshape(2, -1)
        self.hotspot_loc = np.array([
            np.mean(self.locs['user'][0]),
            np.mean(self.locs['user'][1])
        ]).reshape(2, -1)
        self.n_users = self.locs['user'].shape[-1]
        self.check_locations_in_bound()

        observation = get_obs_flattened(
            locs=self.locs,
            mode="sarl"
        )
        kpis = self.get_kpis()
        self.info = {'kpis': kpis}

        # No. of steps elapsed in the episode, for truncation condition
        self.n_steps = 0

        return observation, self.info

    def step(self, action):
        self.n_steps += 1
        self.move_uav(
            action=action,
            # clipping=False
        )
        new_kpis = self.get_kpis()
        reward = self.get_reward(new_kpis)
        self.info['kpis'] = new_kpis
        observation = get_obs_flattened(
            locs=self.locs,
            mode="sarl"
        )
        terminated = np.all(
            np.isclose(self.locs['self'], self.hotspot_loc, atol=self.d_terminate)
        ).item()
        terminated |= self.n_steps >= (self.max_cycles - 1)

        return observation, reward, terminated, False, self.info

    def state(self):
        '''Return a global view of the environment.'''
        return gen_hist2d(self.locs, mode="sarl")

    def gen_user_init_locs(self):
        rng = self.np_random
        stddev = 300
        n_users = 100
        hotspot_loc = rng.uniform(-self.bound + stddev, self.bound - stddev, size=2)
        xlocs = rng.normal(hotspot_loc[0], stddev, n_users)
        ylocs = rng.normal(hotspot_loc[1], stddev, n_users)

        locs = np.asarray([xlocs, ylocs])
        outliers = (locs[0] < -self.bound) | (locs[0] > self.bound)
        outliers |= (locs[1] < -self.bound) | (locs[1] > self.bound)
        locs = np.delete(locs, outliers, axis=1)

        return locs

    def gen_uav_init_loc(self):
        return self.np_random.uniform(-self.bound, self.bound, size=(2, 1))

    def check_locations_in_bound(self):
        for val in self.locs.values():
            assert np.all(val <= self.bound), "all locations must in range [-bound, bound]"
            assert np.all(val >= -self.bound), "all locations must in range [-bound, bound]"

    def move_uav(self, action: np.ndarray, clipping=True):
        '''Update the location of the corresponding uav given an action'''
        direction = self._action_to_direction[action]       # shape=(2,)
        next_loc = self.locs['self'].flatten() + direction * self.uav_velocity  # shape=(2,)
        if clipping is True:
            next_loc = np.clip(next_loc, -self.bound, self.bound)
        self.locs['self'] = next_loc.reshape(2, -1)

    def get_kpis(self):
        hdist_ = get_horizontal_dist(self.locs['self'], self.locs['user'])
        snr_ = get_snr_uavbs_db(self.np_random, hdist_, fading=False)[0]
        drates = get_drate_bps(snr_, 'uav')
        kpis = {
            'drates': drates,
            'drate_avg': drates.mean(),
            'n_satisfied': np.sum(drates >= self.drate_threshold)
        }

        return kpis

    def get_reward(self, new_kpis):
        old_kpis = self.info['kpis']
        if new_kpis['n_satisfied'] > old_kpis['n_satisfied']:
            n_satisfied_score = 1
        elif new_kpis['n_satisfied'] < old_kpis['n_satisfied']:
            n_satisfied_score = -1
        else:
            n_satisfied_score = -0.05

        if new_kpis['drate_avg'] > old_kpis['drate_avg']:
            drate_score = 1
        elif new_kpis['drate_avg'] < old_kpis['drate_avg']:
            drate_score = -1
        else:
            drate_score = -0.1

        return (1 - self.drate_rw_ratio) * n_satisfied_score\
            + self.drate_rw_ratio * drate_score - self.step_penalty
