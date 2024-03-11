import numpy as np
from typing import Dict, Any, Tuple, List
# import functools
import gymnasium
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from envs.utils import gen_hist2d


def get_horizontal_dist(
    bs_loc: np.ndarray,     # x- and y-coordinates of the BS, shape=(2,)
    users_loc: np.ndarray   # x- and y-coordinates of users, shape=(2 n_users)
) -> np.ndarray:            # horizontal distance to the BS for each user, shape=(n_users,)
    '''Calculate the horizontal distance (in meters) to the base station for each user.'''
    delta = bs_loc.reshape(2, 1) - users_loc

    return np.sqrt(np.sum(delta**2, axis=0))


def get_drate_bps(
    snr_db: np.ndarray,         # SNR [dB] of all users, shape=(n_users,)
    bw_mhz: np.ndarray = 15,    # bandwidth in MHz for each user, shape=(n_users,)
) -> np.ndarray:                # data rate [bps] for each user, shape = n_users
    '''Calculate the data rate [bps] via the SNR [dB] and channel bandwidth [MHz]'''
    return bw_mhz * 1e6 * np.log2(1 + convert_from_dB(snr_db))


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
        "name": "sagin_v1",
        "description": "AEC version (turn-based games)",
        "render_modes": ["rgb_array"],
        "is_parallelizable": True,
        "action_masking": False
    }

    def __init__(self, bound=1000, n_uavs=3, n_mbss=1, uav_altitude=120,
                 uav_velocity=25, continuous=False,
                 hotspots=None, max_cycles=1800, drate_threshold=20e6,
                 local_reward_ratio=0, drate_reward_ratio=0, seed=42):
        self.bound = bound      # boundary [m] of the area, x, y in range [-bound, bound]
        self.n_uavs = n_uavs    # Number of drone base stations
        self.n_mbss = n_mbss    # Number of macro base stations
        self.uav_altitude = uav_altitude    # The UAV's flying altitude
        self.uav_velocity = uav_velocity    # The UAV's velocity (in m/s)
        self.hotspots = hotspots            # Infos about hotspot areas
        self.max_cycles = max_cycles  # Max no. of steps in an episode

        self.agents = ["uav_" + str(r) for r in range(self.n_uavs)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_uavs))))
        self._agent_selector = agent_selector(self.agents)

        self.local_rw_ratio = local_reward_ratio
        self.drate_rw_ratio = drate_reward_ratio
        self.drate_threshold = drate_threshold  # satisfactory data rate level in bps

        self.obs_shape = gen_hist2d(locs={}).shape  # observation shape of one agent
        self.continuous = continuous    # action space: continuous/discrete

        assert self.n_uavs > 1, "n_uavs must be greater than 1 (multi-agent)"

        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    gymnasium.spaces.Box(
                        low=0,
                        high=255,
                        shape=self.obs_shape,
                        dtype=np.uint8,
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
        if self.hotspots is None:
            self.hotspots = [
                (0, 0, 600, 100),               # (x0, y0, stddev, n_users)
            ]
            for _ in range(3):
                self.hotspots.append((
                    self.np_random.uniform(-1000, 1000),        # x0
                    self.np_random.uniform(-1000, 1000),        # y0
                    200,                                        # stddev
                    100                                         # n_users
                ))
        self.n_users = np.array([hotspot[-1] for hotspot in self.hotspots]).sum()

        # Init locations of all network entities (users, macroBSs, and droneBSs)
        # Shape of location arrays: (2, n_users), (2, n_mbss), and (2, n_uavs)
        self.locs = {
            'user': self.gen_user_init_locs(self.hotspots),
            'mbs': self.gen_mbs_loc(),
            'uav': self.gen_uav_init_locs()
        }
        self.uav_init_locs = self.locs['uav']
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
        kpis = self.get_drates_and_link_assignment_greedy()
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
            new_kpis = self.get_drates_and_link_assignment_greedy()

            # Calculate rewards for each agent
            self.rewards = self.get_rewards(new_kpis)

            # Must be updated after calculating rewards: update .infos['global']
            # self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
            self.infos['global'] = new_kpis

            # Check truncation conditions (overwrites termination conditions)
            self.truncate = self.n_steps >= self.max_cycles
            self.truncations = {agent: self.truncate for agent in self.agents}

            # Check termination conditions
            self.terminations = {agent: self.terminate for agent in self.agents}

            self.n_steps += 1
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()

    def render(self):
        pass

    def close(self):
        '''Close any resources that should be released.'''
        pass

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

        # Option 4: UAV-BS see users with unsatisfied data rates from the mBS within 700m
        drates_mbs = self.infos['global']['drates_map'][0]
        drates_uav_alt = self.infos['global']['drates_map'][1:, :].copy()
        drates_uav_alt = np.delete(drates_uav_alt, i, 0)
        drates_uav_alt = np.max(drates_uav_alt, axis=0)
        h_dist = get_horizontal_dist(locs_['self'], locs_['user'])
        mask = drates_mbs < self.drate_threshold        # not satisfied with the mBS
        mask &= drates_uav_alt < self.drate_threshold   # not satisfied with other droneBSs
        mask &= h_dist <= 700

        locs_['user'] = locs_['user'][:, mask]

        return gen_hist2d(locs_)

    def state(self):
        '''Return a global view of the environment.'''
        return gen_hist2d(self.locs)

    def gen_mbs_loc(self):
        assert self.n_mbss == 1, "Currently support one MBS only"
        return np.array([self.bound, self.bound]).reshape(2, -1)

    def gen_uav_init_locs(self):
        '''Generate initial locations of all drroneBSs. Output shape = (2, n_uavs).'''
        xlocs = self.np_random.uniform(-self.bound, self.bound, self.n_uavs)
        ylocs = self.np_random.uniform(-self.bound, self.bound, self.n_uavs)
        return np.round(np.asarray([xlocs, ylocs]))

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
        locs = np.clip(locs, -self.bound, self.bound)

        return locs

    def move_uav(self, uav: str, action: np.ndarray):
        '''Update the location of the corresponding uav given an action'''
        agent_id = self.agent_name_mapping[uav]
        direction = self._action_to_direction[action]       # shape=(2,)
        next_loc = self.locs['uav'][:, agent_id] + direction * self.uav_velocity    # shape=(2,)
        next_loc = np.clip(next_loc, -self.bound, self.bound).flatten()
        self.locs['uav'][:, agent_id] = next_loc

    def get_snr_macrobs_db(
        self,
        h_dist: np.ndarray,         # horizontal distance (m) to the mBS, shape=(n_users,)
        logF_db: np.float_ = 2,     # std var for the log-normally distributed shadowing
        noise_db: np.float_ = -90,  # total noise power in dBm
        pTx_dBm: np.float_ = 46,    # transmit power in dBm of the macro BS
        n_samples: np.float_ = 250  # no. of samples/s for SNR w/ coherent time ~40ms @fc=2GHz, v=15m/s
    ) -> Tuple[Any]:     # snr [dB] for each user, shape=(n_users)
        '''Calculate the path loss [dB] and SNR [dB] for each user via the macro BS's link.

        References:
        - https://www.arib.or.jp/english/html/overview/doc/STD-T63v9_20/5_Appendix/Rel5/25/25942-530.pdf
        - https://en.wikipedia.org/wiki/Coherence_time_(communications_systems) (coherent time)
        '''
        h_dist = np.maximum(h_dist, 1)          # at leat 1 m (the reference dist), avoid log(0)
        p_loss_mean_db = 128.1 + 37.6 * np.log10(h_dist / 1e3)
        fading_db = self.np_random.normal(
            loc=0, scale=logF_db, size=(len(h_dist), n_samples)
        ).mean(axis=-1)                 # considering coherent time
        p_loss_db = p_loss_mean_db + fading_db
        snr_mean_db = pTx_dBm - p_loss_mean_db - noise_db
        snr_db = pTx_dBm - p_loss_db - noise_db
        return (snr_db, p_loss_db, snr_mean_db)

    def get_snr_uavbs_db(
        self,
        h_dist_m: np.ndarray,           # horizontal distance to the drone BS, shape=(n_users,)
        flying_alt: np.float_ = 120,    # flying altitude of the drone BS
        ref_pw_db: np.float_ = -47,     # reference signal power [dB] at d0=1m and fc=5.8GHz
        noise_db: np.float_ = -90,      # total noise power in dBm
        pTx_dBm: np.float_ = 30,        # transmit power in dBm of the drone BS
        kappa: np.float_ = 50,          # coefficient for the Rician channel effect
        p_loss_coeff: np.float_ = 2.7,  # path loss's coefficient
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
        psi_rician = np.sqrt(kappa / (1 + kappa)) \
            + np.sqrt(1 / (1 + kappa)) * self.np_random.normal(size=(len(h_dist_m), n_samples))
        psi_rician = np.mean(psi_rician, axis=-1)   # considering coherent time
        snr_db = pTx_dBm + ref_pw_db + to_dB(psi_rician**2) - p_loss_db - noise_db
        snr_mean_db = pTx_dBm + ref_pw_db - p_loss_db - noise_db

        return (snr_db, p_loss_db, snr_mean_db)

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
        drates_map = np.zeros(shape=(self.n_mbss + self.n_uavs, self.n_users))
        for i in range(self.n_mbss + self.n_uavs):
            h_dist_ = get_horizontal_dist(bs_locs[:, i], user_locs)
            if i < self.n_mbss:
                snr_ = self.get_snr_macrobs_db(h_dist_)[0]
            else:
                snr_ = self.get_snr_uavbs_db(h_dist_)[0]
            drates_map[i, :] = get_drate_bps(snr_)

        # (V1.0) User association: assign users to mBS/droneBS with the strongest signal
        # drates = np.max(drates_map, axis=0)
        # bs_mapping = np.argmax(drates_map, axis=0)

        # V1.1: only assign to droneBS if the mBS's signal is not stronog enough
        drates_mbs = drates_map[:self.n_mbss, :]
        drates = np.max(drates_mbs, axis=0)
        bs_mapping = np.argmax(drates_mbs, axis=0)
        drates_uav = drates_map[self.n_mbss:, :]
        mask1 = drates < self.drate_threshold
        mask2 = drates < np.max(drates_uav, axis=0)
        mask = mask1 & mask2
        drates[mask] = np.max(drates_uav[:, mask], axis=0)
        bs_mapping[mask] = self.n_mbss + np.argmax(drates_uav[:, mask], axis=0)

        # For tracking KPIs
        drate_avg = drates.mean()
        n_satisfied = np.sum(drates >= self.drate_threshold)
        avg_drates_by_uavbs = []
        n_users_by_uavbs = []
        for i in range(self.n_uavs):
            rs = drates[bs_mapping == i + self.n_mbss]
            n_users_by_uavbs.append(rs.size)
            if rs.size == 0:        # no users are assigned to this drone BS
                avg_drates_by_uavbs.append(0)
            else:
                avg_drates_by_uavbs.append(rs.mean())
        kpis = {
            'drates': drates,
            'drates_map': drates_map,
            'bs_mapping': bs_mapping,
            'drate_avg': drate_avg,
            'n_satisfied': n_satisfied,
            'avg_drates_by_uavbs': np.asarray(avg_drates_by_uavbs),
            'n_users_by_uavbs': np.asarray(n_users_by_uavbs)
        }

        return kpis

    def get_global_reward(self, new_kpis: Dict[str, np.ndarray]) -> float:
        old_kpis = self.infos['global']

        # if new_kpis['n_satisfied'] > old_kpis['n_satisfied']:
        #     n_satisfied_score = 1
        # elif new_kpis['n_satisfied'] < old_kpis['n_satisfied']:
        #     n_satisfied_score = -1
        # else:
        #     n_satisfied_score = 0

        # if new_kpis['drate_avg'] > old_kpis['drate_avg']:
        #     drate_score = 1
        # elif new_kpis['drate_avg'] < old_kpis['drate_avg']:
        #     drate_score = -1
        # else:
        #     drate_score = 0

        # return (1 - self.drate_rw_ratio) * n_satisfied_score\
        #     + self.drate_rw_ratio * drate_score

        n_satisfied_score = new_kpis['n_satisfied'] - old_kpis['n_satisfied']

        return n_satisfied_score

    def get_local_rewards(self, new_kpis: Dict[str, np.ndarray]) -> np.ndarray:
        old_kpis = self.infos['global']

        # n_users_scores = np.zeros(self.n_uavs)
        # for i in range(self.n_uavs):
        #     if new_kpis['n_users_by_uavbs'][i] > old_kpis['n_users_by_uavbs'][i]:
        #         n_users_scores[i] = 1
        #     elif new_kpis['n_users_by_uavbs'][i] < old_kpis['n_users_by_uavbs'][i]:
        #         n_users_scores[i] = -1
        #     else:
        #         n_users_scores[i] = 0

        # drate_scores = np.zeros(self.n_uavs)
        #     if new_kpis['avg_drates_by_uavbs'][i] > old_kpis['avg_drates_by_uavbs'][i]:
        #         drate_scores[i] = 1
        #     elif new_kpis['avg_drates_by_uavbs'][i] < old_kpis['avg_drates_by_uavbs'][i]:
        #         drate_scores[i] = -1
        #     else:
        #         drate_scores[i] = 0

        # return (1 - self.drate_rw_ratio) * n_users_scores\
        #     + self.drate_rw_ratio * drate_scores

        return new_kpis['n_users_by_uavbs'] - old_kpis['n_users_by_uavbs']

    def get_rewards(self, new_kpis: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        global_rewards = self.get_global_reward(new_kpis)
        local_rewards = self.get_local_rewards(new_kpis)
        rewards = (1 - self.local_rw_ratio) * global_rewards\
            + self.local_rw_ratio * local_rewards

        return dict(zip(self.agents, rewards))

    def check_locations_in_bound(self):
        for val in self.locs.values():
            assert np.all(val <= self.bound)
            assert np.all(val >= -self.bound)
