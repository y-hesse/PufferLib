'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.quoridor import binding

class Quoridor(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=11, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=-2, high=1,
            shape=(((10*2-1) * (10*2-1) + 3),), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(184)

        self.render_mode = render_mode
        self.num_agents = num_envs*2
        self.log_interval = log_interval

        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i*2:(i+1)*2],
                self.actions[i*2:(i+1)*2],
                self.rewards[i*2:(i+1)*2],
                self.terminals[i*2:(i+1)*2],
                self.truncations[i*2:(i+1)*2],
                seed)
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_win_balance(env, episodes=1000):
    env.reset()
    wins = {1: 0, 2: 0, -1: 0}
    step_limit = 200
    cache_size = 1024
    actions = np.random.randint(0, env.single_action_space.n, (cache_size, env.num_agents))

    i = 0
    total_games = 0

    while total_games < episodes:
        obs, rewards, terminals, truncations, info = env.step(actions[i % cache_size])
        i += 1

        for b in range(env.num_agents // 2):
            if terminals[b * 2]:  # One env is done
                total_games += 1
                winner = -1
                if rewards[b * 2] == 1.0:
                    winner = 1
                elif rewards[b * 2 + 1] == 1.0:
                    winner = 2
                wins[winner] += 1

    print("\nFinal results after", episodes, "games:")
    print("Player 1 wins:", wins[1])
    print("Player 2 wins:", wins[2])
    print("Ties/timeouts:", wins[-1])
    winrate_1 = wins[1] / episodes * 100
    winrate_2 = wins[2] / episodes * 100
    print(f"Winrate P1: {winrate_1:.2f}%, Winrate P2: {winrate_2:.2f}%")



if __name__ == '__main__':
    env = Quoridor(num_envs=64)
    test_win_balance(env, episodes=1000000)
