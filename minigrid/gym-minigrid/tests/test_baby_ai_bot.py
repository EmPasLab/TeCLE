from __future__ import annotations

import gymnasium as gym
import pytest

from minigrid.utils.baby_ai_bot import BabyAIBot


broken_bonus_envs = {
    "BabyAI-PutNextS5N2Carrying-v0",
    "BabyAI-PutNextS6N3Carrying-v0",
    "BabyAI-PutNextS7N4Carrying-v0",
    "BabyAI-KeyInBox-v0",
}


babyai_envs = []
for k_i in gym.envs.registry.keys():
    if k_i.split("-")[0] == "BabyAI":
        if k_i not in broken_bonus_envs:
            babyai_envs.append(k_i)


@pytest.mark.parametrize("env_id", babyai_envs)
def test_bot(env_id):

    env = gym.make(env_id)


    curr_seed = 0

    num_steps = 240
    terminated = False
    while not terminated:
        env.reset(seed=curr_seed)


        expert = BabyAIBot(env)

        last_action = None
        for _step in range(num_steps):
            action = expert.replan(last_action)
            obs, reward, terminated, truncated, info = env.step(action)
            last_action = action
            env.render()

            if terminated:
                break


        curr_seed += 1

    env.close()
