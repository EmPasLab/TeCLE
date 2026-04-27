from __future__ import annotations


try:
    from minigrid.envs.wfc.wfcenv import WFCEnv
except ImportError:

    class WFCEnv:

        def __init__(self, *args, **kwargs):
            from gymnasium.error import DependencyNotInstalled

            raise DependencyNotInstalled(
                'WFC dependencies are missing, please run `pip install "minigrid[wfc]"`'
            )
