import random
import torch
import numpy as np
import hydra
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from fw_flightcontrol.agents import ppo
from fw_flightcontrol.utils.train_utils import make_env
from fw_jsbgym.trim.trim_point import TrimPoint
# from fw_jsbgym.envs.tasks.attitude_control.ac_bohn_nova import ACBohnNoVaIErrTask
from fw_jsbgym.envs.tasks.waypoint_tracking.wp_tracking import WaypointTracking



@hydra.main(config_name='tdmpc2_default', config_path='config')
def main(cfg: DictConfig):
    env = WaypointTracking(cfg_env=cfg.env, telemetry_file='telemetry/telemetry.csv', render_mode='none')
    env.init()


if __name__ == "__main__":
    main()
