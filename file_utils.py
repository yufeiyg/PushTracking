"""File utilities for interfacing with dairlib repo.  Requires ci_mpc_utils repo
to be installed at same directory as dairlib repo."""

import numpy as np
import os
import os.path as op
import yaml


HEAD_DIR = op.abspath(op.dirname(op.dirname(__file__)))
DAIRLIB_DIR = op.join(HEAD_DIR, 'dairlib')

assert op.isdir(DAIRLIB_DIR), f'Did not find dairlib at {DAIRLIB_DIR}'


### Directories ###
def dairlib_dir():
    return DAIRLIB_DIR

def cimpc_dir():
    return op.join(HEAD_DIR, 'ci_mpc_utils')

def example_dir():
    return op.join(dairlib_dir(), 'examples/sampling_c3')

def subexample_dir(system: str):
    assert system is not None, f'Need to provide sub-example for sampling-' + \
        f'based C3.'
    sub_dir = op.join(example_dir(), system)
    assert op.exists(sub_dir), f'Did not find {sub_dir}'
    return sub_dir

def tmp_dir():
    tmp_dir = op.join(cimpc_dir(), 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir

def calibration_dir():
    cam_cal_dir = op.join(cimpc_dir(), 'calibrations')
    os.makedirs(cam_cal_dir, exist_ok=True)
    return cam_cal_dir

def calibration_subdir(subdir: str):
    cal_sub_dir = op.join(calibration_dir(), subdir)
    os.makedirs(cal_sub_dir, exist_ok=True)
    return cal_sub_dir

def urdf_dir():
    return op.join(example_dir(), 'urdf')

def vis_urdf_dir():
    return op.join(cimpc_dir(), 'urdfs')


### Add dairlib lcmtypes to path ###
"""This is required for importing `dairlib` in a python file to access dairlib
LCM types, e.g. dairlib.lcmt_c3_state."""
def add_dair_lcmtypes_to_path():
    import sys
    lcmtypes_dir = op.join(dairlib_dir(), 'bazel-bin', 'lcmtypes')
    sys.path.append(lcmtypes_dir)


### Filepaths ###
def jack_urdf_path():
    return op.join(urdf_dir(), 'jack.sdf')

def ground_urdf_path():
    return op.join(urdf_dir(), 'ground.urdf')

def end_effector_urdf_path():
    return op.join(urdf_dir(), 'end_effector_full.urdf')

def jack_with_triad_urdf_path():
    return op.join(vis_urdf_dir(), 'jack_with_triad.urdf')

def goal_triad_urdf_path():
    return op.join(vis_urdf_dir(), 'goal_triad.urdf')

def push_t_urdf_path():
    return op.join(vis_urdf_dir(), 'T_vertical_obj.urdf')

def goal_push_t_urdf_path():
    return op.join(vis_urdf_dir(), 'T_vertical_obj_green.urdf')

def camera_urdf_path(first: bool = True):
    filename = 'camera_model.urdf' if first else 'camera_model_2.urdf'
    return op.join(vis_urdf_dir(), filename)


### Load parameters from yaml files ###
def load_franka_sim_params(system: str):
    sub_dir = subexample_dir(system)
    yaml_file = op.join(sub_dir, 'parameters', 'franka_sim_params.yaml')
    with open(yaml_file) as f:
        sim_params = yaml.load(f, Loader=yaml.FullLoader)

    return sim_params

def load_p_world_to_franka(system: str):
    sim_params = load_franka_sim_params(system)
    return np.array(sim_params['p_world_to_franka'])

def load_p_franka_to_ground(system: str):
    sim_params = load_franka_sim_params(system)
    return np.array(sim_params['p_franka_to_ground'])
