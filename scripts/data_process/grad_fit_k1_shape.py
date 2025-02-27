import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from phc.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)

import joblib
import torch
import torch.nn.functional as F
import math
from phc.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
from tqdm.notebook import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from phc.utils.torch_k1_humanoid_batch import Humanoid_Batch, K1_ROTATION_AXIS



k1_fk = Humanoid_Batch(extend_head=True, 
                       extend_hand=True,
                       mjcf_file="/home/admin-1/workspace/kepler_ws/resources/Robots/Kepler/K1/mjcf/mjmodel.xml") # load forward kinematics model

print(f"nodes of robot: {k1_fk.model_names}")
print(f"joints of robot: {k1_fk.model_names}")

k1_joint_names = ['pelvis', 'left_hip_roll', 'left_hip_yaw', 'left_hip_pitch', 
 'left_knee_pitch', 'left_ankle_pitch', 'left_ankle_link', 'right_hip_roll', 
 'right_hip_yaw', 'right_hip_pitch', 'right_knee_pitch', 'right_ankle_pitch', 
 'right_ankle_link', 'waist_roll', 'torso_link', 'left_shoulder_pitch', 
 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow_pitch', 'left_elbow_yaw', 
 'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
   'right_elbow_pitch', 'right_elbow_yaw']

#### Define corresonpdances between h1 and smpl joints
k1_joint_names_augment = k1_joint_names + ["left_hand_link", "right_hand_link", "head_link"]

print("Model nodes: ", k1_fk.model_names)
k1_joint_pick =  ['pelvis', 
                  'left_hip_roll', 'left_knee_pitch', 'left_ankle_link', 
                  'right_hip_roll', 'right_knee_pitch', 'right_ankle_link', 
                  'left_shoulder_pitch',  'left_elbow_pitch', 'left_hand_link',
                  'right_shoulder_pitch',  'right_elbow_pitch', 'right_hand_link',
                  'head_link']
print(f"The number of K1 joint pick: {len(k1_joint_pick)}")
k1_joint_pick_idx = [k1_joint_names_augment.index(j) for j in k1_joint_pick]

smpl_joint_pick = ["Pelvis", "L_Hip",  "L_Knee", "L_Ankle",  "R_Hip", "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand", "Head"]
print(f"The number of K1 smpl joint pick: {len(smpl_joint_pick)}")
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
joint_num = k1_fk.joints_range.shape[0]


#### Preparing fitting varialbes
if __name__ == "__main__":
    device = torch.device("cpu")
    print(joint_num)
    # Initialize K1 pose with identity rotation repeated for each joint
    pose_aa_k1 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], joint_num+3, axis=2), 1, axis=1)
    pose_aa_k1 = torch.from_numpy(pose_aa_k1).float()

    # Initialize degrees of freedom positions and concatenate with H1 rotation axis
    dof_pos = torch.zeros((1, joint_num))
    pose_aa_k1 = torch.cat([torch.zeros((1, 1, 3)),
                         K1_ROTATION_AXIS * dof_pos[..., None], 
                        torch.zeros((1, 2, 3))], axis=1)

    # Initialize root translation
    root_trans = torch.zeros((1, 1, 3))

    # Prepare SMPL default pose for K1
    pose_aa_stand = np.zeros((1, 72)) # why (23+1)*3?
    rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
    pose_aa_stand[:, :3] = rotvec
    pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)# there are 24 keypoints (including root) in smpl human model
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Shoulder')] = sRot.from_euler("xyz", [0, 0, -np.pi/2], degrees=False).as_rotvec()
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Shoulder')] = sRot.from_euler("xyz", [0, 0, np.pi/2], degrees=False).as_rotvec()
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Elbow')] = sRot.from_euler("xyz", [0, -np.pi/2, 0], degrees=False).as_rotvec()
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Elbow')] = sRot.from_euler("xyz", [0, np.pi/2, 0], degrees=False).as_rotvec()
    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))# why 72?

    # Initialize SMPL parser with neutral gender model
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

    # Shape fitting initialization
    trans = torch.zeros([1, 3])
    beta = torch.zeros([1, 10])
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta, trans)
    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset

    # Forward kinematics for K1
    fk_return = k1_fk.fk_batch(pose_aa_k1[None, ], root_trans_offset[None, 0:1])

    # Initialize shape and scale variables for optimization
    shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    optimizer_shape = torch.optim.Adam([shape_new, scale], lr=0.2)

    # Optimization loop for shape fitting
    for iteration in range(1000):
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
        root_pos = joints[:, 0]# pelvis location
        joints = (joints - joints[:, 0]) * scale + root_pos
        # len(k1_joint_pich_idx) is 14
        diff = fk_return.global_translation_extend[:, :, k1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
        loss_g = diff.norm(dim=-1).mean()
        loss = loss_g
        if iteration % 100 == 0:
            print(iteration, loss.item() * 1000)

        optimizer_shape.zero_grad()
        loss.backward()
        optimizer_shape.step()

    # Save the optimized shape and scale
    os.makedirs("data/k1", exist_ok=True)
    joblib.dump((shape_new.detach(), scale), "data/k1/shape_optimized_v1.pkl")
    print(f"shape fitted and saved to data/k1/shape_optimized_v1.pkl")
