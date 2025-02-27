import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from grad_fit_k1_shape import k1_joint_names, k1_joint_names_augment, k1_joint_pick,\
          smpl_joint_pick, k1_joint_pick_idx, smpl_joint_pick_idx, joint_num

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES, 
)
import joblib
from phc.utils.rotation_conversions import axis_angle_to_matrix
from phc.utils.torch_k1_humanoid_batch import Humanoid_Batch
from phc.utils.torch_k1_humanoid_batch import K1_ROTATION_AXIS
from torch.autograd import Variable
from tqdm import tqdm
import argparse


def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return 
    framerate = entry_data['mocap_framerate']


    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=str, default="/home/admin-1/workspace/demo_retarget/data/AMASS/AMASS_Complete")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    k1_rotation_axis = K1_ROTATION_AXIS.to(device)

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    smpl_parser_n.to(device)


    shape_new, scale = joblib.load("data/k1/shape_optimized_v1.pkl")
    shape_new = shape_new.to(device)


    amass_root = args.amass_root
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}
    print(key_name_to_pkls.keys())
    key_name_to_pkls = {k: v for k, v in key_name_to_pkls.items() if "0-HumanEva (1)_HumanEva_S2_Walking_1_poses" in k}
    print(key_name_to_pkls.keys())
    
    if len(key_name_to_pkls) == 0:
        raise ValueError(f"No motion files found in {amass_root}")

    k1_fk = Humanoid_Batch(device = device)
    data_dump = {}
    pbar = tqdm(key_name_to_pkls.keys())
    print(f"Total number of data: {len(key_name_to_pkls)}")
    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        skip = int(amass_data['fps']//30)
        trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(device)
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(np.concatenate((amass_data['pose_aa'][::skip, :66], np.zeros((N, 6))), axis = -1)).float().to(device)


        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset

        pose_aa_k1 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], joint_num+3, axis = 2), N, axis = 1)
        pose_aa_k1[..., 0, :] = (sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()
        pose_aa_k1 = torch.from_numpy(pose_aa_k1).float().to(device)
        gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)

        dof_pos = torch.zeros((1, N, joint_num, 1)).to(device)

        dof_pos_new = Variable(dof_pos, requires_grad=True)
        optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=150)

        for iteration in range(500):
            # get verts and joints of the current pose from smpl
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            
            pose_aa_k1_new = torch.cat([gt_root_rot[None, :, None], \
                                        k1_rotation_axis * dof_pos_new, \
                                        torch.zeros((1, N, 2, 3)).to(device)], axis = 2).to(device)
            
            # get the forward kinematics of the current pose from h1
            fk_return = k1_fk.fk_batch(pose_aa_k1_new, root_trans_offset[None, ])
            # get the difference between the current pose and the smpl pose
            diff = fk_return['global_translation_extend'][:, :, k1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            loss_g = diff.norm(dim = -1).mean() 
            loss = loss_g
            
            pbar.set_description_str(f"{iteration} {loss.item() * 1000}")

            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_pose.step()
            
            dof_pos_new.data.clamp_(k1_fk.joints_range[:, 0, None], k1_fk.joints_range[:, 1, None])
            
        dof_pos_new.data.clamp_(k1_fk.joints_range[:, 0, None], k1_fk.joints_range[:, 1, None])
        pose_aa_k1_new = torch.cat([gt_root_rot[None, :, None], k1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2)
        fk_return = k1_fk.fk_batch(pose_aa_k1_new, root_trans_offset[None, ])

        root_trans_offset_dump = root_trans_offset.clone()

        root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08
        
        data_dump[data_key]={
                "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
                "pose_aa": pose_aa_k1_new.squeeze().cpu().detach().numpy(),   
                "dof": dof_pos_new.squeeze().detach().cpu().numpy(), 
                "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
                "fps": 30
                }
        
        print(f"dumping {data_key} for testing, remove the line if you want to process all data")
        joblib.dump(data_dump, "data/k1/test.pkl")
    
        
    joblib.dump(data_dump, "data/k1/amass_all.pkl")
