"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Visualize motion library
"""
import glob
import os
import sys
import pdb
import os.path as osp

import json
import os

sys.path.append(os.getcwd())

import joblib
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from phc.utils.motion_lib_h1 import MotionLibH1
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.flags import flags


flags.test = True
flags.im_eval = True


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# simple asset descriptor for selecting from a list


class AssetDesc:

    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_root = "./"
h1_xml = "resources/robots/h1/h1.xml"
h1_urdf = "resources/robots/h1/urdf/h1.urdf"
asset_descriptors = [
    # AssetDesc(h1_xml, False),
    AssetDesc(h1_urdf, False),
]
sk_tree = SkeletonTree.from_mjcf(h1_xml)

motion_file = "data/h1/test.pkl"
if os.path.exists(motion_file):
    print(f"loading {motion_file}")
else:
    raise ValueError(f"Motion file {motion_file} does not exist! Please run grad_fit_h1.py first.")

# parse arguments
args = gymutil.parse_arguments(description="Joint monkey: Animate degree-of-freedom ranges",
                               custom_parameters=[{
                                   "name": "--asset_id",
                                   "type": int,
                                   "default": 0,
                                   "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)
                               }, {
                                   "name": "--speed_scale",
                                   "type": float,
                                   "default": 1.0,
                                   "help": "Animation speed scale"
                               }, {
                                   "name": "--show_axis",
                                   "action": "store_true",
                                   "help": "Visualize DOF axis"
                               }])

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
    quit()

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

if not args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
# asset_root = "amp/data/assets"
asset_file = asset_descriptors[args.asset_id].file_name

asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
# asset_options.flip_visual_attachments = asset_descriptors[
#     args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True
asset_options.thickness = 0.001
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.armature = 0.01
asset_options.disable_gravity = False
asset_options.flip_visual_attachments = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.collapse_fixed_joints = True


print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)



# set up the env grid
num_envs = 1
num_per_row = 5
spacing = 5
env_lower = gymapi.Vec3(-spacing, spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(0, -10.0, 3)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

num_dofs = gym.get_asset_dof_count(asset)
print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 0.0)
    pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    joint_names = gym.get_actor_joint_names(env,0)
    print(f"[Info] Joint num: {len(joint_names)}")
    print(f"[Info] Joint names: {joint_names}")
    actor_handles.append(actor_handle)


    # set default DOF positions
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)


gym.prepare_sim(sim)



device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))

motion_lib = MotionLibH1(motion_file=motion_file, device=device, 
                         masterfoot_conifg=None, fix_height=False,
                           multi_thread=False, mjcf_file=h1_xml)
num_motions = 1
curr_start = 0
motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False)
motion_keys = motion_lib.curr_motion_keys

current_dof = 0
speeds = np.zeros(num_dofs)

time_step = 0
rigidbody_state = gym.acquire_rigid_body_state_tensor(sim)
rigidbody_state = gymtorch.wrap_tensor(rigidbody_state)
rigidbody_state = rigidbody_state.reshape(num_envs, -1, 13)

actor_root_state = gym.acquire_actor_root_state_tensor(sim)
actor_root_state = gymtorch.wrap_tensor(actor_root_state)

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "previous")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "next")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_G, "add")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "print")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "next_batch")
motion_id = 0
motion_acc = set()




env_ids = torch.arange(num_envs).int().to(args.sim_device)


## Create sphere actors
radius = 0.1
color = gymapi.Vec3(1.0, 0.0, 0.0)
sphere_params = gymapi.AssetOptions()

sphere_asset = gym.create_sphere(sim, radius, sphere_params)


st_collected_data = [] # Tao Sun add this
while not gym.query_viewer_has_closed(viewer):
    # step the physics

    motion_len = motion_lib.get_motion_length(motion_id).item()
    motion_time = time_step % motion_len
    # motion_time = 0
    # import pdb; pdb.set_trace()
    # print(motion_id, motion_time)
    motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]).to(args.compute_device_id), torch.tensor([motion_time]).to(args.compute_device_id))

    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
    if args.show_axis:
        gym.clear_lines(viewer)
        
    gym.clear_lines(viewer)
    gym.refresh_rigid_body_state_tensor(sim)
    # import pdb; pdb.set_trace()
    idx = 0
    for pos_joint in rb_pos[0, 1:]: # idx 0 torso (duplicate with 11)
        sphere_geom1 = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(1, 0.0, 0.0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
        gymutil.draw_lines(sphere_geom1, gym, viewer, envs[0], sphere_pose) 

    key_body_names = ["left_ankle_joint","right_ankle_joint"]
    key_body_ids = [joint_names.index(tmp)+1 for tmp in key_body_names]
    
    key_body_pos = []
    for pos_joint in rb_pos[0, key_body_ids]: # idx 0 torso (duplicate with 11)
        sphere_geom2 = gymutil.WireframeSphereGeometry(0.1, 3, 3, None, color=(0, 0.0, 1.0))
        key_body_pos.append(pos_joint-rb_pos[0,11])
        sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
        gymutil.draw_lines(sphere_geom2, gym, viewer, envs[0], sphere_pose) 
    key_body_pos = torch.cat(key_body_pos)



    root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1).repeat(num_envs, 1)
    
    # gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(env_ids), len(env_ids))

    gym.refresh_actor_root_state_tensor(sim)

    dof_state = torch.stack([dof_pos, dof_vel], dim=-1).squeeze().repeat(num_envs, 1)
    # Debug joint names and its index
    #dof_state[joint_names.index('right_hip_pitch_joint'),0] = 1.57
    #dof_state[joint_names.index('left_hip_pitch_joint'),0] = -1.57
    #dof_state[joint_names.index('right_shoulder_pitch_joint'),0] = -1.57

    gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state), gymtorch.unwrap_tensor(env_ids), len(env_ids))

    # Tao SUn add this
    st_collected_data.append(torch.cat([root_states.squeeze()[2:], dof_pos.squeeze(), dof_vel.squeeze(), key_body_pos], dim=-1).unsqueeze(0).numpy().squeeze().tolist())

    gym.simulate(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.fetch_results(sim, True)
    

    # print((rigidbody_state[None, ] - rigidbody_state[:, None]).sum().abs())
    # print((actor_root_state[None, ] - actor_root_state[:, None]).sum().abs())

    # pose_quat = motion_lib._motion_data['0-ACCAD_Female1Running_c3d_C5 - walk to run_poses']['pose_quat_global']
    # diff = quat_mul(quat_inverse(rb_rot[0, :]), rigidbody_state[0, :, 3:7]); np.set_printoptions(precision=4, suppress=1); print(diff.cpu().numpy()); print(torch_utils.quat_to_angle_axis(diff)[0])

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    # time_step += 1/5
    time_step += dt

    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "previous" and evt.value > 0:
            motion_id = (motion_id - 1) % num_motions
            print(f"Motion ID: {motion_id}. Motion length: {motion_len:.3f}. Motion Name: {motion_keys[motion_id]}")
        elif evt.action == "next" and evt.value > 0:
            motion_id = (motion_id + 1) % num_motions
            print(f"Motion ID: {motion_id}. Motion length: {motion_len:.3f}. Motion Name: {motion_keys[motion_id]}")
        elif evt.action == "add" and evt.value > 0:
            motion_acc.add(motion_keys[motion_id])
            print(f"Adding motion {motion_keys[motion_id]}")
        elif evt.action == "print" and evt.value > 0:
            print(motion_acc)
        elif evt.action == "next_batch" and evt.value > 0:
            curr_start += num_motions
            motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
            motion_keys = motion_lib.curr_motion_keys
            print(f"Next batch {curr_start}")

            ############################################################################
            # Tao Sun add this
            data_field = ["root_pos_z", 
                          "root_rot_x", "root_rot_y", "root_rot_z", "root_rot_w",
                          "root_vel_x", "root_vel_y", "root_vel_z",
                           "root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z"] + \
                            [key + "_dof_pos" for key in joint_names] +\
                            [key + "_dof_vel" for key in joint_names] +\
                            [k1+ "_"+ k2 for k1 in key_body_names for k2 in ["x", "y", "z"]]
            # Assuming saved_data is a dictionary
            saved_data = {
                "LoopMode": "Wrap",
                "FrameDuration": 0.021,
                "EnableCycleOffsetPosition": True,
                "EnableCycleOffsetRotation": True,
                "MotionWeight": 0.5,
                "Fields": data_field,
                "Frames": st_collected_data
            }
            print(len(data_field))
            assert len(st_collected_data[0]) == len(data_field), f"Data field length {len(data_field)} does not match collected data length {len(st_collected_data[0])}"
            data_idx = data_idx + 1 if "data_idx" in locals() else 0
            # Define the path to the output file
            output_file_path = os.path.join(os.path.dirname(os.path.abspath(motion_file)), str(data_idx)+"_saved_data.txt")
            # Save the dictionary to a text file in JSON format
            with open(output_file_path, 'w') as file:
                json.dump(saved_data, file, indent=4)
            print(f"Saved data to {output_file_path}")
            st_collected_data = [] # Tao Sun add this
            ############################################################################

            print("End of motion library")
            break

        time_step = 0
print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
