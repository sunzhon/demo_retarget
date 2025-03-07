import torch
import numpy as np
import phc.utils.rotation_conversions as tRot
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters
import smpl_sim.poselib.core.rotation3d as pRot

K1_ROTATION_AXIS = torch.tensor(
    [
        [
            [1, 0, 0],  # l_hip_roll
            [0, 0, 1],  # l_hip_yaw
            [0, 1, 0],  # l_hip_pitch
            [0, 1, 0],  # l_knee pitch
            [0, 1, 0],  # l_ankle picth
            [1, 0, 0],  # l_ankle roll
            [1, 0, 0],  # r_hip_roll
            [0, 0, 1],  # r_hip_yaw
            [0, 1, 0],  # r_hip_pitch
            [0, 1, 0],  # r_knee pitch
            [1, 0, 0],  # r_ankle pitch
            [0, 1, 0],  # r_ankle roll
            [1, 0, 0],  # torso roll
            [0, 0, 1],  # torso yaw
            [0, 1, 0],  # l_shoulder_pitch
            [1, 0, 0],  # l_shoulder_roll
            [0, 0, 1],  # l_shoulder_yaw
            [0, 1, 0],  # l_elbow pitch
            [0, 0, 1],  # l_elbow yaw
            [0, 1, 0],  # r_shoulder_pitch
            [1, 0, 0],  # r_shoulder_roll
            [0, 0, 1],  # r_shoulder_yaw
            [0, 1, 0],  # r_elbow pitch
            [0, 0, 1],  # r_elbow yaw
        ]
    ]
)


class Humanoid_Batch:

    def __init__(
        self,
        mjcf_file=f"/home/admin-1/workspace/kepler_ws/resources/Robots/Kepler/K1/mjcf/mjmodel.xml",
        device=torch.device("cpu"),
    ):
        self.device=device
        self.mjcf_data = mjcf_data = self.from_mjcf(mjcf_file)
        #
        self._parents = mjcf_data["parent_indices"]
        self.model_names = mjcf_data["node_names"]
        self._offsets = mjcf_data["local_translation"][None,].to(
            device
        )  # local translation of each links
        self._local_rotation = mjcf_data["local_rotation"][None,].to(
            device
        )  # local orentations of each links

        self.joints_range = mjcf_data["joints_range"].to(device)
        self.joints_name = mjcf_data["joints_name"]
        self._local_rotation_mat = tRot.quaternion_to_matrix(
            self._local_rotation
        ).float()  # w, x, y ,z

    def from_mjcf(self, path):
        # function from Poselib:
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")

        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")

        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        joints_name = []

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(
                xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" "
            )
            quat = np.fromstring(
                xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" "
            )
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint")
            for joint in all_joints:
                if not joint.attrib.get("range") is None:
                    joints_range.append(
                        np.fromstring(joint.attrib.get("range"), dtype=float, sep=" ")
                    )
                    joints_name.append(joint.attrib.get("name"))

            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            return node_index

        _add_xml_node(xml_body_root, -1, 0)
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(
                np.array(parent_indices, dtype=np.int32)
            ),
            "local_translation": torch.from_numpy(
                np.array(local_translation, dtype=np.float32)
            ),
            "local_rotation": torch.from_numpy(
                np.array(local_rotation, dtype=np.float32)
            ),
            "joints_range": torch.from_numpy(np.array(joints_range)),
            "joints_name": joints_name,
        }

    def fk_batch(self, rotation, trans, return_full=False, dt=1 / 30):
        """
        pose [B, seq_len, link_num, 3], orientation
        trans [B, seq_len, 3],

        
        """
        B, seq_len = rotation.shape[:2]
        rotation = rotation[
            ..., : len(self._parents), :
        ]  # H1 fitted joints might have extra joints

        quat = tRot.axis_angle_to_quaternion(rotation)
        pose_mat = tRot.quaternion_to_matrix(quat)


        # Compute forward kinematics
        wbody_pos, wbody_mat = self.forward_kinematics_batch(
            pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans
        )

        return_dict = EasyDict()

        wbody_rot = tRot.wxyz_to_xyzw(tRot.matrix_to_quaternion(wbody_mat))

        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot

        return return_dict

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """

        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []

        expanded_offsets = (
            self._offsets[:, None].expand(B, seq_len, J, 3).to(self.device)
        )
        # print(expanded_offsets.shape, J)

        for i in range(J):# link number
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (
                    torch.matmul(
                        rotations_world[self._parents[i]][:, :, 0],
                        expanded_offsets[:, :, i, :, None],
                    ).squeeze(-1)
                    + positions_world[self._parents[i]]
                )
                rot_mat = torch.matmul(
                    rotations_world[self._parents[i]],
                    torch.matmul(
                        self._local_rotation_mat[:, (i) : (i + 1)],
                        rotations[:, :, (i - 1) : i, :],
                    ),
                )
                positions_world.append(jpos)
                rotations_world.append(rot_mat)

        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world

    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if guassian_filter:
            velocity = torch.from_numpy(
                filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")
            ).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)

        return velocity

    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        # assume the second last dimension is the time axis
        diff_quat_data = pRot.quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = pRot.quat_mul_norm(
            r[..., 1:, :, :], pRot.quat_inverse(r[..., :-1, :, :])
        )
        diff_angle, diff_axis = pRot.quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(
                filters.gaussian_filter1d(
                    angular_velocity.numpy(), 2, axis=-3, mode="nearest"
                ),
            )
        return angular_velocity


if __name__ == "__main__":
    humanoid = Humanoid_Batch()
    print(f"node num: {len(humanoid.model_names)}, {humanoid.model_names} ")
    print(f"joint num: {len(humanoid.joints_name)}, {humanoid.joints_name}")
    print(f"node idx: {humanoid._parents}")
    print(f"rotation shape: {humanoid._local_rotation.shape}")
    print(f"offset shape: {humanoid._offsets.shape}")
    print(f"joint number: {humanoid.joints_range.shape}")


    root_trans_offset = torch.zeros([1, 3])
    joint_num=24
    dof_pos = torch.zeros((1, joint_num))
    pose_aa_k1 = torch.cat([torch.zeros((1, 1, 3)),
                         K1_ROTATION_AXIS * dof_pos[..., None], 
                        torch.zeros((1, 2, 3))], axis=1)

    fk_return = humanoid.fk_batch(pose_aa_k1[None, ], root_trans_offset[None,])