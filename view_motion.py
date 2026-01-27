import numpy as np
import os, argparse

import torch
import mujoco, mujoco_viewer
from tqdm import tqdm

from utils.motion_lib import MotionLib


class MotionViewEnv:
    def __init__(self, motion_file, robot_type="g1", device="cuda", record_video=False):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        
        self.motion_file = motion_file  # Store full path for format detection
        self.motion_file_name = os.path.basename(motion_file)
        self.is_npz = motion_file.endswith('.npz')  # NPZ uses wxyz, PKL uses xyzw
        
        self.motion_lib = MotionLib(motion_file=motion_file, device=device)
        self.motion_ids = torch.tensor([0], dtype=torch.long, device=device)
        self.motion_len = self.motion_lib.get_motion_length(self.motion_ids)
        
        model_path_root = "assets/robots"
        
        if robot_type == "g1":
            model_path = os.path.join(model_path_root, "g1/g1.xml")
        else:
            raise NotImplementedError("Robot type not supported")
        
        if self.record_video:
            self.sim_duration = self.motion_len.item()
        else:
            self.sim_duration = 10*self.motion_len.item()
        
        self.fps = self.motion_lib.get_motion_fps(self.motion_ids).item()
        print(f"Motion FPS: {self.fps}")
        self.sim_dt = 1.0 / self.fps
        self.sim_decimation = 1
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
        if self.record_video:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen')
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.distance = 5.0
        
    def run(self):
        if self.record_video:
            import imageio
            # video_name: motion_file_name without extension + .mp4
            video_name = f"{os.path.splitext(self.motion_file_name)[0]}.mp4"
            mp4_writer = imageio.get_writer(video_name, fps=self.fps)
            print(f"Recording video to {video_name} with FPS {self.fps}")
        
        for i in tqdm(range(int(self.sim_duration / self.control_dt)), desc="Running simulation..."):
            curr_time = i * self.control_dt
            motion_time = torch.tensor([curr_time], dtype=torch.float, device=self.device) % self.motion_len
            root_pos, root_rot, _, _, dof_pos, _ = self.motion_lib.calc_motion_frame(self.motion_ids, motion_time)
            
            root_pos_np = root_pos[0].cpu().numpy()
            root_rot_np = root_rot[0].cpu().numpy()
            
            # MuJoCo expects quaternion in wxyz format
            # NPZ files (Isaac Lab): already wxyz format, no conversion needed
            # PKL files: stored in xyzw format, need to convert to wxyz
            if self.is_npz:
                mujoco_quat = root_rot_np  # Already wxyz
            else:
                mujoco_quat = root_rot_np[[3, 0, 1, 2]]  # xyzw -> wxyz
            
            self.data.qpos[:3] = root_pos_np
            self.data.qpos[3:7] = mujoco_quat
            
            dof_pos_val = dof_pos[0].cpu().numpy()
            if dof_pos_val.shape[0] == 29:
                # Remap from Isaac Lab order (29D) to MuJoCo order (23D)
                # Isaac Lab order (29 joints, interleaved left-right):
                # 0: left_hip_pitch,  1: right_hip_pitch,  2: waist_yaw,
                # 3: left_hip_roll,   4: right_hip_roll,   5: waist_roll,
                # 6: left_hip_yaw,    7: right_hip_yaw,    8: waist_pitch,
                # 9: left_knee,      10: right_knee,
                # 11: left_shoulder_pitch, 12: right_shoulder_pitch,
                # 13: left_ankle_pitch,    14: right_ankle_pitch,
                # 15: left_shoulder_roll,  16: right_shoulder_roll,
                # 17: left_ankle_roll,     18: right_ankle_roll,
                # 19: left_shoulder_yaw,   20: right_shoulder_yaw,
                # 21: left_elbow,          22: right_elbow,
                # 23: left_wrist_roll,     24: right_wrist_roll,
                # 25: left_wrist_pitch,    26: right_wrist_pitch,
                # 27: left_wrist_yaw,      28: right_wrist_yaw
                #
                # MuJoCo order (23 joints, grouped by body part):
                # 0: left_hip_pitch, 1: left_hip_roll, 2: left_hip_yaw,
                # 3: left_knee, 4: left_ankle_pitch, 5: left_ankle_roll,
                # 6: right_hip_pitch, 7: right_hip_roll, 8: right_hip_yaw,
                # 9: right_knee, 10: right_ankle_pitch, 11: right_ankle_roll,
                # 12: waist_yaw, 13: waist_roll, 14: waist_pitch,
                # 15: left_shoulder_pitch, 16: left_shoulder_roll, 17: left_shoulder_yaw, 18: left_elbow,
                # 19: right_shoulder_pitch, 20: right_shoulder_roll, 21: right_shoulder_yaw, 22: right_elbow
                
                # Mapping: mujoco_idx -> isaac_idx
                isaac_to_mujoco = [
                    0,   # mj 0: left_hip_pitch    <- isaac 0
                    3,   # mj 1: left_hip_roll     <- isaac 3
                    6,   # mj 2: left_hip_yaw      <- isaac 6
                    9,   # mj 3: left_knee         <- isaac 9
                    13,  # mj 4: left_ankle_pitch  <- isaac 13
                    17,  # mj 5: left_ankle_roll   <- isaac 17
                    1,   # mj 6: right_hip_pitch   <- isaac 1
                    4,   # mj 7: right_hip_roll    <- isaac 4
                    7,   # mj 8: right_hip_yaw     <- isaac 7
                    10,  # mj 9: right_knee        <- isaac 10
                    14,  # mj 10: right_ankle_pitch <- isaac 14
                    18,  # mj 11: right_ankle_roll  <- isaac 18
                    2,   # mj 12: waist_yaw         <- isaac 2
                    5,   # mj 13: waist_roll        <- isaac 5
                    8,   # mj 14: waist_pitch       <- isaac 8
                    11,  # mj 15: left_shoulder_pitch  <- isaac 11
                    15,  # mj 16: left_shoulder_roll   <- isaac 15
                    19,  # mj 17: left_shoulder_yaw    <- isaac 19
                    21,  # mj 18: left_elbow           <- isaac 21
                    12,  # mj 19: right_shoulder_pitch <- isaac 12
                    16,  # mj 20: right_shoulder_roll  <- isaac 16
                    20,  # mj 21: right_shoulder_yaw   <- isaac 20
                    22,  # mj 22: right_elbow          <- isaac 22
                ]
                new_pos = np.zeros(23)
                for mj_idx, isaac_idx in enumerate(isaac_to_mujoco):
                    new_pos[mj_idx] = dof_pos_val[isaac_idx]
                dof_pos_val = new_pos
            elif dof_pos_val.shape[0] != 23:
                 print(f"Warning: Motion DoF {dof_pos_val.shape[0]} != 23. Slicing.")
                 dof_pos_val = dof_pos_val[:23]

            self.data.qpos[7:] = dof_pos_val
            
            mujoco.mj_forward(self.model, self.data)
            
            self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
            
            if self.record_video:
                img = self.viewer.read_pixels()
                mp4_writer.append_data(img)
            else:
                self.viewer.render()
        
        self.viewer.close()
        if self.record_video:
            mp4_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--motion_file', type=str, default="walk_stand.pkl")
    parser.add_argument('--record_video', action='store_true', help="Record video to .mp4")
    args = parser.parse_args()
    
    motion_path = args.motion_file
    if not os.path.exists(motion_path):
        motion_path = os.path.join("assets/motions", args.motion_file)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_env = MotionViewEnv(motion_file=motion_path, robot_type="g1", device=device, record_video=args.record_video)
    motion_env.run()
    