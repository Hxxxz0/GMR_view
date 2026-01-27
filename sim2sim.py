# -----------------------------------------------------------------------------
# Copyright [2025] [Zixuan Chen, Mazeyu Ji, Xuxin Cheng, Xuanbin Peng, Xue Bin Peng, Xiaolong Wang]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is adapted from the open-source script:
# https://github.com/zixuan417/smooth-humanoid-locomotion/blob/main/simulation/legged_gym/legged_gym/scripts/sim2sim.py
# -----------------------------------------------------------------------------


import argparse, os, time
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional

from utils.motion_lib import MotionLib


@dataclass
class DomainRandomizationConfig:
    """Domain randomization configuration for sim-to-real transfer."""
    
    # Enable/disable domain randomization
    enabled: bool = False
    
    # === Physical Parameter Randomization ===
    # Friction coefficient randomization (multiplier range)
    friction_range: Tuple[float, float] = (0.5, 2.0)
    
    # Joint stiffness randomization (multiplier range)
    stiffness_range: Tuple[float, float] = (0.8, 1.2)
    
    # Joint damping randomization (multiplier range)
    damping_range: Tuple[float, float] = (0.8, 1.2)
    
    # Torque limit randomization (multiplier range)
    torque_limit_range: Tuple[float, float] = (0.9, 1.0)
    
    # Link mass randomization (multiplier range)
    mass_range: Tuple[float, float] = (0.9, 1.1)
    
    # Center of mass offset randomization (meters)
    com_offset_range: Tuple[float, float] = (-0.02, 0.02)
    
    # === External Disturbance ===
    # Push force range (N) - random pushes applied to the robot
    push_force_range: Tuple[float, float] = (0.0, 100.0)
    
    # Push interval range (seconds)
    push_interval_range: Tuple[float, float] = (2.0, 5.0)
    
    # Push duration (seconds)  
    push_duration: float = 0.1
    
    # === Sensor Noise ===
    # IMU angular velocity noise (rad/s, std dev)
    imu_ang_vel_noise: float = 0.02
    
    # IMU orientation noise (rad, std dev)
    imu_orientation_noise: float = 0.01
    
    # Joint position noise (rad, std dev)
    joint_pos_noise: float = 0.01
    
    # Joint velocity noise (rad/s, std dev)
    joint_vel_noise: float = 0.05
    
    # === Action Delay ===
    # Action delay range (in control steps)
    action_delay_range: Tuple[int, int] = (0, 2)
    
    # === Ground Randomization ===
    # Ground height variation (meters)
    ground_height_range: Tuple[float, float] = (-0.02, 0.02)
    
    # Randomization interval (seconds) - how often to re-randomize
    randomization_interval: float = 5.0


class DomainRandomizer:
    """Applies domain randomization during simulation."""
    
    def __init__(self, config: DomainRandomizationConfig, model: mujoco.MjModel, 
                 original_stiffness: np.ndarray, original_damping: np.ndarray,
                 original_torque_limits: np.ndarray):
        self.config = config
        self.model = model
        self.original_stiffness = original_stiffness.copy()
        self.original_damping = original_damping.copy()
        self.original_torque_limits = original_torque_limits.copy()
        
        # Store original model parameters
        self.original_friction = model.geom_friction.copy()
        self.original_mass = model.body_mass.copy()
        self.original_ipos = model.body_ipos.copy()
        
        # Current randomized parameters
        self.current_stiffness = original_stiffness.copy()
        self.current_damping = original_damping.copy()
        self.current_torque_limits = original_torque_limits.copy()
        
        # Push state
        self.next_push_time = 0.0
        self.push_end_time = 0.0
        self.current_push_force = np.zeros(3)
        self.push_body_id = 1  # Usually the root body
        
        # Action delay buffer
        self.action_buffer = deque(maxlen=config.action_delay_range[1] + 1)
        self.current_delay = 0
        
        # Last randomization time
        self.last_randomization_time = 0.0
        
        if config.enabled:
            self.randomize_all()
    
    def randomize_all(self):
        """Randomize all domain parameters."""
        if not self.config.enabled:
            return
            
        self._randomize_physics()
        self._randomize_action_delay()
        self._schedule_next_push()
    
    def _randomize_physics(self):
        """Randomize physical parameters."""
        cfg = self.config
        
        # Randomize friction
        friction_scale = np.random.uniform(cfg.friction_range[0], cfg.friction_range[1])
        self.model.geom_friction[:] = self.original_friction * friction_scale
        
        # Randomize stiffness and damping
        stiffness_scale = np.random.uniform(cfg.stiffness_range[0], cfg.stiffness_range[1], 
                                            size=self.original_stiffness.shape)
        damping_scale = np.random.uniform(cfg.damping_range[0], cfg.damping_range[1],
                                          size=self.original_damping.shape)
        self.current_stiffness = self.original_stiffness * stiffness_scale
        self.current_damping = self.original_damping * damping_scale
        
        # Randomize torque limits
        torque_scale = np.random.uniform(cfg.torque_limit_range[0], cfg.torque_limit_range[1],
                                         size=self.original_torque_limits.shape)
        self.current_torque_limits = self.original_torque_limits * torque_scale
        
        # Randomize mass (skip first body which is usually world)
        for i in range(1, len(self.original_mass)):
            if self.original_mass[i] > 0:
                mass_scale = np.random.uniform(cfg.mass_range[0], cfg.mass_range[1])
                self.model.body_mass[i] = self.original_mass[i] * mass_scale
        
        # Randomize center of mass offset
        for i in range(1, len(self.original_ipos)):
            com_offset = np.random.uniform(cfg.com_offset_range[0], cfg.com_offset_range[1], size=3)
            self.model.body_ipos[i] = self.original_ipos[i] + com_offset
    
    def _randomize_action_delay(self):
        """Randomize action delay."""
        self.current_delay = np.random.randint(
            self.config.action_delay_range[0], 
            self.config.action_delay_range[1] + 1
        )
        self.action_buffer.clear()
    
    def _schedule_next_push(self):
        """Schedule the next push disturbance."""
        interval = np.random.uniform(
            self.config.push_interval_range[0],
            self.config.push_interval_range[1]
        )
        self.next_push_time = self.next_push_time + interval
        
        # Random direction and magnitude
        force_magnitude = np.random.uniform(
            self.config.push_force_range[0],
            self.config.push_force_range[1]
        )
        direction = np.random.randn(3)
        direction[2] = 0  # Only horizontal push
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        self.current_push_force = direction * force_magnitude
    
    def add_sensor_noise(self, dof_pos: np.ndarray, dof_vel: np.ndarray, 
                         quat: np.ndarray, ang_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add noise to sensor readings."""
        if not self.config.enabled:
            return dof_pos, dof_vel, quat, ang_vel
        
        cfg = self.config
        
        # Add noise to joint positions
        noisy_dof_pos = dof_pos + np.random.normal(0, cfg.joint_pos_noise, size=dof_pos.shape)
        
        # Add noise to joint velocities
        noisy_dof_vel = dof_vel + np.random.normal(0, cfg.joint_vel_noise, size=dof_vel.shape)
        
        # Add noise to angular velocity
        noisy_ang_vel = ang_vel + np.random.normal(0, cfg.imu_ang_vel_noise, size=ang_vel.shape)
        
        # Add noise to quaternion (simplified - add noise to euler then convert back would be more accurate)
        # Here we just add small noise to quaternion components and renormalize
        noisy_quat = quat + np.random.normal(0, cfg.imu_orientation_noise, size=quat.shape)
        noisy_quat = noisy_quat / (np.linalg.norm(noisy_quat) + 1e-8)
        
        return noisy_dof_pos.astype(np.float32), noisy_dof_vel.astype(np.float32), \
               noisy_quat.astype(np.float32), noisy_ang_vel.astype(np.float32)
    
    def apply_push(self, data: mujoco.MjData, current_time: float):
        """Apply push disturbance if it's time."""
        if not self.config.enabled:
            return
            
        if current_time >= self.next_push_time:
            if current_time < self.next_push_time + self.config.push_duration:
                # Apply push force
                data.xfrc_applied[self.push_body_id, :3] = self.current_push_force
            else:
                # Push ended, schedule next push
                data.xfrc_applied[self.push_body_id, :3] = 0
                self._schedule_next_push()
    
    def delay_action(self, action: np.ndarray) -> np.ndarray:
        """Apply action delay."""
        if not self.config.enabled or self.current_delay == 0:
            return action
            
        self.action_buffer.append(action.copy())
        
        if len(self.action_buffer) > self.current_delay:
            return self.action_buffer[-self.current_delay - 1]
        else:
            return np.zeros_like(action)
    
    def maybe_rerandomize(self, current_time: float):
        """Re-randomize parameters periodically."""
        if not self.config.enabled:
            return
            
        if current_time - self.last_randomization_time >= self.config.randomization_interval:
            self._randomize_physics()
            self._randomize_action_delay()
            self.last_randomization_time = current_time
            print(f"[DR] Re-randomized at t={current_time:.2f}s")
    
    def get_stiffness(self) -> np.ndarray:
        return self.current_stiffness
    
    def get_damping(self) -> np.ndarray:
        return self.current_damping
    
    def get_torque_limits(self) -> np.ndarray:
        return self.current_torque_limits

@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec

class HumanoidEnv:
    def __init__(self, policy_path, motion_path, robot_type="g1", device="cuda", record_video=False,
                 domain_randomization_config: Optional[DomainRandomizationConfig] = None):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        self.motion_path = motion_path
        self.dr_config = domain_randomization_config or DomainRandomizationConfig(enabled=False)
        
        if robot_type == "g1":
            model_path = "assets/robots/g1/g1.xml"
            self.stiffness = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40,
                40, 40, 40, 40,
            ])
            self.damping = np.array([
                2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2,
                4, 4, 4,
                5, 5, 5, 5,
                5, 5, 5, 5,
            ])
            self.num_actions = 23
            self.num_dofs = 23
            self.default_dof_pos = np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.4, 0.0, 1.2,
                0.0, -0.4, 0.0, 1.2,
            ])
            self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25,
                25, 25, 25, 25,
            ])
            self.dof_names = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
                              "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
                              "waist_yaw", "waist_roll", "waist_pitch",
                              "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
                              "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]

        else:
            raise ValueError(f"Robot type {robot_type} not supported!")
        
        self.obs_indices = np.arange(self.num_dofs)
        
        self.sim_duration = 60.0
        self.sim_dt = 0.001
        self.sim_decimation = 20
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        if self.record_video:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen')
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.distance = 5.0
        
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.5
        
        self.tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        if robot_type == "g1":
            self.n_priv = 0
            self.n_proprio = 3 + 2 + 3*self.num_actions
            self.n_priv_latent = 1
            self.key_body_ids = [29, 37,  6, 14,  4, 12, 25, 33, 20]
            
        self.history_len = 20
        self.priv_latent = np.zeros(self.n_priv_latent, dtype=np.float32)
        
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.ang_vel_scale = 0.25
        
        self._motion_lib = MotionLib(self.motion_path, self.device)
        self.sim_duration = self._motion_lib.get_total_length() * 1000.0 # Loop many times
        print(f"Simulation duration set to {self.sim_duration:.1f}s (approx {self.sim_duration/3600:.1f} hours)")
        self._init_motion_buffers()
        
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        
        print("Loading jit for policy: ", policy_path)
        self.policy_path = policy_path
        self.policy_jit = torch.jit.load(policy_path, map_location=self.device)
        
        # Initialize Domain Randomizer
        self.domain_randomizer = DomainRandomizer(
            config=self.dr_config,
            model=self.model,
            original_stiffness=self.stiffness,
            original_damping=self.damping,
            original_torque_limits=self.torque_limits
        )
        if self.dr_config.enabled:
            print("[DR] Domain Randomization ENABLED")
        
        self.last_time = time.time()
    
    def _init_motion_buffers(self):
        self.tar_obs_steps = torch.tensor(self.tar_obs_steps, device=self.device, dtype=torch.int)
        
    def _get_mimic_obs(self, curr_time_step):
        num_steps = len(self.tar_obs_steps)
        motion_times = torch.tensor([curr_time_step * self.control_dt], device=self.device).unsqueeze(-1)
        obs_motion_times = self.tar_obs_steps * self.control_dt + motion_times
        obs_motion_times = obs_motion_times.flatten()
        motion_ids = torch.zeros(num_steps, dtype=torch.int, device=self.device)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _ = self._motion_lib.calc_motion_frame(motion_ids, obs_motion_times)
        
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(1, num_steps, 1)
        pitch = pitch.reshape(1, num_steps, 1)
        yaw = yaw.reshape(1, num_steps, 1)
        
        root_vel = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
        
        root_pos = root_pos.reshape(1, num_steps, 3)
        root_vel = root_vel.reshape(1, num_steps, 3)
        root_ang_vel = root_ang_vel.reshape(1, num_steps, 3)
        dof_pos = dof_pos.reshape(1, num_steps, -1)
        
        if dof_pos.shape[-1] == 29:
             # Retarget 29 DoF to 23 DoF
             # 0-19: Legs (12) + Waist (3) + Left Arm (4, excluding wrist)
             # 22-26: Right Arm (4, excluding wrist)
             dof_pos = torch.cat([dof_pos[..., :19], dof_pos[..., 22:26]], dim=-1)
        
        if self.robot_type == "g1":
            mimic_obs_buf = torch.cat((
                root_pos[..., 2:3],
                roll, pitch,
                root_vel,
                root_ang_vel[..., 2:3],
                dof_pos,
            ), dim=-1)
        
        mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
        
        return mimic_obs_buf.detach().cpu().numpy().squeeze()
        
    def extract_data(self):
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        quat = self.data.sensor('orientation').data.astype(np.float32)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        self.dof_vel = torch.from_numpy(dof_vel).float().unsqueeze(0).to(self.device)
        return (dof_pos, dof_vel, quat, ang_vel)

    def check_fall(self):
        # A simple heuristic: if root height < 0.5 (initial is ~0.79)
        return self.data.qpos[2] < 0.0

    def reset_robot(self, time_s):
        motion_time = torch.tensor([time_s], dtype=torch.float, device=self.device)
        motion_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_time)
        
        # Reset internal buffers
        self.last_action[:] = 0
        for i in range(len(self.proprio_history_buf)):
            self.proprio_history_buf[i][:] = 0
            
        # Set MuJoCo state
        self.data.qpos[:3] = root_pos[0].cpu().numpy()
        self.data.qpos[3:7] = root_rot[0].cpu().numpy()[[3, 0, 1, 2]] # (x,y,z,w) -> (w,x,y,z)
        
        dof_pos_val = dof_pos[0].cpu().numpy()
        dof_vel_val = dof_vel[0].cpu().numpy()
        
        # Retargeting for 29 DoF motion if needed
        if dof_pos_val.shape[0] == 29:
            dof_pos_val = np.concatenate([dof_pos_val[:19], dof_pos_val[22:26]])
            dof_vel_val = np.concatenate([dof_vel_val[:19], dof_vel_val[22:26]])
        elif dof_pos_val.shape[0] != 23:
            # Fallback slicing
            dof_pos_val = dof_pos_val[:23]
            dof_vel_val = dof_vel_val[:23]

        self.data.qpos[7:] = dof_pos_val
        self.data.qvel[:3] = root_vel[0].cpu().numpy()
        self.data.qvel[3:6] = root_ang_vel[0].cpu().numpy()
        self.data.qvel[6:] = dof_vel_val
        
        mujoco.mj_forward(self.model, self.data)
        
    def run(self):
        motion_name = os.path.basename(self.motion_path).split('.')[0]
        if self.record_video:
            import imageio
            video_name = f"{self.robot_type}_{''.join(os.path.basename(self.policy_path).split('.')[:-1])}_{motion_name}.mp4"
            path = "mujoco_videos/"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            mp4_writer = imageio.get_writer(video_name, fps=50)
        
        for i in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Running simulation..."):
            current_time = i * self.sim_dt
            dof_pos, dof_vel, quat, ang_vel = self.extract_data()
            
            # Apply sensor noise if domain randomization is enabled
            dof_pos, dof_vel, quat, ang_vel = self.domain_randomizer.add_sensor_noise(
                dof_pos, dof_vel, quat, ang_vel
            )
            
            # Apply push disturbance
            self.domain_randomizer.apply_push(self.data, current_time)
            
            # Periodic re-randomization
            self.domain_randomizer.maybe_rerandomize(current_time)
            
            if i % self.sim_decimation == 0:
                if self.check_fall():
                    curr_time_s = (i // self.sim_decimation) * self.control_dt 
                    # Optionally add a small offset to advance past the tricky part, but user said "from where it fell" implies continuing.
                    # Since we are tracking a motion, resetting to ground truth at T is effectively "putting it back on track".
                    # Let's just reset to T.
                    self.reset_robot(curr_time_s)
                    dof_pos, dof_vel, quat, ang_vel = self.extract_data()
                
                curr_timestep = i // self.sim_decimation
                mimic_obs = self._get_mimic_obs(curr_timestep)
                
                rpy = quatToEuler(quat)
                obs_dof_vel = dof_vel.copy()
                obs_dof_vel[[4, 5, 10, 11]] = 0.
                obs_prop = np.concatenate([
                    ang_vel * self.ang_vel_scale,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    obs_dof_vel * self.dof_vel_scale,
                    self.last_action,
                ])
                
                assert obs_prop.shape[0] == self.n_proprio, f"Expected {self.n_proprio} but got {obs_prop.shape[0]}"
                obs_hist = np.array(self.proprio_history_buf).flatten()

                if self.robot_type == "g1":
                    obs_buf = np.concatenate([mimic_obs, obs_prop, obs_hist])
                
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    raw_action = self.policy_jit(obs_tensor).cpu().numpy().squeeze()
                
                self.last_action = raw_action.copy()
                
                # Apply action delay if domain randomization is enabled
                raw_action = self.domain_randomizer.delay_action(raw_action)
                
                raw_action = np.clip(raw_action, -10., 10.)
                scaled_actions = raw_action * self.action_scale
                
                step_actions = np.zeros(self.num_dofs)
                step_actions = scaled_actions
                
                pd_target = step_actions + self.default_dof_pos
                
                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                if self.record_video:
                    img = self.viewer.read_pixels()
                    mp4_writer.append_data(img)
                else:
                    self.viewer.render()

                self.proprio_history_buf.append(obs_prop)
                
        
            # Use randomized stiffness/damping/torque_limits from DomainRandomizer
            stiffness = self.domain_randomizer.get_stiffness()
            damping = self.domain_randomizer.get_damping()
            torque_limits = self.domain_randomizer.get_torque_limits()
            
            torque = (pd_target - dof_pos) * stiffness - dof_vel * damping
            torque = np.clip(torque, -torque_limits, torque_limits)
            
            self.data.ctrl = torque
            
            mujoco.mj_step(self.model, self.data)
        
        self.viewer.close()
        if self.record_video:
            mp4_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--robot', type=str, default="g1")
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--record_video', action='store_true')
    # support both the README-style flag (--motion) and the original flag (--motion_file)
    parser.add_argument('--motion', '--motion_file', dest='motion_file', type=str, default="walk_stand.pkl")
    
    # Domain Randomization arguments
    parser.add_argument('--domain_random', action='store_true', help='Enable domain randomization')
    parser.add_argument('--dr_friction', type=float, nargs=2, default=[0.5, 2.0], metavar=('MIN', 'MAX'),
                        help='Friction coefficient multiplier range (default: 0.5 2.0)')
    parser.add_argument('--dr_stiffness', type=float, nargs=2, default=[0.8, 1.2], metavar=('MIN', 'MAX'),
                        help='Joint stiffness multiplier range (default: 0.8 1.2)')
    parser.add_argument('--dr_damping', type=float, nargs=2, default=[0.8, 1.2], metavar=('MIN', 'MAX'),
                        help='Joint damping multiplier range (default: 0.8 1.2)')
    parser.add_argument('--dr_mass', type=float, nargs=2, default=[0.9, 1.1], metavar=('MIN', 'MAX'),
                        help='Link mass multiplier range (default: 0.9 1.1)')
    parser.add_argument('--dr_push_force', type=float, nargs=2, default=[0.0, 100.0], metavar=('MIN', 'MAX'),
                        help='Push force range in Newtons (default: 0.0 100.0)')
    parser.add_argument('--dr_push_interval', type=float, nargs=2, default=[2.0, 5.0], metavar=('MIN', 'MAX'),
                        help='Push interval range in seconds (default: 2.0 5.0)')
    parser.add_argument('--dr_action_delay', type=int, nargs=2, default=[0, 2], metavar=('MIN', 'MAX'),
                        help='Action delay range in control steps (default: 0 2)')
    parser.add_argument('--dr_joint_noise', type=float, default=0.01,
                        help='Joint position noise std dev in radians (default: 0.01)')
    parser.add_argument('--dr_imu_noise', type=float, default=0.02,
                        help='IMU angular velocity noise std dev in rad/s (default: 0.02)')
    args = parser.parse_args()
    
    jit_policy_pth = "assets/pretrained_checkpoints/pretrained.pt"
    assert os.path.exists(jit_policy_pth), f"Policy path {jit_policy_pth} does not exist!"
    print(f"Loading model from: {jit_policy_pth}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    motion_file = args.motion_file
    if not os.path.exists(motion_file):
        motion_file = os.path.join("assets/motions", args.motion_file)
    
    # Setup Domain Randomization config
    dr_config = DomainRandomizationConfig(
        enabled=args.domain_random,
        friction_range=tuple(args.dr_friction),
        stiffness_range=tuple(args.dr_stiffness),
        damping_range=tuple(args.dr_damping),
        mass_range=tuple(args.dr_mass),
        push_force_range=tuple(args.dr_push_force),
        push_interval_range=tuple(args.dr_push_interval),
        action_delay_range=tuple(args.dr_action_delay),
        joint_pos_noise=args.dr_joint_noise,
        imu_ang_vel_noise=args.dr_imu_noise,
    )
    
    env = HumanoidEnv(
        policy_path=jit_policy_pth, 
        motion_path=motion_file, 
        robot_type=args.robot, 
        device=device, 
        record_video=args.record_video,
        domain_randomization_config=dr_config
    )
    
    env.run()
        
        
