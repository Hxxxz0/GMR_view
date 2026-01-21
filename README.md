<h1 align="center">General Motion Tracking for <br> Humanoid Whole-Body Control</h1>


<p align="center">
    <a href="https://zixuan417.github.io/"><strong>Zixuan Chen<sup>*,1,2</sup></strong></a>
    |
    <a href="https://jimazeyu.github.io/"><strong>Mazeyu Ji<sup>*,1</sup></strong></a>
    |
    <a href="https://chengxuxin.github.io/"><strong>Xuxin Cheng<sup>1</sup></strong></a>
    |
    <a href="https://xuanbinpeng.github.io/"><strong>Xuanbin Peng<sup>1</sup></strong></a>
    <br>
    <a href="https://xbpeng.github.io/"><strong>Xue Bin Peng†<sup>2</sup></strong></a>
    |
    <a href="https://xiaolonw.github.io/"><strong>Xiaolong Wang†<sup>1</sup></strong></a>
    <br>
    <sup>1</sup> UC San Diego
    &nbsp
    <sup>2</sup> Simon Fraser University
    <br>
    * Equal Contribution
    &nbsp
    † Equal Advising
</p>

<p align="center">
<h3 align="center"><a href="https://gmt-humanoid.github.io">Website</a> | <a href="https://arxiv.org/abs/2506.14770">arXiv</a> | <a href="https://youtu.be/n6p0DzpYjDE?si=6oIIx_Er36Ch7XWY">Video</a> </h3>
<div align="center"></div>
</p>

<p align="center">
<img src="./gmt-cover.jpeg" width="90%"/>
</p>

## Overview

This is a fork of [GMT (General Motion Tracking)](https://github.com/zixuan417/humanoid-general-motion-tracking) with added support for **visualizing NPZ motion files** from Isaac Lab.

**Key Features:**
- 🎬 Visualize G1 robot motions in MuJoCo
- 📁 Support both **PKL** (original) and **NPZ** (Isaac Lab) formats
- 🔄 Automatic quaternion format conversion (wxyz ↔ xyzw)
- 🤖 Joint order remapping from Isaac Lab (29 DOF) to MuJoCo (23 DOF)

## Installation && Running

First, clone this repo and install all the dependencies:
```bash
conda create -n gmt python=3.8 && conda activate gmt
pip3 install torch torchvision torchaudio
pip install "numpy==1.23.0" pydelatin tqdm opencv-python ipdb imageio[ffmpeg] mujoco mujoco-python-viewer scipy matplotlib
```
Then you can start to test the pretrained policy's performance on several example motions by running the following command:
```bash
python sim2sim.py --robot g1 --motion walk_stand.pkl
```
To change motions, you can replace `walk_stand.pkl` with other motions in the [motions](assets/motions/) folder.

You can also view the kinematics motion by running:
```bash
# PKL format (original)
python view_motion.py --motion assets/motions/dance.pkl

# NPZ format (Isaac Lab)
python view_motion.py --motion your_motion.npz
```

### Supported Motion File Formats

| Format | Quaternion | Field Names |
|--------|------------|-------------|
| **PKL** | `xyzw` | `root_pos`, `root_rot`, `dof_pos` |
| **NPZ** (Isaac Lab) | `wxyz` | `root_pos` or `body_pos_w`, `root_rot` or `body_quat_w`, `joint_pos` |

**NPZ Format Details:**
- `root_pos`: Shape `(T, 3)` — Root position in meters
- `root_rot`: Shape `(T, 4)` — Root rotation quaternion in **wxyz** format
- `joint_pos`: Shape `(T, 29)` — Joint angles in radians (Isaac Lab G1 joint order)
- `fps`: Scalar — Frame rate

**Isaac Lab G1 Joint Order (29 DOF):**
```
[0-1]   hip_pitch (L/R)      [2]     waist_yaw
[3-4]   hip_roll (L/R)       [5]     waist_roll
[6-7]   hip_yaw (L/R)        [8]     waist_pitch
[9-10]  knee (L/R)
[11-12] shoulder_pitch (L/R) [13-14] ankle_pitch (L/R)
[15-16] shoulder_roll (L/R)  [17-18] ankle_roll (L/R)
[19-20] shoulder_yaw (L/R)   [21-22] elbow (L/R)
[23-28] wrist_roll/pitch/yaw (L/R)
```
## ‼️Alert & Disclaimer
Although the pretrained policy has been successfully tested on our machine, the performance of the policy might vary on different robots. We cannot guarantee the success of deployment on every machine. The model we provide is for research use only, and we disclaim all responsibility for any harm, loss, or malfunction arising from its deployment.

## News
- [ ] Data processing and retargeter code will be released soon.

## Acknowledgements
+ The Mujoco simulation script is originally adapted from [LCP](https://github.com/zixuan417/smooth-humanoid-locomotion).
+ For human motion part, we mainly refer to [ASE](https://github.com/nv-tlabs/ASE) and [PHC](https://github.com/ZhengyiLuo/PHC).

## Citation
If you find this codebase useful, please consider citing our work:
```bibtex
@article{chen2025gmt,
title={GMT: General Motion Tracking for Humanoid Whole-Body Control},
author={Chen, Zixuan and Ji, Mazeyu and Cheng, Xuxin and Peng, Xuanbin and Peng, Xue Bin and Wang, Xiaolong},
journal={arXiv:2506.14770},
year={2025}
}
```
