# Robot Motion Generation - Representative Samples

## Overview
Generated 10 representative robot motions using the **robot_38d** checkpoint (iteration 50,016).

**Generation Parameters:**
- Checkpoint: `/root/StableMoFusion/checkpoints/robot/robot_38d/opt.txt`
- Model: EMA model with 38D motion representation
- Sampler: DPM-Solver with 50 inference steps
- Random Seed: 0
- Precision: float32
- Dataset: Robot (30 joints, 50 FPS)

---

## Generated Motions

### 1. **Kicking** (000000.npz)
- **Prompt:** "a man kicks something or someone with his left leg."
- **Duration:** 150 frames (3.0 seconds)
- **Category:** Leg action

### 2. **Jumping** (000001.npz)
- **Prompt:** "a man full-body sideways jumps to his left."
- **Duration:** 150 frames (3.0 seconds)
- **Category:** Full-body locomotion

### 3. **Walking with Direction Change** (000002.npz)
- **Prompt:** "a person walks slowly forward then toward the left hand side and stands facing that direction."
- **Duration:** 250 frames (5.0 seconds)
- **Category:** Complex locomotion

### 4. **Waving** (000003.npz)
- **Prompt:** "a man waves his right hand."
- **Duration:** 150 frames (3.0 seconds)
- **Category:** Arm gesture

### 5. **Running and Jumping** (000004.npz)
- **Prompt:** "a person runs and jumps forward"
- **Duration:** 200 frames (4.0 seconds)
- **Category:** Dynamic locomotion

### 6. **Sitting and Standing** (000005.npz)
- **Prompt:** "a man carefully sits down on the ground and then stands back up"
- **Duration:** 250 frames (5.0 seconds)
- **Category:** Pose transition

### 7. **Walking with Turn** (000006.npz)
- **Prompt:** "a person walks forward then turns around and walks back."
- **Duration:** 250 frames (5.0 seconds)
- **Category:** Locomotion with rotation

### 8. **Bending with Hand Motion** (000007.npz)
- **Prompt:** "a person bends down and motion the right hand in circles."
- **Duration:** 200 frames (4.0 seconds)
- **Category:** Combined body and hand motion

### 9. **Arm Raising** (000008.npz)
- **Prompt:** "a person raises his right arm and moves his left arm and left hand up and down in a regular fashion."
- **Duration:** 200 frames (4.0 seconds)
- **Category:** Coordinated arm motion

### 10. **Complex Walking Sequence** (000009.npz)
- **Prompt:** "a person walks toward the front, turns to the right, bounces into a squat, and places both arms in front of chest before placing them on the knees."
- **Duration:** 300 frames (6.0 seconds)
- **Category:** Multi-stage complex motion

---

## Motion Categories Covered

The 10 representative prompts cover diverse motion types from the robot training dataset:

1. ✅ **Kicking** - Single leg actions
2. ✅ **Jumping** - Full-body vertical motion
3. ✅ **Walking** - Basic locomotion with variations (straight, diagonal, turning)
4. ✅ **Running** - Dynamic locomotion
5. ✅ **Waving** - Hand/arm gestures
6. ✅ **Sitting/Standing** - Pose transitions
7. ✅ **Bending** - Torso motion
8. ✅ **Arm Raising** - Upper body manipulation
9. ✅ **Complex Sequences** - Multi-stage compound motions

---

## Output Files

### NPZ Files (Complete Robot Motion Data)
Located in: `npz/`
- **Format:** 38D representation
  - Joint positions (29D)
  - Root velocity XY (2D)
  - Root Z position (1D)
  - Root rotation 6D (6D)
- **Files:** 000000.npz through 000009.npz

### NPY Files (Legacy Position Data)
Located in: `npy/`
- **Format:** Root positions only
- **Files:** 000000.npy through 000009.npy

### Metadata
- `prompts.txt` - Text descriptions for each motion
- `lengths.txt` - Frame counts for each motion
- `GENERATION_SUMMARY.md` - This file

---

## Usage

These generated motions can be:
1. **Visualized** using robot animation tools
2. **Evaluated** for quality metrics (FID, diversity, etc.)
3. **Used as references** for motion synthesis
4. **Converted** to other formats (BVH, SMPL, etc.)

---

## Technical Details

**Model Architecture:**
- Base dim: 512
- Dim mults: [2, 2, 2, 2]
- Number of layers: 8
- Latent dim: 512
- Text latent dim: 256

**Training Configuration:**
- Training steps: 50,000
- Batch size: 128
- Diffusion steps: 1000
- Beta schedule: linear
- Prediction type: sample

Generated on: 2026-01-22

