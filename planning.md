# Weird-JEPA: Hierarchical Slot-Based World Model

## Motivation

A middle ground between Mini-JEPA (~2M params, state-based, task-engineered) and LeWM (~15M params, pixel-based, minimal structure). Target: ~6-8M params, image input + proprioception, generalizable across manipulation tasks without per-task cost shaping.

Core idea: combine JEPA-style latent dynamics with object-centric slot representations and a lightweight subgoal planner for long-horizon tasks (e.g. putting 5 cubes in a bowl).

## Architecture Overview

```
Image (96×96) → MobileNetV3-Small (truncated) → 6×6 spatial feature map (36 tokens)
                                                          ↓
Proprio (joints/gripper) → MLP → robot slot        Slot Attention (K=10, 3 iters)
                                                          ↓
                                                   10 object slots (64D each)
                                                          ↓
                              ┌────────────────────────────┴──────────────────────────┐
                              ↓                                                       ↓
                    Subgoal Planner (high level)                        Low-Level World Model + Policy
                    "which slot, where should it go"                    "predict slot dynamics, output actions"
```

## Components

### 1. Encoder: MobileNetV3-Small (truncated)

- Truncate at block ~8 to get a 6×6 spatial map (~48 channels) from 96×96 input
- Project spatial features to 64D tokens (36 tokens total)
- Why MobileNetV3: depthwise separable convs are param-efficient (~1.5M for features), designed for edge/on-device which suits robotics
- Proprioception (joint angles, gripper state) encoded separately via a small MLP → 64D "robot slot"

### 2. Slot Attention

- K=10 slots, 64D each, 3 refinement iterations
- Softmax over the SLOT axis (not token axis) → slots compete for tokens → forces object separation
- Initialized from learned Gaussian (mu + sigma per slot)
- Update: GRU cell + MLP residual per iteration
- 36 spatial tokens (from 6×6 map) gives enough resolution to separate nearby objects
- No object labels needed — trained end-to-end from prediction pressure
- Extra slots go empty gracefully (handles variable object counts)

### 3. Subgoal Planner (high level)

Lightweight planner that decomposes long-horizon goals into single next-step subgoals.

- Input: current slots + goal slots (target scene configuration)
- Output: (which_slot_to_act_on, target_position_for_that_slot)
- Operates greedily — outputs one subgoal at a time, re-queries after each is achieved
- Slot selection: score each slot given global context (current vs goal), pick the one to act on next
- Target: predict where that slot should end up (slot-delta in position space)
- Training signal: keyframes from demonstrations (detected via contact events, gripper state changes, object displacement spikes)

This is NOT full TAMP — no symbolic operators, no precondition/effect models, no search. Just a learned "what's the next thing to do" predictor that reasons over object slots.

### 4. Low-Level World Model

Same spirit as Mini-JEPA's recurrent predictor, but over the slot representation.

- Input: current slots + action
- Prediction: next slot states (8-16 step horizon)
- Architecture: GRU over flattened scene + stacked residual transition blocks
- Trained on frame-to-frame slot prediction (MSE + regularizer, similar to Mini-JEPA/LeWM)

### 5. Low-Level Policy

- Input: current slots + subgoal (target slot + target position)
- Output: action
- Trained via BC on demonstration segments
- MPC refinement: use world model to score/refine action proposals (CEM over short horizon)
- Termination: slot distance to subgoal target drops below threshold

## Execution Loop

```
while task not done:
    encode scene → slots (image + proprio)
    subgoal_planner(current_slots, goal_slots) → (slot_id, target_position)
    while slot_id not at target_position:
        policy(slots, slot_id, target_position) → action
        optionally: MPC refine action via world model rollout
        step environment
        re-encode scene → slots
    # subgoal reached, re-query planner from new state
```

## Training

| Component | Data | Loss |
|---|---|---|
| MobileNetV3 + Slot Attention | All frames | Prediction loss (downstream) + optional reconstruction |
| Low-level world model | Frame-to-frame transitions | MSE on next slots + VICReg/SIGReg |
| Low-level policy | Demo segments | BC (action MSE) |
| Subgoal planner | Keyframe-labeled demos | Cross-entropy (which slot) + MSE (target pos) |

### Keyframe extraction from demos

Segment demonstrations at transition points:
- Gripper open → close (grasp event)
- Gripper close → open (release event)
- Large object displacement spike (push/contact)
- Each segment = one subtask with a labeled (which_object, where_it_ended_up)

## Param Budget

| Component | ~Params |
|---|---|
| MobileNetV3-Small (truncated) | 1.5M |
| Slot Attention | 0.5M |
| Subgoal Planner | 1.0M |
| Low-Level World Model | 1.5M |
| Low-Level Policy | 0.5M |
| Probes/decoders | 0.5M |
| **Total** | **~6M** |

## Key Differences from Mini-JEPA

- Image input (not simulator state)
- Object-centric (slots) instead of single flat latent
- Learned subgoal decomposition for long-horizon tasks
- No task-specific cost shaping (the planner handles sequencing)
- Slightly larger but still compact

## Key Differences from LeWM

- Object-centric slots vs single scene vector
- Hierarchical (subgoal planner + low-level) vs flat autoregressive
- MobileNetV3 CNN vs ViT-Tiny (cheaper encoder)
- Structured for manipulation (slot-delta subgoals) vs task-agnostic
- Smaller (~6M vs ~15M)

## Open Questions

- Is 6×6 spatial resolution enough for slot attention to separate nearby cubes? May need to truncate backbone earlier (12×12) or use higher-res input.
- How to get goal_slots at eval time? Encode a goal image, or define goal in position space and decode to slot targets?
- Subgoal planner greedy vs lookahead — greedy re-planning may get stuck in adversarial configurations (e.g. need to move cube A out of the way to reach cube B).
- Should the world model predict per-slot independently or jointly? Joint (flattened) is simpler but doesn't compositionally generalize to new object counts.
- Reconstruction auxiliary loss for slot attention — needed for clean separation, or does prediction pressure suffice?
