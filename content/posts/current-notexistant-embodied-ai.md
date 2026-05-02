---

# Basic info

title: "NOTES: The embodied AI stack, taught from systems first"
date: 2026-05-01T10:00:00-07:00
draft: false
description: "Isaac Lab, cuRobo, Warp kernels, Slurm eval pipelines, VLAs, sim2real, and why benchmarking is secretly the most strategic place to stand in a modern robotics lab."
tags: ["notes", "robotics", "embodied-ai", "isaac-sim", "curobo", "vla", "simulation", "sim2real", "ml-systems", "gpu", "ai"]
author: "Me"


# Metadata & SEO

canonicalURL: "https://canonical.url/to/page"
hidemeta: false
searchHidden: true

# Table of contents

showToc: true
TocOpen: false
UseHugoToc: true

# Post features

ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
comments: false

# Syntax highlighting

disableHLJS: false
disableShare: false

# Edit link
editPost:
  URL: "https://github.com/t-avil/blog/tree/main/content"
  Text: "Suggest Changes"   
  appendFilePath: true      
--------------------

For about fifteen years there were two tribes in robotics. One wrote C++ on CPUs, proved convergence properties of control laws, and argued about quaternions at lunch. The other trained neural nets, ran sweeps on GPUs, and measured everything in loss curves. They barely talked. Manipulation - getting an arm to pick up a cup - sat awkwardly in the middle: too messy for clean control theory, too physical for pure ML.

What changed around 2022 is that the ML tribe started winning, and what they won with was scale. The same recipe that gave us ChatGPT - big transformers on big data - turned out to work for robots, but only if you fed them the right data, in the right simulator, on the right GPU, with the right motion planner underneath. That stack is what this post is about.

Assuming you’ve taken a DL class - transformers, attention, backprop - and you’ve at least squinted at a CUDA kernel. Not assuming you know what inverse kinematics is, what a URDF file is, or why mimic joints ruin every motion planner’s afternoon. We’ll build that up.

---

## The Onboarding Stack: Five Things You’ll Actually Touch on Day One

Before the intellectual history, the plumbing. There are five tools you’d be expected to handle on day one in a modern manipulation lab: Isaac Lab, cuRobo, NVIDIA Warp, Slurm, and containers. Most labs bleed engineering hours here, and this is exactly where someone with a systems background has leverage.

### Isaac Sim and Isaac Lab

Isaac Sim is NVIDIA’s robotics simulator, built on Omniverse. Under the hood it runs **PhysX**, NVIDIA’s rigid-body engine - originally a game physics library, acquired from Ageia in 2008. The thing Isaac Sim does that Gazebo and PyBullet couldn’t is run physics on the GPU. Every joint update, every collision query, every contact force, vectorized across thousands of parallel environments on one GPU.

If you’ve written CUDA, the mental model is: each "simulation environment" is a lane in an SIMD batch, and you’re running thousands of lanes at once. That’s how you go from `~100 robot steps/s` on CPU physics to `~100,000 steps/s` on GPU physics. Three orders of magnitude. That speedup is the reason modern RL for robotics is even tractable. Without it, training a locomotion policy is a multi-week affair; with it, overnight.

**Isaac Lab** sits one layer above. Isaac Sim is engine + rendering + USD scene description; Isaac Lab is the opinionated Python framework that says "here’s how you write an RL environment, here’s how you define a task, here’s how observations and actions plug in." If you’re coming from Gym/Gymnasium, Isaac Lab is basically a GPU-vectorized Gym, with much more structure because robotics tasks need kinematics, sensors, and scene resets that Gym’s interface is too thin for.

One gotcha that will eat a day of your life: **USD** (Universal Scene Description, Pixar’s format) is the scene file. It is **not** URDF. URDF is the old ROS format for describing a robot - joints, links, meshes. USD can import URDF but adds material, sensor, and physics properties. When you onboard, expect to spend a day figuring out why your robot’s gripper looks right in URDF but the fingers clip through objects in USD. It’s almost always a missing collision mesh or a bad mass property.

---

### cuRobo and the Mimic Joint That Ruins Your Week

cuRobo is NVIDIA’s GPU-accelerated motion planning library. To explain why it matters, a quick detour into kinematics.

**Forward kinematics** is the easy direction: given joint angles, where is the hand? It’s a chain of matrix multiplications - each joint contributes a 4x4 homogeneous transform, you multiply them, you get the hand pose. Linear time in joint count. Afternoon project.

**Inverse kinematics** is the hard direction: given a desired hand pose, what joint angles get me there? For anything with more than three joints this is nonlinear, often has multiple solutions, sometimes has no solution, and is the central subroutine of every manipulation stack. Classical IK solvers - KDL, TracIK, IKFast - are CPU-based, single-query, and either iterative (Newton’s method on the Jacobian) or analytic (solved symbolically for a known arm). They’re fast for one query.

But what if you want ten thousand queries? Sampling candidate grasps. Evaluating a batch of goal poses. Running MPC where each control step needs many rollouts. Suddenly the CPU IK solver is your bottleneck.

cuRobo’s pitch: do all of it on the GPU, in batch. Thousands of IK queries, thousands of collision checks, thousands of trajectory optimizations - at once. Internally it’s a GPU implementation of trajectory optimization - basically batched Levenberg-Marquardt / gradient descent on a cost function combining pose error, joint limits, collision avoidance, and smoothness. Collision checking uses **signed distance fields (SDFs)**: 3D grids where each voxel stores distance to the nearest obstacle. SDF queries on the GPU are essentially texture lookups, which is why it’s fast.

Now the **mimic joint** thing, because this is a perfect example of where robotics bites you in ways a pure-software background wouldn’t anticipate.

A parallel-jaw gripper - the two-finger gripper you see on most arms - often has two finger joints that are mechanically coupled: when one closes, the other mirrors it. In URDF this is expressed as a "mimic" relationship: joint B mimics joint A with some multiplier. The IK solver needs to respect this. If it treats the two joints as independent, it will happily find a solution where one finger goes left and the other goes right, which is physically impossible and will crash your planner, cause self-collision, or silently produce invalid rollouts.

This is exactly the kind of bug that stops an entire lab’s work for weeks if nobody fixes it. It is also invisible to researchers - they see "the planner fails sometimes" and move on. The person who finds it, patches it, and pushes the fix becomes indispensable very quickly. That’s the shape of leverage in an infra role.

---

### NVIDIA Warp: the Quiet MVP of the Stack

NVIDIA Warp is a Python-to-CUDA JIT compiler for simulation and graphics kernels. If you’ve written CUDA the traditional way, you know the cycle: write a `.cu` file, compile with nvcc, link into Python with pybind11 or Cython, cross your fingers. Warp lets you write something like:

```python
@wp.kernel
def step(state: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    state[tid] = state[tid] + wp.vec3(0.0, -9.8 * dt, 0.0)
```

…and it compiles that function to **PTX** (NVIDIA’s GPU assembly), caches it, and launches as a CUDA kernel. Under the hood it’s doing roughly what Numba or Triton does, but with first-class support for the types you actually need in simulation: 3D vectors, quaternions, homogeneous transforms, spatial matrices, triangle meshes, signed distance fields.

Why does this matter for robotics? Because a lot of robotics logic is "for each of N environments, update this state based on these rules" - and that is exactly the CUDA programming model. A plan-following state machine - "if the arm reached waypoint 3, advance to waypoint 4; if it’s stuck, reset" - is naturally per-environment logic. If you write it in Python and loop over environments on the CPU, you’re paying a Python-to-GPU roundtrip per environment per step. If you write it as a Warp kernel, the state machine runs on the GPU in the same address space as the physics state. No copy. Massive parallelism.

A 20x speedup from porting a state machine to Warp isn’t magic - it’s **Amdahl’s Law in reverse**. Physics was already on GPU. The state machine was on CPU. Every step you were synchronizing CPU and GPU and copying state back and forth; that synchronization cost grows with environment count. Moving the state machine to the GPU side of the fence eliminates the copy, collapses per-step latency, and lets you scale environment count up until something else becomes the bottleneck. This is the kind of mechanical sympathy that a C++/CUDA person sees instantly and an ML-only person often misses entirely.

Mental model: Warp is what CUDA would look like if it had been designed in 2024 by someone who actually wanted people to use it.

---

### Slurm Eval Pipelines

Slurm is the workload manager for essentially every academic HPC cluster. The mental model is simple: Slurm is to a multi-node GPU cluster what the OS scheduler is to a multi-core CPU. You write a job script - a shell script with some `#SBATCH` directives at the top specifying GPUs, memory, walltime - and submit it with `sbatch`. Slurm finds a node with free resources and runs your script there.

For robotics eval the shape of the problem is usually: "I have 50 trained checkpoints, 200 tasks, I want every checkpoint evaluated on every task." That’s 10,000 jobs. You don’t submit them individually. You use a **Slurm job array** - `#SBATCH --array=0-9999` - which tells Slurm to run 10,000 copies of the script, each with a different `$SLURM_ARRAY_TASK_ID`, which your script uses to figure out which checkpoint-by-task pair it’s responsible for.

The tricky parts nobody tells you about:

- **Environment setup** - your Python env has to be available on every compute node, usually via a shared filesystem or a container.
- **Result aggregation** - every job writes a result file, you collect them at the end, usually into Weights & Biases or a Postgres database.
- **Failure handling** - about 1% of jobs will fail because a node had a bad GPU or got preempted, and you need to re-run just those without re-running the other 9,900.

A well-built Slurm eval pipeline turns "kick off the eval before bed and check in the morning" from a prayer into an operation. 24/7 eval automation is exactly this.

---

### Containerized Deploy

You know Docker. The robotics-specific thing to know is that containers for ML workloads are usually built on top of the **NVIDIA NGC** base images - `nvidia/cuda:12.x-cudnn9-devel-ubuntu22.04` - because they ship with the right CUDA toolchain baked in. You layer Python env on top, then code, then entry point.

Two gotchas that break everyone the first time:

- The **NVIDIA Container Toolkit** has to be installed on the host for `--gpus all` to work. Without it, your container can’t see the GPU and everything silently falls back to CPU, which on a training workload looks like "my code is a thousand times slower than it should be."
- Mount your data as a volume, **don’t bake it into the image**. Your image will become many gigabytes if you bake training data in, and pushing that image to every node gets painful fast.

Containerizing eval is how you make it reproducible across your laptop, the cluster, and a cloud VM. Same pattern, different payload.

That’s the onboarding stack. Now the intellectual history.

---

## The Tech Tree: Three Eras Repeating in Every Subfield

Everything that follows fits into one of three eras, and the same story repeats in every subfield.

- **Era 1 - Classical.** Hand-designed algorithms, CPU, works on a few known problems, brittle outside them. Usually peaks 2010–2018.
- **Era 2 - Deep learning, narrow.** Train a neural net per task. Better than classical, but each task needs its own dataset and model. Roughly 2018–2022.
- **Era 3 - Foundation models.** One big model pretrained on everything, finetuned or prompted for specifics. 2022–present.

Perception, policies, planning, simulation - every subfield is at a slightly different point on this arc.

---

### Policies (the non-VLA Lineage)

A "policy" is a function from observation to action. Give it what the robot sees; get back what the robot should do. In ML terms it’s a regression model. What makes it hard: action space is continuous, the correct output is often multimodal (multiple valid actions for the same observation), and data is expensive to collect.

**Legacy.** A multilayer perceptron that takes a flattened image and outputs joint velocities. Classic behavior cloning - supervised learning where the label is "what the expert did." It works for trivial tasks and fails the moment the robot sees a state the expert never demonstrated. This is **distributional shift**, and it compounds every timestep, which is why behavior cloning has a bad reputation it partly deserves. Gaussian Mixture Models were a slight improvement because they can represent multimodality - "either go left or go right, not straight through the middle" - but they don’t scale to high dimensions.

**Current.** Two architectures matter.

**ACT** - Action-Chunking Transformer, from the ALOHA paper by Tony Zhao at Stanford. Key idea: instead of predicting one action at a time, predict a chunk of actions (say, the next 50 timesteps) and execute them open-loop before re-querying. This sidesteps compounding error. Re-query every step, small prediction errors knock you off the demonstration distribution, next prediction is worse, and so on. Predict a chunk and commit to it, the loop breaks. Architecturally it’s a transformer encoder-decoder - encoder ingests camera images and proprioception (joint states), decoder autoregressively generates the action chunk.

**Diffusion Policy** - Cheng Chi and Shuran Song at Columbia. Reframe policy learning as conditional generation. The policy doesn’t directly predict an action; it iteratively denoises Gaussian noise into an action trajectory, conditioned on the observation. Same underlying math as Stable Diffusion, but in action space instead of pixel space. The win: diffusion models are exceptionally good at representing multimodal distributions. If there are three valid ways to grab the cup, a diffusion policy represents all three; a vanilla MLP averages them and reaches for empty space between.

**Cutting edge.** Flow matching - diffusion’s more efficient cousin; same generative-modeling idea, simpler training objective, fewer sampling steps. And **hierarchical VLAs**, where a big slow brain emits "latent actions" (a compressed plan, maybe 10 Hz) and a small fast model turns those into motor torques at 500 Hz. The dual-system pattern is lifted pretty directly from Kahneman’s System 1 and System 2 - which is why Figure’s version is literally called Helix with an explicit slow "System 2" and fast "System 1."

---

### VLAs: the Main Event

A **Vision-Language-Action** model takes camera images plus a language instruction like "pick up the red cup" and outputs robot actions. The dream: train on enough robot data that the model generalizes to new cups, new tables, new instructions, the way an LLM generalizes across text.

Here’s how it actually works under the hood, because this is the piece most people hand-wave.

You start with a pretrained Vision-Language Model - something that already understands images and text, like PaLI or LLaVA. These models were originally trained to output **text tokens** describing an image. Now you want them to output **actions**. The trick: tokenize the actions. Take each dimension of the action vector - joint velocities, end-effector pose, gripper state - and discretize it into, say, 256 bins per dimension. Treat those bins as extra tokens in the vocabulary. Now "predict the next action" is just "predict the next token," which is exactly what the model was pretrained to do. You finetune on robot demonstrations, the model emits action tokens, you decode back to continuous actions.

That’s the core trick of RT-2, OpenVLA, and their descendants. Everything else - architecture choices, data mixtures, finetuning recipes - is secondary. The reason it works: language grounding transfers. If the VLM already knows what "red" and "cup" mean in pixels, you get that grounding for free, and the robot-specific finetuning only has to learn the action decoding.

**Legacy** is BC-Z and RT-1 from Google. These predated the VLM-backbone idea. They trained custom architectures from scratch on robot data, which meant they needed *a lot* of robot data - Everyday Robots inside Google collected hundreds of thousands of episodes - and still generalized poorly. The insight that landed with RT-2: internet-scale pretraining gives you the generalization, robot data gives you the grounding. You don’t need to teach the model what a cup is; you only need to teach it how its body relates to a cup.

**Current.** **OpenVLA** - open-source flagship, 7B parameters, Llama-based language backbone, trained on Open X-Embodiment. **Octo** - encoder-decoder transformer that handles any observation-action interface, enabling cross-embodiment. **π0** from Physical Intelligence - flow-matching policy on top of a VLM, the first widely-celebrated VLA from a robotics-first startup.

**Cutting edge.** **π0.5** generalizes further to long-horizon and cross-embodiment tasks. **Gemini Robotics** is Google DeepMind’s end-to-end VLA built on top of Gemini, notable because Google has both the VLM and the robot data in-house. **NVIDIA GR00T N1/N2** target humanoids specifically. **Figure Helix** runs a slow VLM alongside a fast visuomotor policy on a humanoid body.

There’s a useful sanity-check on all of this from Luke Zettlemoyer’s recent line of work on multimodal foundation models out of Meta/UW (**Chameleon**, **Transfusion**, **Mixture-of-Transformers**). The VLA discrete-tokenization trick - quantize actions into 256 bins per dimension, treat as extra vocab - is structurally the same move Chameleon made for images: tokenize everything, drop it in one softmax, train like a text model. It works, but Zettlemoyer reports the thing anyone scaling this approach hits: training stability falls apart in ways text-only setups don’t prepare you for. Image (or action) tokens compete with text tokens in a shared softmax, norms drift, and you end up bolting on KQ-norms, z-loss, and extra normalization layers just to keep a 34B model from diverging. The discretize-everything path scales, but ungracefully.

**Transfusion** keeps a single transformer but uses continuous embeddings with a diffusion loss for images and an autoregressive loss for text - multitask training, mode-switching at inference. It collapses image representation from ~1,000 tokens to ~16, trains far more stably, and is structurally the same move **π0** makes on the action side: flow matching over continuous action chunks instead of next-token prediction over discretized bins. If you squint, the diffusion-vs-tokenization debate in policy learning is the same debate as in image generation, lagging by about a year. The follow-up - **Mixture of Transformers** - gets ~3x flop efficiency by giving each modality its *own* transformer parameters with deterministic routing by modality, and surfaces an awkward finding: when you visualize embeddings from "unified" models, the modalities cluster separately anyway. The joint-representation pitch is partially a fiction; MoT just makes it explicit and pockets the efficiency. The uncomfortable implication for VLAs: "one model, all modalities" is a useful organizing fiction, but the efficient version of that idea may end up looking more like "one attention graph, modality-specific weights" than a single shared backbone.

The real open question in VLA-land is not "does this work" - it does, kind of - but "how do we make it work without huge amounts of robot data?" Few-shot imitation is the holy grail.

---

### Simulation: Three Family Trees

Three lineages matter.

**Gazebo, V-REP, PyBullet** - CPU physics, built around ROS, fine for mobile robots and simple manipulation, too slow for modern large-scale RL. Think of these as the equivalent of CPU inference for a neural network: it works, but you’re leaving two orders of magnitude on the table.

**MuJoCo** - Emo Todorov’s physics engine, originally a single-threaded C codebase loved by control theorists for its numerical stability and differentiable contact model. Google DeepMind acquired MuJoCo from Roboti LLC in 2021 and open-sourced it. **MJX** is the JAX reimplementation that runs on GPU and TPU by vectorizing over environments. If you see a control paper from DeepMind, Berkeley, or a theory-flavored lab, it’s probably MuJoCo.

**Isaac Sim / Isaac Lab** - the NVIDIA stack from earlier. PhysX under the hood, GPU-first, with high-quality RTX rendering which matters when you’re generating synthetic images to train a VLA.

**Cutting edge.**

- **Genesis** - 2024 open-source simulator that claims faster-than-Isaac performance. The jury’s still out on whether it replicates at production scale, but it gets attention.
- **NVIDIA Cosmos** - completely different animal. A generative video model trained on enormous amounts of real-world video, used as a "world simulator": give it a starting frame and an action, it predicts the next frame. This is the bleeding edge of where "simulation" and "world model" are merging.
- **3D Gaussian Splatting** - a 2023 rendering technique from Kerbl et al. that represents a scene as millions of colored 3D blobs. Photorealistic, fast. Robotics application: scan a real room, turn it into a Gaussian splat, drop a simulated robot in, and train policies that work in a pixel-perfect replica of the real space.
- **Differentiable simulators** - DiffTaichi, Brax - let you backprop *through* physics, so you can optimize controllers with gradient descent instead of RL. Mathematically gorgeous, numerically fragile, not yet mainstream.

---

### Motion Planning and Control: From RRT to cuRobo to Neural

Classical motion planning is a search problem: given start, goal, and obstacles, find a collision-free path.

**RRT** (Rapidly-exploring Random Tree) is the workhorse. Sample random configurations, connect each sample to the nearest tree node if the edge is collision-free, grow the tree until you reach the goal. Probabilistically complete, widely implemented in **OMPL**. **MoveIt** is the ROS standard that wraps OMPL plus IK plus planning-scene management. Probably seen RRT in an algorithms class, so I’ll leave it there.

The limitation: RRT produces jagged, suboptimal paths. Layer on trajectory optimization - **CHOMP**, **TrajOpt** - which takes an initial path and smooths it while keeping it collision-free via gradient descent on a cost function. All on CPU.

**cuRobo** is the GPU replacement for this whole pipeline. The speedup is enabling, not incremental: a 10 Hz MPC loop that replans from scratch every control step was impossible with CPU planners; routine with cuRobo.

**MPC** in robotics - Model Predictive Control - means: at every timestep, solve an optimization that says "given my current state, what sequence of actions minimizes this cost over the next N steps?" Take the first action, throw away the rest, re-solve next step. **OCS2** and **Crocoddyl** are the C++ libraries people use for this in locomotion.

**Cutting edge.** Diffusion-based motion planning - generate trajectories with a diffusion model trained on successful paths. Neural MPC - learn the control policy end-to-end. **Whole-body MPC for humanoids** is genuinely hard because you have 30+ DoF and contact dynamics are discontinuous (foot on ground vs. foot in air), which makes the optimization ill-conditioned.

---

### Perception: the Least Contested Battlefront

Perception for manipulation used to be hand-crafted. **ICP** (Iterative Closest Point) aligns two point clouds by iteratively matching points and minimizing distance - the workhorse for pose estimation from depth. **AprilTags** are fiducial markers you glue onto objects so the camera can see them reliably. Both still get used because they’re dead reliable when you can use them.

The modern story is simple: **use foundation models.**

- **CLIP** - joint image-text embeddings.
- **DINOv2** - self-supervised dense features. No labels needed, trained on massive unlabeled image data, and the features are shockingly good at segmentation and correspondence.
- **SAM / SAM 2** (Segment Anything, video version) - click-to-segment.
- **GroundingDINO** - text prompt → bounding boxes.

You compose them: GroundingDINO finds the red cup, SAM segments it, DINO features track it across frames, cuRobo plans to its pose. That’s a modern perception stack, assembled entirely from pretrained pieces that someone else trained for you.

**Cutting edge.** **Feature fields** - instead of 2D image features, build 3D volumetric feature fields where each point in space has a CLIP embedding baked into it. F3RM and LERF are the seminal papers. Now "pick up the red cup" becomes a 3D query: find the voxel whose CLIP embedding matches "red cup" and reach there. **V-JEPA 2** is Meta’s video foundation model that learns predictive representations from massive unlabeled video; people are starting to use its features as policy inputs. **CoTracker** and **TAPIR** are point-tracking models - given a point on one frame, track it through the whole video - surprisingly useful as policy inputs because they give the policy a stable visual reference across time.

---

### World Models

A world model is a learned simulator. Given state and action, predict the next state. **Dreamer** (Danijar Hafner, v1–v3) trained a recurrent state-space model and used it for planning - imagine the future in latent space, choose the action that leads to the best imagined outcome. Worked on Atari and toy control, struggled on real robots.

**DayDreamer** got Dreamer running on a real quadruped in a few hours of real-world training, a genuine milestone. **RoboDreamer** scaled it further.

**Cutting edge.** **Genie 2/3** from DeepMind - action-conditioned video models that generate playable worlds from a single image prompt. Cosmos again. V-JEPA 2 used for prediction. The bet: if your world model is good enough, you don’t need a simulator; you can train policies *inside* the world model. Whether that bet pays off is one of the bigger open questions of the next two years.

---

### Data Collection and Teleop

This is where robotics-specific practical ingenuity lives, and it is underrated by outsiders.

**Legacy.** Kinesthetic teaching: physically grab the robot, move it through the motion, record joint angles. Works for slow, friendly robots. SpaceMouse: 6-DoF joystick for end-effector control. Everyone in the field has used one; almost nobody likes them.

**Current.** **ALOHA** and **Mobile ALOHA** (Tony Zhao again) are leader-follower bimanual teleop rigs. Build a miniature version of the robot out of cheap parts, the full-size robot mirrors your motions. The iPhone of teleop: cheap enough that every lab has one, good enough to collect training data faster than anything else. **UMI** (Universal Manipulation Interface), from Cheng Chi, is a handheld gripper with a camera that a human uses to manipulate objects directly; the video becomes training data without the robot needing to be present during collection. That decoupling is a big deal. **GELLO** is another leader-follower design, simpler and cheaper.

**Cutting edge.** Egocentric human video - **Ego4D** and **EgoExo4D** from Meta - thousands of hours of humans doing things with cameras on their heads. Retargeting that video to a robot morphology is an active research area. And cross-embodiment pretraining: **Open X-Embodiment** is a consortium dataset combining robot data from 30+ institutions across 20+ robot types. Pretrain on everything, finetune on your specific robot. **DROID** is the large-scale Stanford-led manipulation dataset. These are the ImageNets of robotics.

---

### Sim2Real and RL: the Hardest Open Problem

Sim2real is the core open problem of embodied AI. Policies trained in simulation don’t work on real robots, because physics is hard and real sensors are noisy. The gap has roughly five components:

- **Dynamics** - friction, inertia, mass distributions are never quite what the simulator thinks.
- **Actuation** - real motors have delay, deadband, and torque limits the sim skips.
- **Perception** - lighting, textures, sensor noise on a real camera differ from sim.
- **Contact** - the single hardest one. PhysX’s contact model is not the real world’s contact model. Rigid-body assumptions fail the moment you touch anything soft.
- **Morphology** - the simulated robot’s exact kinematics may not match the real one. Small errors compound.

**Legacy answer.** **Domain randomization.** Randomize physics params, textures, lighting, and camera positions across training, and the policy becomes robust enough to transfer. OpenAI’s Rubik’s Cube robot in 2019 famously used this. Needs a lot of compute.

**Current answer.** **RMA** (Rapid Motor Adaptation) - train a policy with privileged information in sim (full ground truth), then distill it into a policy that infers the same information from observation history alone. That teacher-student pattern is everywhere now. **Residual RL** - learn a small correction policy on top of a behavior-cloned base policy, using real-world data to fix the sim-trained base.

**Cutting edge.** RL-finetune a pretrained VLA on real-world rollouts. Co-train on sim, real, and human video simultaneously (the π0.5 approach). The direction of travel: stop treating sim and real as separate domains and start treating them as two distributions the model has to handle in one training pipeline.

One note on **PPO** (Proximal Policy Optimization, Schulman et al. 2017), because it comes up everywhere. Actor-critic with a clipped objective that prevents too-large policy updates from destabilizing training. The workhorse RL algorithm for robotics. Isaac Lab + PPO at 4,096 parallel envs is the modern locomotion recipe. Billions of environment steps per GPU per day.

---

### Benchmarks: the Strategically Most Important Subfield

This deserves more attention than people give it.

**Legacy.** **Meta-World** (50 scripted tabletop tasks) and **RoboSuite** (a MuJoCo-based suite of manipulation tasks). Both are closed worlds - fixed task definitions, fixed objects, fixed everything.

**Current.** **LIBERO** for language-conditioned tasks. **RLBench**, a broad task set in CoppeliaSim. **RoboMimic**, the canonical benchmark for imitation learning algorithms. **CALVIN** for long-horizon language-conditioned manipulation.

**Cutting edge.** Open X-Embodiment isn’t really a benchmark, it’s a pretraining corpus, though people report on it. DROID similar. **BEHAVIOR-1K** from Stanford is 1,000 household tasks in high-fidelity simulation - wildly ambitious in scope. And the **crowd-authored benchmark direction** - instead of hand-designing tasks, crowdsource them from many authors and then use foundation models (CLIP embeddings, t-SNE visualization) to verify diversity - is where the research frontier actually is right now.

That direction is interesting as *research*, not just as infrastructure: the diversity-verification step turns benchmark construction into its own ML problem.

---

### Humanoids, Briefly

**Legacy** was **ZMP** (Zero Moment Point), the classical planning framework for bipedal walking. Honda’s ASIMO used it. MIT Cheetah used MPC-based control. Both worked but were hand-tuned per robot.

**Current** is learned locomotion: train a locomotion policy in Isaac Lab with PPO + domain randomization, distill it, deploy on real hardware. That’s how Unitree’s robots walk. Boston Dynamics still uses a lot of classical control; nearly everyone else has switched to learning.

**Cutting edge** is unified VLAs that handle both locomotion and manipulation - GR00T, Helix - and VR-based teleop for humanoid data collection, because demonstrating a two-arm two-leg task is hard any other way.

---

## How It All Fits Together in a Mobile Manipulation Benchmark

Mapping the stack to day-to-day work on a benchmark project:

- **Isaac Lab** is where you define tasks.
- **cuRobo** is how baseline planners solve those tasks (so you need IK working correctly, which is where the mimic joint patch came from).
- **Warp kernels** are where you write eval-time state machines and per-environment logic.
- **Slurm** is where you run eval sweeps across thousands of checkpoint-by-task pairs.
- **Containers** are how you ship the whole thing reproducibly across lab clusters and cloud.
- **Existing benchmarks** - LIBERO, RoboMimic, CALVIN - are what you compare against, and the **CLIP + t-SNE** pipeline is exactly how you prove your benchmark is more diverse than theirs.
- **VLAs** like OpenVLA and π0.5 are what you evaluate *on* your benchmark - that’s the 24/7 eval pipeline.
- **Sim2real** is the downstream research question your benchmark enables other people to study.

The benchmark sits in the middle of all of this. Every researcher in the lab who wants to evaluate a new policy uses it. Every paper that reports numbers on it cites it. That’s the leverage, and that’s why it’s the right place to stand.

---

## Closing Thought

The honest meta-point: most people who join a robotics lab try to become ML researchers and end up frustrated, because the ML layer sits on top of a stack of unglamorous engineering - simulators, kernels, schedulers, containers, IK solvers - that nobody writes papers about but that every paper depends on. Show up as the person who makes that stack work and you’re not competing with ten first-year PhDs for one author slot; you’re enabling all of their work, and ending up on all of their papers.

The field is about five years into its foundation-model era. The models are converging; the stacks underneath them are not. That’s where the next few years of leverage are.

---