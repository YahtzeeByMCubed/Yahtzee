Embodied DQN Agent for Yahtzee Gameplay
1. Executive Summary

This project integrates a Deep Q-Network (DQN) with a UR3e robotic manipulator and YOLOv8-based vision to play Yahtzee autonomously against a human opponent. This document outlines the complete architecture where the AI agent acts as the primary decision-maker, facilitated by a ROS 2 (Humble) Action Server and a robust logic-filtered state machine.

The system is designed to achieve a target average score of 210+ through high-fidelity simulation training. By utilizing a strictly sparse reward strategy, the AI learns holistic, forward-looking strategies over the entirety of a game rather than chasing immediate, greedy payouts. Utilizing a physical "remove-from-play" dice management strategy via the UR3e gripper, the system physically executes the AI's intent, while an Intel RealSense D435if camera provides RGB-D perception interpreted by a CNN model.
2. System Architecture

To ensure modularity and parallel development, the system is partitioned into three distinct subsystems. This "Brain-in-a-Vat" approach allows the AI to learn in a high-speed simulated environment before being connected to the visual dashboard.
2.1. Subsystem 1: The Logic Engine (Environment)

The Logic Engine acts as the "referee" and "physics" of the game. It is responsible for enforcing the rules of Yahtzee, maintaining the scorecard, and generating the state representation for the AI.

    Primary Function: Manages the 13-turn game cycle and calculates the final sparse reward.

    Measurable Requirements:

        Rule Integrity: Must validate 100% of move attempts against game rules (e.g., preventing a user from scoring in a category already filled).

        State Generation: Must produce a clean 24-dimensional state vector ($24 \times 1$) for every step.

        Reward Accuracy: Must yield a reward of 0 for all intermediate turns and a normalized terminal reward (Total Score / 100) exactly at Turn 13.

2.2. Subsystem 2: The AI Agent (The Brain)

This subsystem houses the Deep Q-Network (DQN) and the decision-making logic. It receives environmental data and selects the action with the highest predicted future value.

    Primary Function: Maps the current game state to one of 45 discrete actions.

    Measurable Requirements:

        Inference Speed: Must return an action index within 10ms of receiving a state vector.

        Performance Target: Through training, the agent must achieve a consistent average score of 190+ over a 100-game evaluation period.

        Action Logic: Must correctly apply a -infinity mask to all illegal actions provided by the Logic Engine to ensure 0% illegal move attempts.

2.3. Subsystem 3: The GUI (Dashboard)

The GUI provides a real-time visualization of the game's progress. It serves as a passive monitor for the AI's "thought process" and the current board state.

    Primary Function: Synchronizes the digital game state with a visual user interface.

    Measurable Requirements:

        Telemetry Accuracy: Must display the AI's "Confidence Chart" (Q-values for all 45 indices).

        State Sync: Every change in the Logic Engine's scorecard must be reflected on the GUI.

2.4. Subsystem Interface Specifications

Logic Engine Interface
	

Input
	

Output
	

Frequency

reset()
	

None
	

state_vector (24-D)
	

Start of Game

step(action_idx)
	

int (0–44)
	

next_state, reward, done
	

Once per Turn/Roll

get_legal_mask()
	

None
	

list (45-D, 0 or -inf)
	

Once per Turn/Roll

AI Agent Interface
	

Input
	

Output
	

Frequency

get_action(state, mask)
	

tensor (24-D), mask
	

int (Best Action Index)
	

Once per Turn/Roll

update_memory()
	

(s, a, r, s', d)
	

None
	

During Training

save_weights()
	

File Path
	

.pt Model File
	

On Demand

GUI Interface
	

Input
	

Output
	

Frequency

refresh_ui(state)
	

state_vector
	

Visual Update
	

60Hz

display_q_chart(qs)
	

list (45 Q-values)
	

Bar Chart Update
	

Once per Turn/Roll
Interface tables
1. Engine — src/engine/

Symbol
	

Signature
	

Returns
	

Purpose

dice.roll 
	

roll(rng, n=5) 
	

np.ndarray (n,) int 1–6 
	

Roll n dice 

dice.reroll 
	

reroll(rng, current, keep_mask) 
	

np.ndarray (5,) 
	

Keep where mask=True, reroll rest 

dice.face_counts 
	

face_counts(dice) 
	

np.ndarray (6,) int 
	

Count of each face 1..6 

scorecard.ScorecardManager() 
	

constructor 
	

— 
	

13-cell scorecard 

.commit 
	

commit(category_idx, dice) 
	

int 
	

Score category, raise if locked 

.get_open 
	

get_open() 
	

list[int] len 13 
	

1=open, 0=locked 

.upper_prog 
	

upper_prog() 
	

float ≤1.0 
	

upper_sum / 63 

.has_bonus 
	

has_bonus() 
	

int 0/1 
	

Yahtzee joker flag 

.is_full 
	

is_full() 
	

bool 
	

All 13 locked 

.total 
	

total() 
	

int 
	

Sum + bonuses 

yahtzee_env.is_hold 
	

is_hold(idx) 
	

bool 
	

idx ∈ [0,32) 

yahtzee_env.is_score 
	

is_score(idx) 
	

bool 
	

idx ∈ [32,45) 

yahtzee_env.decode_hold_action 
	

decode_hold_action(idx) 
	

list[bool] len 5 
	

Bit i → keep die i 

yahtzee_env.decode_score_action 
	

decode_score_action(idx) 
	

int 0–12 
	

Category index 

yahtzee_env.sparse_terminal_reward 
	

sparse_terminal_reward(total) 
	

float 
	

total / 100 

YahtzeeStateMachine() 
	

(vision_node=None, robot_client=None, seed=None) 
	

— 
	

Game orchestrator 

.reset 
	

reset() 
	

np.ndarray (24,) 
	

Initial state 

.step 
	

step(action_idx) 
	

(state, reward, done) 
	

One game step 

.construct_state_vector 
	

construct_state_vector() 
	

np.ndarray (24,) 
	

Build 24-D state 

.get_legal_mask 
	

get_legal_mask() 
	

np.ndarray (45,) 
	

0.0 legal, -inf illegal 

Constants: STATE_DIM=24, NUM_ACTIONS=45, NUM_HOLD_ACTIONS=32, SCORE_OFFSET=32, MAX_ROLLS_PER_TURN=3, TURNS_PER_GAME=13, UPPER_BONUS_THRESHOLD=63, UPPER_BONUS_VALUE=35, YAHTZEE_BONUS_VALUE=100.

Canonical category indices: 0–5 Ones..Sixes · 6 Three-of-a-Kind · 7 Four-of-a-Kind · 8 Full House (25) · 9 Small Straight (30) · 10 Large Straight (40) · 11 Yahtzee (50, +100/extra) · 12 Chance.
2. Agent — src/agent/

Symbol
	

Signature
	

Returns
	

Purpose
	

 
	

 

model.DQN() 
	

DQN(state_dim=24, action_dim=45) 
	

nn.Module 
	

MLP 24→128→128→64→45 
	

 
	

 

.forward 
	

forward(state) 
	

Tensor (B,45) 
	

Raw Q-values 
	

 
	

 

dqn_agent.Transition 
	

@dataclass(state, action, reward, next_state, done) 
	

— 
	

Replay entry 
	

 
	

 

select_legal_action 
	

select_legal_action(q, mask) 
	

int 0–44 
	

argmax(q+mask) 
	

 
	

 

PrioritizedReplayBuffer() 
	

(capacity, alpha=0.6) 
	

— 
	

Sum-tree PER 
	

 
	

 

.push 
	

push(transition) 
	

None 
	

At max priority 
	

 
	

 

.sample 
	

sample(batch_size, beta=0.4) 
	

(transitions, idx, is_weights) 
	

Priority-weighted 
	

 
	

 

.update_priorities 
	

update_priorities(idx, td_errors) 
	

None 
	

(\
	

td\
	

+ε)^α 

DQNAgent() 
	

(buffer_capacity=200000, batch_size=64, gamma=0.99, lr=1e-4, target_sync_steps=1000, eps_start=1.0, eps_end=0.05, eps_decay_steps=200000, device="cpu") 
	

— 
	

Double-DQN agent 
	

 
	

 

.get_action 
	

get_action(state, legal_mask) 
	

int 
	

Epsilon-greedy + masking 
	

 
	

 

.epsilon 
	

epsilon(step) 
	

float 
	

Linear decay 
	

 
	

 

.update_memory 
	

update_memory(s,a,r,s',done) 
	

None 
	

Push to buffer 
	

 
	

 

.learn 
	

learn() 
	

float \\| None 
	

Loss; None during warmup 
	

 
	

 

.save_weights / .load_weights 
	

(path) 
	

None 
	

torch.save/load 
	

 
	

 

train 
	

train(num_episodes=100000, save_path="models/dqn.pt", eval_interval=1000) 
	

None 
	

Outer loop 
	

 
	

 

evaluate 
	

evaluate(agent, num_games=100) 
	

float 
	

Greedy mean score 
	

 
	

 
3. Perception — src/perception/vision_node.py

Symbol
	

Signature
	

Returns
	

Purpose

VisionSanityError 
	

class(RuntimeError) 
	

— 
	

≠5 dice visible 

VisionNode (Protocol) 
	

— 
	

— 
	

Vision contract 

.get_dice_faces 
	

get_dice_faces() 
	

np.ndarray (5,) int 1–6 
	

Face values 

.get_dice_poses 
	

get_dice_poses() 
	

list (5 poses, base frame) 
	

For robot only 

RealSenseYoloVisionNode() 
	

(weights_path="models/yolov8_dice.pt") 
	

— 
	

Hardware impl 

SimVisionNode() 
	

(dice_source: Callable[[], np.ndarray]) 
	

— 
	

Sim impl, ground truth 
4. Robotics — src/robotics/

Symbol
	

Signature
	

Returns
	

Purpose

RobotResult 
	

@dataclass(success: bool, final_poses: list, error_message: str) 
	

— 
	

Move outcome 

RobotClient (Protocol) 
	

— 
	

— 
	

Robot contract 

.execute_physical_move 
	

execute_physical_move(action_index) 
	

RobotResult 
	

Run hold/score move 

.cancel_goal 
	

cancel_goal() 
	

None 
	

E-Stop, idempotent 

Ur3eRobotClient() 
	

() 
	

— 
	

ROS 2 ActionClient 

SimRobotClient() 
	

(dice_ref, rng) 
	

— 
	

In-memory fake 

PlayYahtzeeActionServer() 
	

() 
	

— 
	

ROS 2 ActionServer 
5. GUI — src/gui/gui.py

Symbol
	

Signature
	

Purpose

Dashboard(robot_client=None) 
	

constructor 
	

QMainWindow, 3-column layout 

.refresh_ui 
	

refresh_ui(state_vector: np.ndarray) 
	

60 Hz panel update 

.display_q_chart 
	

display_q_chart(q_values, legal_mask=None) 
	

Q bar chart, highlight argmax 

.show_error 
	

show_error(message: str) 
	

Status-bar errors 

ScorecardView() 
	

constructor 
	

15×2 QTableWidget 

.update_human / .update_ai 
	

(scorecard_manager) 
	

Render scorecard column 

QChartView() 
	

constructor 
	

45-bar Q chart 

.update 
	

update(q_values, legal_mask) 
	

Plot, grey illegal 
Example data for tests
State vector (24-D) — fresh game, dice = [1,3,3,4,6], roll 1
import numpy as np
state = np.array([
    1, 0, 2, 1, 0, 1,           # [0:6]  face counts (1s, 2s, 3s, 4s, 5s, 6s)
    1, 0, 0,                    # [6:9]  one-hot roll (Roll 1)
    1,1,1,1,1,1,1,1,1,1,1,1,1,  # [9:22] all 13 categories open
    0.0,                        # [22]   upper progress 0/63
    0.0,                        # [23]   no Yahtzee bonus yet
], dtype=np.float32)
assert state.shape == (24,)
Legal mask (45-D) — roll 1, all open
legal_mask = np.zeros(45, dtype=np.float32)  # everything legal

Roll 3, only categories 0, 5, 12 open:
legal_mask = np.full(45, -np.inf, dtype=np.float32)
legal_mask[32 + 0] = 0.0   # Ones
legal_mask[32 + 5] = 0.0   # Sixes
legal_mask[32 + 12] = 0.0  # Chance
# all hold actions (0..31) stay -inf — must score on roll 3
Action decoding
# decode_hold_action
assert decode_hold_action(0)  == [False]*5            # reroll all
assert decode_hold_action(31) == [True]*5             # keep all
assert decode_hold_action(0b00101) == [True, False, True, False, False]  # idx=5

# decode_score_action  (idx - 32)
assert decode_score_action(32) == 0    # Ones
assert decode_score_action(40) == 8    # Full House
assert decode_score_action(44) == 12   # Chance
Scorecard scoring fixtures
sm = ScorecardManager()

# Upper section
assert sm.commit(0, [1,1,1,2,3]) == 3      # Ones: count of 1s × 1
assert sm.commit(5, [6,6,6,6,2]) == 24     # Sixes: 4×6

# Lower section (fixed values)
assert sm.commit(8,  [3,3,5,5,5]) == 25    # Full House
assert sm.commit(9,  [1,2,3,4,6]) == 30    # Small Straight
assert sm.commit(10, [2,3,4,5,6]) == 40    # Large Straight
assert sm.commit(11, [4,4,4,4,4]) == 50    # Yahtzee
assert sm.commit(12, [1,2,3,4,5]) == 15    # Chance: sum

# Upper bonus (sum upper >= 63 → +35)
sm2 = ScorecardManager()
for cat, dice in zip(range(6), [
    [1]*5, [2]*5, [3]*5, [4]*5, [5]*5, [6,6,6,2,2]
]):
    sm2.commit(cat, dice)
# upper subtotal = 5+10+15+20+25+18 = 93 → +35 bonus
assert sm2.upper_prog() == 1.0
Reward
assert sparse_terminal_reward(0)   == 0.0
assert sparse_terminal_reward(250) == 2.5
assert sparse_terminal_reward(375) == 3.75   # design-doc target ≈190 → ~1.9
Transition for replay buffer
t = Transition(
    state=np.zeros(24, dtype=np.float32),
    action=37,
    reward=0.0,
    next_state=np.zeros(24, dtype=np.float32),
    done=False,
)
Terminal transition (only nonzero reward)
t_term = Transition(
    state=last_state,        # shape (24,)
    action=44,               # score Chance
    reward=2.45,             # total_score=245 / 100
    next_state=np.zeros(24, dtype=np.float32),
    done=True,
)
Full episode skeleton (for env smoke test)
env = YahtzeeStateMachine(seed=42)
s = env.reset()
assert s.shape == (24,)

total_reward = 0.0
done = False
while not done:
    mask = env.get_legal_mask()
    legal = np.flatnonzero(mask == 0.0)
    a = int(np.random.default_rng(0).choice(legal))
    s, r, done = env.step(a)
    total_reward += r

assert total_reward >= 0.0   # sparse: only terminal reward is nonzero
DQN forward shape
import torch
net = DQN()
out = net(torch.zeros(8, 24))
assert out.shape == (8, 45)
Seeded RNG fixture (already in test/conftest.py)
rng = np.random.default_rng(seed=42)
# rng.integers(1, 7, size=5) → reproducible dice rolls
3. AI Strategy & Data Design

3.1. DQN State Representation

A streamlined 24-dimensional state vector is constructed by the State Machine and passed to the neural network. This compact vector contains the essential "Memory" of the game, omitting irrelevant past scoring data to accelerate training.

Feature Group
	

Size
	

Description

Dice Faces
	

6
	

Count of each face (1 through 6) currently on the table.

Roll Count
	

3
	

One-hot encoded indicator for the current roll (Roll 1, Roll 2, or Roll 3).

Scorecard
	

13
	

Binary indicators representing available categories (1 = Available, 0 = Filled).

Upper Progress
	

1
	

Normalized float representing the Current Upper Score divided by 63 (the threshold for the 35-point bonus).

Yahtzee Bonus
	

1
	

Binary indicator of active Yahtzee bonus eligibility (crucial for evaluating 100-point bonus opportunities).

3.2. Network Architecture: Standard Deep Q-Network (DQN)

The agent utilizes a standard Deep Q-Network consisting of a Multi-Layer Perceptron (MLP) to evaluate the game environment.

    Direct Q-Value Mapping: The network takes the 24-dimensional state representation as input and directly maps it through fully connected layers to output 45 discrete Q-values. Each output node represents the expected future reward for taking one of the 45 possible actions from that exact state.

    Prioritized Experience Replay (PER): Because standard DQN relies on uniform evaluation and sparse rewards can be difficult to learn from initially, the agent maintains a replay buffer that tags memories based on "surprise" factor (Temporal Difference error). The agent will intentionally study rare, high-impact events (like discovering a sequence of moves that yields a high final score) far more frequently than mundane turns, massively speeding up convergence.

3.3. Action Space & Masking Logic

The AI's action space consists of 45 discrete indices representing two types of choices:

    Indices 0–31 (Hold Combinations): The 32 possible combinations of the 5 dice to keep or reroll. (Note that the dice will always be ordered in ascending order)

    Indices 32–44 (Score Categories): The 13 scorecard categories to lock in points and end the turn.

Logic Filtering (Masking):

The DQN always outputs 45 predictions per step. To prevent the agent from attempting illegal moves, a Logic Filter is applied to the raw outputs.

Valid actions are left untouched, while invalid actions are mathematically penalized with negative infinity. When the system selects the action with the maximum resulting value, illegal moves are physically impossible to execute. For example, during Roll 3, all "Hold/Reroll" actions are heavily penalized, forcing the network to select the most optimal "Score Category" action.

3.4. Training Workflow & Sparse Reward Strategy

Training occurs in a custom Python-based Yahtzee simulator, allowing for rapid convergence prior to physical deployment.

    Reward Normalization: Neural networks perform poorly with large, erratic numbers. All terminal point rewards are scaled down (e.g., divided by 100). A final game score of 250 is fed to the network as a stable reward of 2.5, preventing the network's internal math from destabilizing.

    Strict Sparse Rewards: To force the agent to learn grand strategy rather than greedy local optimization, the system utilizes a strictly sparse reward structure.

        Intermediate Steps: The agent receives 0 points for all intermediate turns and category selections. Selecting a Yahtzee midway through the game yields no immediate mathematical feedback.

        Terminal Step: The agent only receives a reward at the very end of the game (turn 13), corresponding to its total final score.

        Strategic Justification: While sparse rewards require more exploration and a longer initial training time, they eliminate the "greedy trap." The agent is forced to discover that sacrificing a short-term 20-point category to keep the Upper Section open ultimately yields a higher terminal reward via the 35-point bonus.

3.5. System Interfaces & High-Level Pseudocode

3.5.1. The Neural Network (PyTorch Architecture)

This block demonstrates the standard feed-forward DQN architecture mapping the 24-D state to the 45 possible actions directly.
Python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim=24, action_dim=45):
        super(DQN, self).__init__()
        
        # Standard Feed-Forward Neural Network
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # Outputs 45 raw Q-values
        )

    def forward(self, state):
        # Directly predict the value of all actions from the state
        q_values = self.network(state)
        return q_values

3.5.2. Action Masking Filter

This function guarantees game-rule compliance. It sits between the neural network's raw predictions and the robotic execution command.
Python
import numpy as np

def select_legal_action(raw_q_values, legal_moves_mask):
    """
    raw_q_values: Array of 45 predictions from the DQN.
    legal_moves_mask: Array of 45 values provided by the State Machine.
                      (0.0 for legal moves, -infinity for illegal moves).
    """
    # Add the mask directly to the Q-values.
    # Legal moves remain unchanged (Q + 0 = Q).
    # Illegal moves are heavily penalized (Q + -inf = -inf).
    masked_q_values = raw_q_values + legal_moves_mask
    
    # Select the index (0-44) with the highest remaining score
    best_action_index = np.argmax(masked_q_values)
    
    return best_action_index

3.5.3. The State Machine (The Environment Orchestrator)

This class handles the core game loop, formatting the state vector, and calculating the sparse rewards.
Python
class YahtzeeStateMachine:
    def __init__(self, vision_node, robot_client):
        self.vision = vision_node
        self.robot = robot_client
        self.scorecard = ScorecardManager()
        self.current_roll = 1

    def construct_state_vector(self):
        # 1. Fetch visual data from YOLOv8 (sanity check enforced here)
        dice_faces = self.vision.get_dice_faces() 
        
        # 2. Compile the 24-D flat array for the DQN
        state_vector = concatenate([
            dice_faces,                 # 6 features
            encode_roll(self.roll),     # 3 features
            self.scorecard.get_open(),  # 13 features
            self.scorecard.upper_prog(),# 1 feature
            self.scorecard.has_bonus()  # 1 feature
        ])
        return state_vector

    def step(self, action_index):
        # 1. Send the action to the ROS 2 Action Server (Robot moves)
        robot_status = self.robot.execute_physical_move(action_index)
        
        if not robot_status.success:
            raise HardwareException(robot_status.error_message)

        # 2. Update internal game logic based on what the robot just did
        self._update_game_phase(action_index)
        
        # 3. Read the new physical board state
        next_state = self.construct_state_vector()
        
        is_game_over = self.scorecard.is_full()
        
        # 4. Calculate the reward based on the Sparse Reward Strategy
        # Returns 0 unless the game is completely over
        reward = self.calculate_sparse_reward(is_game_over) 
        
        return next_state, reward, is_game_over

3.5.4. Execution Wrapper (ROS 2 Action Server Logic)

This runs purely on the robotics side. It receives the chosen action index from the State Machine and translates it into physical joint trajectories.
Python
class RobotExecutionWrapper:
    def execute_physical_move(self, action_index):
        
        # Decode the integer into a physical intent
        if action_index < 32:
            # Actions 0-31: Hold Combination chosen
            dice_to_keep = decode_hold_action(action_index)
            
            self.move_dice_to_safe_zone(dice_to_keep)
            self.sweep_and_reroll_remaining()
            
        else:
            # Actions 32-44: Score Category chosen
            category = decode_score_action(action_index)
            
            self.return_all_dice_to_start()
            # Wait for human turn confirmation via GUI
            
        return RobotResult(success=True, final_poses=self.get_poses())
4. Perception & Computer Vision

4.1. YOLOv8 Implementation

The perception node runs a YOLOv8 inference engine.

    Touching Dice: The model is trained on a published dataset specifically including dice in close proximity. The bounding box separation in YOLOv8 ensures that even if dice are touching, they are registered as distinct objects.

    Hand-Eye Calibration: The Wrapper utilizes a static coordinate transform to convert image-space coordinates into the UR3e base frame, relying on the underlying Robotics Studio 2 Package.

4.2. Data Validation (The "Sanity Check")

The State Machine enforces a hard requirement for visual data integrity before passing any information to the AI.

    Count Verification: If the total number of detected dice does not equal exactly 5, the State Machine refuses to construct a state vector.

    Feedback: A warning message is pushed to the GUI (e.g., "Error: 4 dice detected. Please clear occlusion.").

    Recovery: The robot halts operations until the user physically corrects the dice layout or engages the Manual Override on the GUI.

5. Robotics & Physical Workflow

5.1. Dice Management Strategy

To simplify the physical workspace and improve perception accuracy, the system uses a "Clear-Table" workflow:

    Identify "Keep": The DQN identifies which dice to hold.

    Remove from Play: The UR3e picks up the kept dice and moves them to a designated "Safe Zone" outside the camera's field of view.

    Reroll: The remaining dice are swept into the cup and rolled again.

    Reset: Once a category is ultimately scored, all 5 dice are returned to the starting area for the next turn.

5.2. Action Server Interface

The Execution Wrapper implements a PlayYahtzee.action interface to safely manage asynchronous robotic movements:

    Goal: Receives the 0-44 integer action index chosen by the DQN.

    Feedback: Continuously broadcasts the current operation string (e.g., "Picking die 2 of 3") and the integer count of dice moved so far.

    Result: Returns a boolean for success, an array of final poses for the kept dice, and an error string if the sequence fails.

6. Human-Robot Interaction (GUI)

The GUI, built with PyQt6 and rclpy, serves as the bridge for 1v1 play and system monitoring.

    Turn Completion: Instead of arbitrary timers, the system enforces a strict state-machine transition. It waits for a GUI Button Input ("Confirm Human Move") to trigger the transition from the Human's turn to the AI's visual scanning phase.

    Dashboard: Displays the live YOLOv8 feed overlaid with bounding boxes and inferred face values, a digital dual-scorecard, and the DQN's calculated "Q-Value Confidence" for its chosen move, providing transparent insight into the AI's logic.

    Safety Constraints: Trajectory interruption is handled by the base package, but the GUI includes a prominent software E-Stop that immediately cancels the Action Server goal. Additionally, the UR3e's internal force-torque sensors are configured to trigger a protective stop if more than 15 Newtons of resistance is encountered during a pick-and-place operation.