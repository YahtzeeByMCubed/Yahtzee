# How the Yahtzee AI Works

A walkthrough of the DQN-based Yahtzee agent in [src/agent/](src/agent/), written for software engineers who don't have a reinforcement-learning background. Every term that's specific to RL is explained the first time it shows up.

---

## 1. The setup, in one paragraph

A neural network plays Yahtzee against itself, millions of times. Each game, it picks an action (which dice to keep, or which category to score), the rules-engine tells it what happened, and the network nudges its parameters slightly so that next time it'll make a marginally smarter pick. After enough games, the nudges add up to a policy that scores ~224 average, vs. a random legal player at ~50 and an optimal mathematical player at ~254.

The interesting parts are: how the network represents "smartness" as a number, how the math of self-improvement works, and the handful of bugs that turn this from "diverges to garbage" into "actually plays well". This doc covers all three.

---

## 2. The shape of the problem: states, actions, rewards

This is the standard "reinforcement learning" framing. It's just a loop:

```
state₀  ──action₀──>  state₁, reward₁  ──action₁──>  state₂, reward₂  ──...──>  done
```

Three things to know:

- **State** is everything the agent observes about the world *right now*. In our case, a 24-dimensional vector of numbers (described below).
- **Action** is what the agent does. In our case, an integer 0–44.
- **Reward** is a single number telling the agent "this was good" (positive) or "neutral" (0) — there are no negative rewards in Yahtzee since you can't lose points.

The agent's job: pick actions to maximize the *total* reward over a whole game.

### The state vector (24 numbers)

Layout matches design doc §3.1:

| Indices | What it represents |
|---|---|
| `[0:6]` | Count of each die face on the table — e.g., `[1, 0, 2, 1, 0, 1]` means one 1, no 2s, two 3s, one 4, no 5s, one 6. |
| `[6:9]` | Roll number, one-hot. `[1,0,0]` = roll 1, `[0,1,0]` = roll 2, `[0,0,1]` = roll 3. |
| `[9:22]` | Which scorecard categories are still open (1 = open, 0 = filled). 13 entries: ones, twos, ..., yahtzee, chance. |
| `[22]` | Upper-section progress, normalized: `upper_total / 63`. (63 is the threshold for the +35 upper bonus.) |
| `[23]` | Yahtzee-bonus eligibility flag (1 if you've scored a Yahtzee already and could earn the +100 bonus on a future Yahtzee roll). |

Notice what's *not* in there: history of past rolls, opponent state, what each die would score in each category. The agent has to figure all of that out from gradient updates. We'll come back to this — it's why the unmodified ("sparse-reward") agent plateaus around 150.

### The action space (45 integers)

| Range | Meaning |
|---|---|
| `0–31` | "Hold" actions. The 5 binary bits of the integer say which of the 5 dice to keep. Action `0` (binary `00000`) rerolls everything; action `31` (binary `11111`) keeps everything; action `5` (binary `00101`) keeps die 0 and die 2, rerolls the rest. |
| `32–44` | "Score" actions. Action `32+k` commits the current dice to category `k` (0 = ones, 1 = twos, …, 12 = chance) and ends the turn. |

So the agent emits a single integer per move and the rules engine handles the rest.

### The reward

Two reward schemes are wired up in this codebase:

- **Strict sparse** (design doc §3.4, the default): every step returns `reward = 0`, except the very last step of the game which returns `reward = final_score / 100`. The 100-divide is normalization — neural networks behave better with rewards in a small range.
- **Shaped** (`--shaped-reward` flag, used for model3): every step returns `reward = (total_after - total_before) / 100`. Hold actions return 0 (the total didn't change); score actions return the score-added/100 immediately, with bonuses appearing as spikes when their thresholds are crossed. Total reward over a game is identical to sparse.

Why this matters is in §10.

---

## 3. The "AI" itself — a neural network outputting Q-values

The neural network is a function. Specifically:

```
network: state (24 numbers)  →  Q-values (45 numbers)
```

A 4-layer MLP (multi-layer perceptron — just stacked matrix multiplies with non-linearities between them):

```
24  →  Linear(128) → ReLU
    →  Linear(128) → ReLU
    →  Linear(64)  → ReLU
    →  Linear(45)
```

That's it. The "intelligence" is encoded in those four weight matrices (~31,000 numbers total). Training adjusts those numbers.

### What a Q-value means

For each of the 45 possible actions, the network outputs a number called a **Q-value**. Read it as:

> "The total future reward I expect to collect if I take action `a` from state `s`, and play optimally afterwards."

So `Q(state, action=37)` is the network's *prediction* of how good it'd be to take action 37 (= score in category 5, "sixes") from the current state.

To pick a move, the agent runs the network forward on the current state and takes the action with the highest Q-value. That's it. The whole "AI" is "function returns 45 numbers, pick the index of the largest one".

The hard part isn't picking the move — it's getting the network to output Q-values that actually correspond to "expected future reward". That's what training does.

---

## 4. Action masking — keeping the agent legal

Most of the 45 actions are illegal at any given moment. On roll 3, you must score (so all hold actions are illegal). If you've already filled a category, you can't score there again.

The rules engine produces a **legal mask** — a 45-D vector where each entry is `0.0` (legal) or `-inf` (illegal):

```python
mask = env.get_legal_mask()
# e.g. on roll 3 with categories 0, 5, 12 still open:
# [-inf, -inf, ..., -inf,  -inf, ..., -inf, 0.0, -inf, -inf, -inf, -inf, 0.0, -inf, -inf, -inf, -inf, -inf, -inf, 0.0]
#  ^ all hold actions illegal              ^ score-ones legal      ^ score-sixes legal               ^ chance legal
```

Then the agent picks like this:

```python
q_values = network(state)        # 45 raw numbers
masked = q_values + mask         # legal entries unchanged, illegal entries = -inf
action = argmax(masked)          # can't possibly pick an illegal action
```

This guarantees **0% illegal moves** with no exception handling, no special cases. The math just makes them unreachable. (Adding `-inf` to a float gives `-inf`, and `-inf` is never the max.)

---

## 5. Q-learning — how the agent learns Q-values

This is the core idea. It's simpler than it sounds.

### The Bellman equation in plain English

If the network's Q-values were perfectly correct, this equation would hold:

```
Q(state, action)  =  reward  +  γ × max Q(next_state, any_action)
```

Read: "the value of taking action from state equals the reward you immediately get, plus a discounted version of the best value reachable from the next state."

The `γ` (gamma) is the **discount factor** — a number between 0 and 1 (we use 0.997). It's there to make the math finite and to express "future rewards are worth slightly less than immediate rewards". Without it, an infinitely-long game would have infinite expected reward.

### Training = forcing the equation to hold

Of course, the equation *doesn't* hold for an untrained network — Q-values are nonsense at first. So training picks a transition `(state, action, reward, next_state)`, computes the **target**:

```
target = reward + γ × max Q(next_state, any_action)
```

…and nudges the network's weights so its prediction `Q(state, action)` moves toward `target`. The "nudging" is just gradient descent on the squared error, exactly like any other supervised-learning task. The difference: the *targets themselves* depend on the network's own outputs (`max Q(next_state, ...)`), so they shift as the network learns. This is called **bootstrapping** and it's what makes RL training delicate.

### One annoying consequence: instability

If the network's Q-values are momentarily a bit too high, the targets it computes are also a bit too high, which makes the next round of training push the Q-values even higher, and so on. Q-values can run away to infinity. This is **divergence**, and it almost killed our training (more in §11).

---

## 6. The two networks: "online" and "target"

To stabilize training, DQN uses two copies of the same network:

- **Online network**: the one that's learning, updated every gradient step.
- **Target network**: a *frozen snapshot* of the online network from some episodes ago. Used to compute the target value in the Bellman equation.

The target network gets refreshed every `target_sync_steps = 1000` steps via a hard copy:

```python
target.load_state_dict(online.state_dict())   # copy weights, freeze grads
```

Why two networks? If you used the same network for both prediction and targets, every parameter update changes the targets, which changes the loss landscape under the optimizer's feet. The frozen target acts as a **stable goalpost** — between syncs, the online network is chasing a fixed objective. Then we move the goalpost a step.

### Double DQN

Standard DQN has a known bias: the `max` in the target equation systematically over-estimates Q-values (it's a maximization over noisy estimates). **Double DQN** fixes this with a small change:

```python
# choose the next action using the online network…
next_action = argmax(online(next_state))

# …but evaluate its Q-value using the target network
next_q = target(next_state)[next_action]

target = reward + γ × next_q
```

Decoupling action selection from action evaluation breaks the over-estimation feedback loop. One-line change with measurable benefit.

---

## 7. Experience replay (and Prioritized Experience Replay)

If the agent learned from each transition immediately and threw it away, two things go wrong:

1. **Correlation**: consecutive transitions are highly correlated (same game, same roll, same category state). SGD assumes independent samples; correlated samples make gradients noisy and unstable.
2. **One-shot learning**: each rare event (a Yahtzee, a high-scoring terminal) only gets used once.

So instead, every transition `(state, action, reward, next_state, done, mask)` gets pushed into a giant ring buffer (capacity 200,000), and each gradient step samples a random batch of 64 transitions from the buffer to learn from. That's **experience replay**.

### Why prioritized replay matters here

In strict-sparse-reward Yahtzee, **only one transition per game has a non-zero reward** (the terminal step). Among 200,000 buffered transitions, maybe 8,000 are terminal and the other 192,000 carry zero reward — totally uninformative for learning what makes a game end well.

**Prioritized Experience Replay (PER)** fixes this: each transition gets a *priority* based on how "surprising" it is to the network (specifically, the magnitude of its temporal-difference error — how wrong the network was about its Q-value). High-surprise transitions get sampled more often. The terminal-reward transitions stay surprising for a long time and get sampled disproportionately, dramatically speeding up learning.

The technical bits:

- **Sum-tree** data structure: lets you sample proportional to priority in O(log N). [src/agent/dqn_agent.py:50](src/agent/dqn_agent.py#L50).
- **β annealing**: PER introduces a sampling bias — high-priority samples are over-represented. To correct, each sample's loss is weighted by `(N × P)^(-β)`, where β starts at 0.4 and anneals to 1.0 over training. Low β early (when biases are small) avoids over-correcting; high β late (when policy is mature) ensures correct gradients. [src/agent/dqn_agent.py:241](src/agent/dqn_agent.py#L241).

You don't need to implement PER from scratch to use the codebase, but it's the single biggest reason sparse-reward training works at all here.

---

## 8. Exploration: ε-greedy

If the agent always picks the action with the highest Q-value, it'll never *try* anything new. Initially the Q-values are random, so "highest" is essentially random — but as the network starts learning, it'll lock onto a few actions and never explore the rest. This is **exploitation without exploration**.

The fix is the **ε-greedy** policy:

- With probability ε, pick a random legal action (explore).
- Otherwise, pick the highest-Q legal action (exploit).

ε starts at 1.0 (pure random) and **decays linearly** to 0.05 over the first 200,000 environment steps. Early in training, the agent explores randomly; late in training, it mostly trusts itself. The 5% residual exploration keeps the buffer fresh — without it, the buffer eventually contains only on-policy transitions and the network can't generalize.

There's one subtlety the codebase handles: ε-greedy *never* picks an illegal action. Random exploration is restricted to legal actions only ([dqn_agent.py:230](src/agent/dqn_agent.py#L230)). Otherwise we'd burn ~80% of the exploration budget on illegal moves.

---

## 9. Loss function — Huber, not MSE

Once the target value is computed, the network's prediction `Q(state, action)` should match it. The "match" is measured by a loss function. The textbook choice is **mean-squared error** (MSE): `loss = (target - prediction)²`.

We use **Huber loss** (also called smooth-L1) instead. Huber behaves like MSE for small errors but switches to absolute value for large errors:

```
huber(x) = 0.5 × x²        if |x| ≤ 1
           |x| - 0.5       if |x| > 1
```

Why this matters: the *gradient* of MSE with respect to the prediction is `2 × (prediction - target)`, which grows without bound as the error grows. So a single weird transition with a huge TD error can produce a huge gradient that yanks the network's weights into a corner. With Huber, the gradient maxes out at ±1 — bounded by construction.

Combined with **gradient clipping** (we additionally cap the L2 norm of the full gradient at 10.0, [dqn_agent.py:283](src/agent/dqn_agent.py#L283)), this prevents the value-blow-up failure mode that we hit twice during development.

---

## 10. Reward sparsity — and the experiment that broke through the plateau

The design doc (§3.4) mandates **strict sparse rewards**: 0 every step, except `final_score / 100` at the end of game 13. The argument is sound — sparse rewards prevent the agent from learning "greedy" behaviour like always taking the biggest-scoring category right now even when it sacrifices the upper bonus or a Yahtzee bonus opportunity.

The cost is that the agent has to learn from one reward per ~30 actions. Credit assignment — figuring out which of the 30 actions in a game contributed to the final score — has to be reconstructed by the network from gradient information alone, propagated backward through the Bellman equation step by step.

Empirically, with this network and reasonable compute, **strict sparse rewards plateau around eval=150**. Adding β-annealing and γ=0.997 buys some headroom but the ceiling is fundamental to the network/state-info combination.

### Reward shaping (`ShapedYahtzeeEnv`)

The model3 experiment relaxes §3.4 and gives the agent **per-commit reward**: every score action returns `score_added / 100`. Total reward over a game is unchanged — this is purely a redistribution of the same reward across the steps where it's earned.

Result: model3 cleared the design-doc 190 target within 30k episodes (~25 minutes wall-clock) and converged at eval=224. Same network, same hyperparameters, same number of training episodes — only the reward distribution differs.

The takeaway isn't that the design doc is wrong. It's that we now have a *quantified* trade-off between greedy-trap risk and learning speed: **the strict-sparse constraint costs ~74 expected score points** on this network. Whether that's worth it is a design call, not a technical one.

---

## 11. The bug that almost killed it

This is worth telling because it's a great example of a deep-learning bug that's only visible from the *training curve*, not from the code.

**Symptom**: loss diverges into the millions over 100k episodes; eval stays at random-play level (~50).

**Cause**: in the Double-DQN bootstrap, the next-state action is chosen by `argmax(online(next_state))`. But this `argmax` doesn't apply the legal mask — it can pick an *illegal* next-state action.

If the network has hallucinated a high Q-value for an illegal next-state action (which it will, because illegal actions are never sampled and so never get corrective gradient signal), the bootstrap target evaluates that illegal Q-value via the target network. The target gets corrupted. The previous-state Q's get pushed toward the corrupted target. Target sync propagates the corruption further. Q-values run to infinity. Game over.

**Fix** ([dqn_agent.py:254](src/agent/dqn_agent.py#L254)): store the next-state legal mask alongside each transition, and add it to `online(next_state)` before the argmax in the bootstrap. Same trick we use everywhere else for action selection — we just weren't using it here.

After the fix, loss stays bounded near 0.001 throughout training and the agent actually learns. This single change is the difference between "fundamentally broken" and "exceeds design-doc target".

---

## 12. The training loop, end to end

Here's the whole flow, in pseudocode that mirrors [dqn_agent.py:430](src/agent/dqn_agent.py#L430):

```python
for episode in range(num_episodes):
    state = env.reset()
    mask = env.get_legal_mask()
    done = False

    while not done:
        # 1. Choose an action (ε-greedy + masking)
        if random() < ε(global_step):
            action = random legal action from mask
        else:
            q = network.online(state)
            action = argmax(q + mask)

        # 2. Step the environment
        next_state, reward, done = env.step(action)
        next_mask = env.get_legal_mask()

        # 3. Store the transition in the replay buffer
        buffer.push(state, action, reward, next_state, done, next_mask)

        # 4. Learn (every learn_every-th step)
        if buffer.size >= batch_size and global_step % learn_every == 0:
            batch, indices, is_weights = buffer.sample(batch_size, β=current_β())

            # Bellman target with Double-DQN + masked next-state argmax
            with torch.no_grad():
                next_q_online_masked = online(next_states) + next_masks
                next_actions = argmax(next_q_online_masked)
                next_q = target(next_states).gather(next_actions)
                target_value = rewards + (1 - done) × γ × next_q

            # Online network's prediction
            predicted = online(states).gather(actions)
            td_errors = target_value - predicted

            # Huber loss, IS-weighted
            loss = (is_weights × huber(td_errors)).mean()

            # Gradient step with clipping
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(online, 10.0)
            optimizer.step()

            # Update buffer priorities based on how surprised we were
            buffer.update_priorities(indices, |td_errors|)

            # Periodic target-network sync
            if global_step % target_sync_steps == 0:
                target.copy_from(online)

        state, mask = next_state, next_mask

    # Periodic eval (greedy, no exploration, no buffer pushes)
    if episode % eval_interval == 0:
        avg = run 100 games with ε=0
        print(f"ep={episode} eval_avg={avg}")
```

That's it. Every advanced trick in here — Double DQN, PER, β annealing, masked bootstrap, Huber, grad clipping — is a tweak to some specific line in this loop.

---

## 13. The hyperparameters that matter

Quick reference for what each tunable controls:

| Flag | What it does | Default | Used |
|---|---|---|---|
| `--num-episodes` | Total games to play. | 100k | 300k–2.3M |
| `--learn-every` | Gradient step every Nth env step. Higher = faster but less sample-efficient. | 1 | 4 (3.5× speedup) |
| `--gamma` | Bellman discount. Higher = future rewards count more. | 0.99 | 0.997 |
| `--beta-anneal-steps` | Env steps for PER β to go 0.4→1.0. Smaller = anneal faster. | 1M | 1M–5M |
| `--eps-start` / `--eps-end` | ε-greedy exploration bounds. | 1.0 / 0.05 | (auto 0.1 on resume) |
| `--shaped-reward` | Use ShapedYahtzeeEnv instead of strict sparse. | off | on for model3 |
| `--resume PATH` | Load weights from a checkpoint and continue. | none | yes |
| `--checkpoint-interval` | Auto-save every N episodes (Ctrl-C safe). | 10000 | 10000–20000 |
| `--device` | `cpu` or `cuda`. Auto-detected if omitted. | auto | cpu |

The internal hyperparameters that *aren't* exposed but still matter (and could be exposed if needed):

- `lr = 5e-5` — Adam learning rate. Lower than the design doc default (1e-4) for stability after the loss-explosion incident.
- `target_sync_steps = 1000` — how often the target network is refreshed.
- `buffer_capacity = 200_000` — replay buffer size.
- `batch_size = 64` — transitions per gradient step.
- `grad_clip_norm = 10.0` — gradient L2 norm cap.

---

## 14. Running it

Train from scratch:

```bash
python main.py train --num-episodes 300000 --learn-every 4 --eval-interval 5000
```

Train from a saved checkpoint:

```bash
python main.py train --resume models/model3.pt --save-path models/model3_v2.pt --num-episodes 700000 ...
```

Watch a trained agent play:

```bash
python main.py demo --weights models/model3.pt --autoplay --speed-ms 500
```

The demo opens a Qt dashboard with a live Q-value bar chart so you can see what the network is "thinking" before each move. `--speed-ms 200` is brisk; `--speed-ms 1000` is follow-along-able.

---

## 15. Performance — three checkpoints, three stories

| Model | Setup | Eval avg | Notes |
|---|---|---|---|
| Random + mask | none | ~50 | Baseline. Picks uniformly among legal actions. |
| **model1** (`dqn_clean.pt`) | Sparse, fixed β=0.4, γ=0.99, 300k eps | **131** | Bare-minimum trained agent. Above random; below the design-doc target. |
| **model2** (`model2.pt`) | + β anneal + γ=0.997, 1.3M eps | **150** | β annealing and longer γ horizon recover ~20 points. Plateaus hard at sparse-reward ceiling. |
| **model3** (`model3.pt`) | + reward shaping, 300k eps | **224** | Cleared 190 target by ep≈30k. Hit 88% of optimal play. Demonstrates the cost of strict-sparse: ~74 score points. |
| Optimal Yahtzee policy | n/a | ~254 | Computed mathematically, not trained. |

The takeaway: **architecture changes that look small individually compound aggressively.** Going from "bootstrap argmax doesn't mask" to "+ Huber" to "+ β annealing" to "+ shaping" turned a divergent training run (eval=50, loss=10⁷) into a near-optimal player (eval=224). No single change was the silver bullet — but each one moved the ceiling, and the ceilings stack.

---

## Glossary

Quick lookup if any term in here was unfamiliar.

| Term | Plain meaning |
|---|---|
| **Action** | A choice the agent makes; in our case, an integer 0–44. |
| **Bellman equation** | `Q(s,a) = r + γ·max Q(s',a')`. The relationship Q-values must satisfy if they're correct. |
| **Bootstrapping** | Computing the target value using the network's own predictions about the future state. The reason RL is unstable. |
| **DQN** | "Deep Q-Network" — using a neural network to approximate Q-values. |
| **Double DQN** | A variant that uses one network to *pick* the next action and another to *evaluate* it. Reduces over-estimation. |
| **ε-greedy** | Pick a random action with probability ε, otherwise pick the argmax. Trades exploration for exploitation. |
| **Episode** | One full game from reset to done. |
| **γ (gamma)** | Discount factor, between 0 and 1. Makes future rewards count slightly less than immediate ones. |
| **Gradient clipping** | Capping the L2 norm of the parameter update each step. Prevents large updates from any single bad transition. |
| **Huber loss** | A loss function: quadratic for small errors, linear for large. Bounded gradient. |
| **Importance sampling weights** | The `is_weights` in PER. Correct for the sampling bias introduced by prioritized sampling. |
| **MLP** | Multi-layer perceptron. Just stacked `Linear → ReLU` layers. |
| **MSE** | Mean-squared error: `(target - prediction)²`. Common loss; we use Huber instead. |
| **PER** | Prioritized Experience Replay. Sample transitions in proportion to "surprise". |
| **Policy** | The agent's strategy: a mapping from states to actions. |
| **Q-value** | The expected total future reward from taking an action in a state. The thing the network predicts. |
| **Replay buffer** | Ring buffer of past transitions, sampled randomly during learning. |
| **Reward** | Single scalar feedback after each action. Sum over a game = the agent's objective. |
| **Reward shaping** | Adding intermediate rewards to make credit assignment easier. We do this with `ShapedYahtzeeEnv`. |
| **State** | What the agent observes — for us, the 24-D vector. |
| **Sum-tree** | A binary tree where each node stores the sum of its leaves. Lets PER sample proportional to priority in O(log N). |
| **Target network** | Frozen copy of the online network used to compute bootstrap targets. Synced periodically. |
| **TD error** | Temporal-difference error: `target - predicted`. Drives both the loss and the PER priority. |
| **Transition** | A tuple `(state, action, reward, next_state, done)` representing one step of experience. |

---

## Where to read the code

- [src/agent/model.py](src/agent/model.py) — the network architecture itself. ~30 lines.
- [src/agent/dqn_agent.py](src/agent/dqn_agent.py) — the agent, replay buffer, training loop. ~450 lines.
- [src/agent/shaped_env.py](src/agent/shaped_env.py) — the reward-shaping wrapper. ~80 lines.
- [src/engine/yahtzee_env.py](src/engine/yahtzee_env.py) — the rules engine.
- [main.py](main.py) — CLI dispatch for `train` and `demo`.

If you only read one file, read [src/agent/dqn_agent.py](src/agent/dqn_agent.py:430) starting at the `train()` function. Everything else is detail.

## Level 1 Flow Chart
```
graph TD
    %% Start of the Loop
    Start((Start Turn)) --> Obs[1. Observe State: AI looks at dice and scorecard]
    
    %% Decision Making
    Obs --> Policy{2. Decision Time: Explore or Exploit?}
    
    Policy -- "Exploration (Epsilon)" --> Random[Pick a random legal move]
    Policy -- "Exploitation (Greedy)" --> ONet[Online Network predicts Q-values]
    
    ONet --> Mask[3. Apply Legal Mask: Block illegal moves]
    Mask --> Best[Select best legal action]
    
    %% Interaction
    Random --> Action[4. Execute Action in Game]
    Best --> Action
    
    Action --> Result[5. Receive Reward and Next State]
    
    %% Memory
    Result --> Store[6. Store experience in Replay Buffer]
    
    %% The Learning Phase
    Store --> Learn{7. Time to learn?}
    
    Learn -- "No" --> Loop((Next Turn))
    Learn -- "Yes" --> Sample[8. Sample a batch of 'surprising' memories]
    
    Sample --> Compute[9. Compute Error using Online vs. Target Nets]
    Compute --> Update[10. Update Online Network weights]
    
    Update --> Sync{11. Target Sync Step?}
    Sync -- "Yes" --> Copy[Copy Online Network to Target Network]
    Sync -- "No" --> Loop
    Copy --> Loop
```

## Level 2 Flow Chart
```
graph TD
    subgraph "Interaction & Masking"
        State[Current State] --> NetIn["Online Network (Forward Pass)"]
        NetIn --> RawQ["Raw Q-Values (45-dim)"]
        LegalMask["Legal Move Mask (-inf)"] --> MaskOp["Add Mask to Q-Values"]
        RawQ --> MaskOp
        MaskOp --> FinalAction["Argmax (Best Legal Action)"]
    end

    FinalAction --> Env[[Yahtzee Environment]]
    Env --> Transition["Experience: (S, A, R, S', Mask')"]

    subgraph "Memory (Prioritized Replay Buffer)"
        Transition --> Push["Push to Buffer"]
        Push --> Tree[["Sum Tree (Stores Priorities)"]]
        Tree -.-> |"High Error = High Priority"| Sample["Sample Mini-batch"]
    end

    subgraph "The Double DQN Mechanism"
        Sample --> Batch["S, A, R, S', Next_Mask"]
        
        %% Action Selection
        Batch -- "S'" --> OnlineNext["Online Network"]
        Next_Mask["Next Legal Mask"] --> OnlineNext
        OnlineNext -- "Pick Best Next Action" --> BestNextA["Action a'"]
        
        %% Value Evaluation
        Batch -- "S'" --> TargetNet["Target Network"]
        BestNextA -.-> |"Evaluate Value of a'"| TargetNet
        TargetNet -- "Target Q-Value" --> Bellman["Compute Target: R + γQ"]
        
        %% Current Prediction
        Batch -- "S" --> OnlineCurrent["Online Network"]
        OnlineCurrent -- "Predicted Q(S,A)" --> LossCalc["Loss Function (Huber)"]
        Bellman --> LossCalc
    end

    subgraph "Optimization"
        LossCalc --> Grad["Gradient Descent"]
        Grad --> Adam[[Adam Optimizer]]
        Adam --> Weights["Update Online Weights"]
        
        LossCalc -- "TD-Error" --> TreeUpdate["Update Tree Priorities"]
        TreeUpdate --> Tree
        
        Weights -- "Every 1,000 steps" --> Sync["Sync Weights to Target Net"]
    endgraph TD
    subgraph "Interaction & Masking"
        State[Current State] --> NetIn["Online Network (Forward Pass)"]
        NetIn --> RawQ["Raw Q-Values (45-dim)"]
        LegalMask["Legal Move Mask (-inf)"] --> MaskOp["Add Mask to Q-Values"]
        RawQ --> MaskOp
        MaskOp --> FinalAction["Argmax (Best Legal Action)"]
    end

    FinalAction --> Env[[Yahtzee Environment]]
    Env --> Transition["Experience: (S, A, R, S', Mask')"]

    subgraph "Memory (Prioritized Replay Buffer)"
        Transition --> Push["Push to Buffer"]
        Push --> Tree[["Sum Tree (Stores Priorities)"]]
        Tree -.-> |"High Error = High Priority"| Sample["Sample Mini-batch"]
    end

    subgraph "The Double DQN Mechanism"
        Sample --> Batch["S, A, R, S', Next_Mask"]
        
        %% Action Selection
        Batch -- "S'" --> OnlineNext["Online Network"]
        Next_Mask["Next Legal Mask"] --> OnlineNext
        OnlineNext -- "Pick Best Next Action" --> BestNextA["Action a'"]
        
        %% Value Evaluation
        Batch -- "S'" --> TargetNet["Target Network"]
        BestNextA -.-> |"Evaluate Value of a'"| TargetNet
        TargetNet -- "Target Q-Value" --> Bellman["Compute Target: R + γQ"]
        
        %% Current Prediction
        Batch -- "S" --> OnlineCurrent["Online Network"]
        OnlineCurrent -- "Predicted Q(S,A)" --> LossCalc["Loss Function (Huber)"]
        Bellman --> LossCalc
    end

    subgraph "Optimization"
        LossCalc --> Grad["Gradient Descent"]
        Grad --> Adam[[Adam Optimizer]]
        Adam --> Weights["Update Online Weights"]
        
        LossCalc -- "TD-Error" --> TreeUpdate["Update Tree Priorities"]
        TreeUpdate --> Tree
        
        Weights -- "Every 1,000 steps" --> Sync["Sync Weights to Target Net"]
    end
    ```