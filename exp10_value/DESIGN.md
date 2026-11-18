# Exp 10: A teacher with a value function, rewarded by compression (pre-registration)

**Idea.** Exp 04-09 used myopic selectors. Every failure they had was long-horizon:

- noise deferred to the end wrecked the final model (Exp 05);
- syllabus order forgot earlier depths (Exp 06);
- blocked order locked in a shortcut (Exp 07).

A Q-learning teacher sees the future value of its choices. Its reward is the negative
prequential code length of each batch, so its return is exactly the negative code length of
the whole dataset: maximising return is minimising MDL.

**Environment.** Exp 05's one-epoch setup with 3,000 items per source (stories, facts, noise,
dupes): 375 steps, each item learnt once, Exp 04's prior as the student.

- **State:** 9 numbers. The student's mean loss on 16 sampled remaining items per source,
  the fraction of each source left, and the fraction of the episode left.
- **Action:** the source of the next batch.

**Teacher.** An MLP (9-64-64-4) trained by fitted Q-iteration:

- 30 episodes, each on a freshly drawn dataset;
- ε-greedy exploration from 1 down to 0.05 over the first 70% of episodes;
- no discount (finite horizon);
- 3 teacher seeds.

**Evaluation.** The greedy teacher and the baselines run on three held-out datasets (data
seeds 101-103):

- **random:** a uniform choice among available sources;
- **greedy:** the source with the highest current loss (myopic);
- **lp:** an Exp3 bandit rewarded by learning progress.

**Hypotheses** (means over the eval datasets, and over teacher seeds for the teacher):

- **H1. The value function compresses better.** The teacher's learnable code length is
  lower than both random's and lp's.
- **H2. It eats unavoidable data early.** The teacher's mean pick position for noise (as a
  fraction of the episode) is below 0.5. Random sits near 0.5.
- **H3. No recency collapse.** The teacher's final held-out story loss is at most random's
  plus 0.10.

Every run is reported. Hyperparameters were set once: Q lr 1e-3, buffer 20,000, 600 fit
iterations per episode, target sync every 200.
