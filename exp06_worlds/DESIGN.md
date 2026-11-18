# Exp 06 and 07: small worlds where learning order is the object (pre-registration)

The code is `worlds.py`. Both experiments train a 4-layer GPT (d 128) from scratch on
arithmetic over Z_23. The loss is on the answer token only. Each experiment uses seeds
1-3. Hyperparameters were set once and not tuned: AdamW at lr 1e-3, batch 128, Exp3
exploration 0.1, reward scale 0.05. A short CPU smoke run checked only that the code runs.

## Exp 06: Does compression progress rediscover the syllabus?

**Data.** Three base functions, a(x) = x+1, b(x) = 2x and c(x) = x²+1, composed to depth
1-4. A depth-d answer can only be computed by composing depth-1 pieces, so the hidden
dependency order is 1 → 2 → 3 → 4. A quarter of the depth-3 and depth-4 compositions are
never trained; they test generalisation to new compositions.

**Policies.** 8,000 steps each.

| Policy | How it picks the next batch |
|---|---|
| random | uniform over depths |
| lp | Exp3 bandit over depths, rewarded by compression progress |
| easy_first | oracle syllabus: a quarter of the steps at each depth, 1 → 4 |
| hard_first | oracle syllabus in reverse, 4 → 1 |

**Hypotheses.**

- **H1. Emergent syllabus.** Under lp, the step at which each depth's pick share peaks rises
  with depth. Kendall τ between peak step and depth is >= 0.67 in each of the 3 seeds.
- **H2. Generalisation.** lp's final accuracy on held-out depth-3 and depth-4 compositions
  (mean of the two) is at least random's plus 0.10.
- **H3. No cost.** lp's final accuracy averaged over depths 1-4 (seen compositions) is at
  least random's minus 0.02.

easy_first and hard_first are reported as reference points. No hypothesis concerns them.

## Exp 07: Does the order decide what the model believes?

**Data.** Every item is "x y # t =" with answer z = x+y mod 23.

- In 80% of the 200,000 fixed items the tag is the answer (t = z). For those items, "copy the
  tag" and "add" fit equally well.
- In the other 20% the tag is random, so only "add" fits.

Every item is learnt exactly once (one epoch). Only the order changes.

**Orders.**

| Order | Sequence |
|---|---|
| random | random permutation |
| rule_first | the 20% rule-only items first |
| rule_last | the 20% rule-only items last |
| lp | Exp3 bandit over the two sources, rewarded by compression progress |

**Belief probe.** 2,048 conflict items where the tag differs from the answer.

- Rule belief: the share answered with z.
- Shortcut belief: the share answered with t.

**Prequential code length.** The loss on each batch before learning it, in bits per item.

**Hypotheses.** The direction of the order effect is not predicted.

- **H1. Order changes belief.** The final rule belief differs by at least 0.20 between
  rule_first and rule_last (means over seeds).
- **H2. Shorter code, rule belief (MDL / Occam).** Across all 12 runs, code length and final
  rule belief correlate at Spearman ρ <= -0.5.
- **H3. Compression progress finds the rule.** lp's final rule belief is >= random's.

Every run is reported.
