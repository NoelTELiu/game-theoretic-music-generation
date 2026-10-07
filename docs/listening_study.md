# Listening Study

A blinded A/B listening study was conducted as a complementary perceptual evaluation.

## Participants

The analysis uses the first **47 completed responses** from an open online voluntary call. This should be interpreted as an **online convenience sample**, not a probability-based or expertise-stratified listener panel. No eligibility screening based on musical training, classical-music familiarity, or professional music experience was applied.

Reported musical background:

- 23 participants: no formal music training
- 7 participants: less than 1 year
- 9 participants: 1-3 years
- 8 participants: more than 4 years

Median self-rated classical-music familiarity was 2 on a 1-5 scale.

## Design

- 47 participants
- 3 method comparisons
- 4 matched seeds per comparison
- 12 A/B trials total per participant
- same real musical seed within each A/B pair
- 16-step real seed prefix removed from the stimulus
- 128 generated 16th-note steps = 8 bars
- approximately 21.3 seconds per clip at 90 BPM in 4/4
- same piano timbre for both clips
- black-screen video presentation to minimize visual cues
- method identities hidden from participants

The three comparisons were:

1. No-game joint vs Lower Stackelberg
2. No-game joint vs QRE-style
3. Lower Stackelberg vs QRE-style

QRE-style and Lower Stackelberg were chosen from validation-stage selections before the expanded 77-chorale final-test analysis.

Participants answered three questions for every pair:

1. Which clip has better coordination between the upper and lower voices?
2. Which clip sounds more natural and coherent overall?
3. Overall, which clip do you prefer?

Responses were `A`, `B`, or `No clear preference`.

## Participant-level analysis

Because each participant answered four matched trials within a method comparison, pooled trial-level votes are not independent. Statistical testing is therefore performed at the participant level.

For each participant, comparison, and question, the four responses are encoded as +1 / -1 / 0 and summed. Positive or negative sums indicate a participant-level preference; zero is treated as a participant-level tie. A two-sided exact binomial test is then applied after excluding participant-level ties.

There are 3 comparisons x 3 questions = **9 participant-level tests**. Raw p-values are provided in `results/listening_study_summary.csv`, and Holm correction is applied across all nine tests.

## Main result

Only **No-game joint vs Lower Stackelberg on perceived coherence** remains below 0.05 after Holm correction. At the participant level, 24 listeners favor No-game joint, 7 favor Lower Stackelberg, and 16 are tied.

This result is most informative when interpreted together with the objective evaluation: Lower Stackelberg reliably improves local 3-gram diversity matching relative to No-game joint, yet listeners still favor No-game joint for perceived coherence. Closer matching of a local corpus statistic therefore does not automatically translate into greater perceived musical quality.

For the complete discussion, see `docs/papers/extended_technical_report.pdf`.
