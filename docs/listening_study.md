# Listening Study

A blinded A/B listening study was conducted as a complementary perceptual evaluation.

## Design

- 47 participants
- 3 method comparisons
- 4 matched pieces per comparison
- 12 A/B trials total per participant
- randomized A/B order
- same seed/piece within each pair
- 16-step seed prefix removed
- 90 BPM
- 8 generated bars per clip
- same piano rendering and black-screen video presentation
- method identities hidden from participants

The three comparisons were:

1. No-game joint vs QRE
2. No-game joint vs lower-leader Stackelberg
3. QRE vs lower-leader Stackelberg

Participants answered three questions for every pair:

1. Which clip has better coordination between the upper and lower voices?
2. Which clip sounds more natural and coherent overall?
3. Which clip do you prefer overall?

Responses were `A`, `B`, or `No clear preference`.

## Participant-level analysis

For each participant, comparison, and question, the four matched trials were encoded as +1 / -1 / 0 and summed. Positive or negative sums indicated a participant-level preference; zero was treated as a tie. A two-sided exact binomial test was then applied after excluding participant-level ties.

There are 3 comparisons × 3 questions = 9 participant-level tests. Raw p-values are provided in `results/listening_study_summary.csv`. After Holm correction across all nine tests, only **No-game joint vs lower Stackelberg on perceived coherence** remained below 0.05.

This result is useful when interpreted together with the objective evaluation: lower Stackelberg reliably improved local 3-gram diversity matching relative to No-game joint, yet listeners still favored No-game joint on perceived coherence. In other words, closer matching of a local diversity statistic did not automatically translate into greater perceived musical coherence.
