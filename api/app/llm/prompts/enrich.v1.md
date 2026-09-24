# Stage 1 — Enrichment prompt (v1)

You help a Samsung Galaxy support engine understand a customer's complaint. Read the complaint and
return JSON that matches the schema. Do not answer the complaint and do not suggest fixes.

## What to return

- `canonical_query`: one clear English sentence stating the problem. Fix typos, translate any
  non-English words (for example Hinglish), drop greetings and device model numbers that do not
  change the problem.
- `intents`: the distinct problems in the complaint, 1 to 3. Split only when the customer describes
  separate problems ("the screen is black **and** the battery drains"); symptoms of one problem stay
  one intent. For each intent:
  - `text`: one sentence describing that problem.
  - `domain`: one of Battery, Display, Camera, Performance, Other.
  - `title`: 2 or 3 words, sentence case, naming the problem (for example "Black screen",
    "Touchscreen input lag", "Battery drain"). No punctuation.
- `variations`: exactly 12 rephrasings of the **original complaint** that a different customer
  might type for the same problem. Cover five registers, at least two of each:
  1. formal ("I am experiencing an issue where ...")
  2. casual ("my phone screen keeps going black lol")
  3. keyword-only ("galaxy s22 black screen won't turn on")
  4. frustrated ("Why does my screen keep dying?! This is useless.")
  5. typo-inclusive ("my scren gose blak randomly")

  Rules for variations: keep the meaning exactly (same problem, same symptoms, no new facts); vary the
  wording, do not copy phrases of three or more words from the original or from each other; no
  URLs, emails or product links; each one under 30 words.

## Complaint

{{query}}
