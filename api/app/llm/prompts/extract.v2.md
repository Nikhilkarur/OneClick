# Stage 2 — Extraction prompt (v2: select sentences)

You pick troubleshooting actions for one customer's complaint from a Samsung support article. The
article is split into numbered sentences. You do **not** write the steps: for each action you list
the ids of the article sentences that are its steps, and the engine uses those sentences word for
word. Return JSON that matches the schema.

## Rules

- Only choose sentences that tell the reader to do something (tap, go to, press, turn on, contact...)
  and that help with this complaint. Skip background, explanations and unrelated sections.
- If the article does not address the complaint at all, return an empty `goals` list. Never add
  anything that is not in the article.
- `goals`: one per distinct problem in the complaint, 1 to 3. Split only separate problems ("the
  screen is black **and** the battery drains"); symptoms of one problem are one goal.
  - `problem`: one English sentence describing that problem (fix typos, translate other languages).
  - `title`: 2 or 3 words, sentence case, naming the problem ("Touchscreen input lag").
  - `topic`: 2 to 4 words in Title Case naming the problem area, not ending in "Troubleshooting"
    ("Touchscreen Issues").
  - `domain`: Battery, Display, Camera, Performance or Other.
- `actions`: the separate fixes, each with:
  - `src_ids`: the sentence ids that are its steps, in the order to follow them.
  - `name`: 2 to 5 words, Title Case, starting with a verb ("Turn Off Full Screen Gestures").
  - `description`: "It will" plus 3 to 5 words ("It will switch to navigation buttons").
  - `screen_path`: the Settings screen the steps open, as `Settings > Menu > Screen` with the
    article's menu names, or "" when the steps do not open a Settings screen.
  - `intent_verb`: enable, disable, set, open, check, restart, reset, visit or none.
- Include restart, safe mode, software update, factory reset and contacting support when the article
  gives them for this problem. Order does not matter.

## Complaint

{{query}}

## Article (numbered sentences, grouped by section)

{{sentences}}
