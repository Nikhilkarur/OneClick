# Stage 2 — Extraction prompt (v1)

You turn a Samsung support article into troubleshooting actions for one customer's complaint. The
article is split into numbered sentences. Return JSON that matches the schema.

## The one rule that matters

Every step must come from the article. Each step cites the sentence ids (`S1`, `S2`, ...) it is
taken from, and a step must say nothing those sentences do not say. If the article does not contain
a fix, leave it out: an empty answer is better than an invented one. Never add steps from your own
knowledge, never add links, websites or email addresses.

## How to write the answer

- One `goal` per intent listed below, using its `intent_index`. Skip an intent the article does not
  address.
- `topic`: 2 to 4 words naming the problem area in Title Case (for example "Touchscreen Issues",
  "Email Connection Issues"). Do not end it with "Troubleshooting".
- `actions`: the separate fixes the article gives that help this complaint. Group steps that happen
  on the same screen into one action; a different fix is a different action. Include restart, safe
  mode, software update, factory reset and contacting support when the article gives them. Order
  does not matter.
- For each action:
  - `name`: 2 to 5 words, Title Case, starting with a verb ("Clear Email App Cache", "Enable Touch
    Sensitivity").
  - `description`: starts with "It will" and is 5 to 7 words in total ("It will clear temporary email
    app data").
  - `screen_path`: the Settings path the steps open, written as `Settings > Menu > Screen` using the
    article's own menu names. Use "" when the steps do not open a Settings screen (physical steps,
    other apps, contacting support).
  - `intent_verb`: `enable` or `disable` when the action turns a setting on or off, `set` when it
    changes a value, `open` when it opens a screen to look or act there, `check` for checks,
    `restart`, `reset` or `visit` for those, `none` otherwise.
  - `category_hint`: `critical` for restart, safe mode, software update or factory reset; `manual`
    for physical or external steps; `auto` for steps inside Settings.
  - `steps`: short imperative sentences, each ending with a period, no numbering. Stay close to the
    article's wording. A sentence with several instructions ("Go to Settings, tap Display, and then
    tap Navigation bar.") may become several steps that all cite it. Each step has `text` and
    `src_ids` (1 to 3 ids).

## Complaint

{{query}}

## Intents

{{intents}}

## Article (numbered sentences, grouped by section)

{{sentences}}
