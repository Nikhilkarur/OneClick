# Query variations prompt (v1)

Rewrite this Samsung Galaxy customer complaint 12 different ways, as 12 different customers with the
same problem might type it. Return JSON that matches the schema. Do not answer the complaint.

- Keep the meaning exactly: same problem, same symptoms, no new facts.
- Cover five registers, at least two of each: formal ("I am experiencing an issue where ..."),
  casual ("my screen keeps going black lol"), keyword-only ("galaxy s22 black screen"), frustrated
  ("Why does my screen keep dying?!"), typo-inclusive ("my scren gose blak").
- Vary the wording: do not copy phrases of three or more words from the complaint or from each other.
- No URLs, emails or links. Each one under 30 words.

## Complaint

{{query}}
