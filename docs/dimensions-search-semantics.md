# Dimensions search semantics: what was measured

How Dimensions `full_data` treats a quoted phrase, punctuation inside it, plurals and boolean
operators. Every rule below comes from paired count queries (counts only, `full_data`, years
2020–2024, phrases escaped as the agent emits them unless marked raw), sent once each through the
agent's throttled DSL client on **2026-11-15**: 58 queries for the first pass, 67 for punctuation and stemming, 23 for a dot before a
digit, 34 for numeric plurals, "&" and "+" inside a word, a dot at the end of a word and the
"-ies"/"-es" plurals, and 65 identity queries over the emitted query shapes.
One query of the first pass ("MIDRC" alone) failed with a transient Dimensions error; its count
(362) is derived from `A NOT C` + `A AND C`.

The plural rules match Lucene's English minimal ("s") stemmer, which reproduces every measured
case; `plural_base` implements it. The query-building and dedupe rules are one module,
`adapters/dimensions_phrase.py`, and one rule table,
`MEASURED_PHRASE_RULES` in `tests/test_dimensions_phrase.py`. The insights snippet matcher and the
extraction term-context tests carry the same table; change all three together. The full-text
evidence attribution (`adapters/fulltext_evidence.py`) is a separate, deliberately looser matcher
that folds every non-alphanumeric character to a space ("GPT-4" matches "GPT-4.5", "bert base"
matches "bert_base"); it does not follow these rules.

## Phrase rules

"=" means the two forms return the same count; counts are papers.

| Rule | Evidence | Query building | Text matching |
|---|---|---|---|
| A hyphen, slash, asterisk, U+2011 or U+2013 is a space | "GPT-4" = "GPT 4" = U+2011 = U+2013 52,109; "Pangu-Weather" = "Pangu Weather" 386; "CPS-FSS" = "CPS/FSS" = "CPS FSS" 107; "O\*NET" = "O NET" 8,530 (escaped or not) | Spellings that differ only here are one search; one is kept | Separator |
| A joined form is another token | "GPT4" 12,146 vs 52,109; "PanguWeather" 96; "ONET" 4,575 | Both forms are searched when both are listed | Never equal to the split form |
| A soft hyphen (U+00AD) stays inside the token | "GPT­4" 11 vs 52,109 | Not folded | Part of the token |
| "," and ";" are separators, with or without a space | "Aurora, Colorado" = "Aurora,Colorado" = "Aurora Colorado" 14,260 vs "AuroraColorado" 10; "Washington; DC" = "Washington;DC" = "Washington DC" 1,252,209 | One search | Separator, except between two digits ("1,000"; not measured) |
| ":" next to whitespace, or between a digit and a letter, is a separator | "BioCLIP: A Vision Foundation Model" = without the colon = " : " 32; ImageNet, scikit-learn, GloVe, Add Health titles equal; "COVID-19:data" = "COVID-19 data" 47,789 vs "COVID-19data" 382 | Escaped (`\:`), one search | Separator |
| ":" between two letters is not a separator | "BioCLIP:A …" 0 vs "BioCLIP A …" 32 ("BioCLIPA …" 0) | Kept as written | Stays inside the token (dropped and kept cannot be told apart) |
| A dot between a letter and a digit is a separator | "NHANES.12" = "NHANES 12" 97 (joined 2); "UK Biobank.23" = "UK Biobank 23" 142 (joined 30) | One search | Separator: a citation superscript ("NHANES.12 We") reads as the name |
| A dot after a letter at the end of a word (before whitespace or the end) is a separator | "Pfizer Inc." = "Pfizer Inc" 31,650; "U.S. Census Bureau" = "U.S Census Bureau" 58,506 | One search | Separator |
| A dot between two digits or two letters is not a separator | "GPT-4.5" 43 vs "GPT-4 5" 471; "GPT-3.5" 26,239 vs 2,002; "Python 3.8" 21,289 vs 87; "ERA5.1" 113 vs "ERA5 1" 313; "scikit.learn" 101 vs "scikit-learn" 153,737; "U.S Census Bureau" 58,506 vs "US …" 46,326 vs "U S …" 2,247 | "GPT-4.5" is not covered by "GPT-4" | Stays inside the token |
| "+" and "&" are separators, also between two word characters; "&" is not the word "and" | "C++" = "C" 24,048,137; "Medical Imaging & Data Resource Center" = "Medical Imaging Data Resource Center" 122 vs "… and …" 170; "AT&T" = "AT T" 659,644 vs "ATT" 182,335 vs "AT and T" 1,094; "R&D" = "R D" 2,550,427 vs "RD" 1,833,482; "Q&A" = "Q A" 460,362 vs "QA" 244,734; "Pfizer+BioNTech" = "Pfizer BioNTech" 38,344 vs joined 5,247 | A name written with "&" or "and" is searched in both spellings | Separator; "and" is a word |
| An apostrophe (straight or curly) before a final "s" is dropped | "Alzheimer's" = "Alzheimer’s" = "Alzheimers" = "Alzheimer" disease 610,032 vs "Alzheimer s" 9,904; "Parkinson's Progression Markers Initiative" four ways 4,162 | One search | Dropped, then the plural folds |
| Parentheses inside a phrase are separators | "Medical Information Mart for Intensive Care (MIMIC)" = without the parentheses 2,164 (escaped or raw) | Escaped, one search | Separator |
| An inner double quote breaks the phrase and widens the query | `The "All of Us" Research Program` 412,891 vs 3,876 without the inner quotes | Never sent: replaced with a space before escaping | Separator |
| "_" is part of the token | "bert_base_uncased" 45 vs split 7,366 vs joined 11; "ERA5_Land" 184 vs 7,324 vs 1,122 | Kept | Part of the token |
| Plurals fold both ways for every token of three or more characters | "All of Us Research Program" = "…Programs" 4,857; "food security supplement(s)" 794; "electronic health record(s)" 235,144; NHANES = NHANE 91,424; LLM = LLMs; CNN = CNNs; MRI = MRIs; ERA5 = ERA5s 46,330; map = maps; "1990s" = "1990" 4,611,963; "2010s" = "2010" 12,627,656; "the 1990s" = "the 1990" 754,282 | Singular and plural are one search | A final "s" folds; a token under three characters does not ("GPT-4s" 78 vs 52,109: the token is "4s") |
| "-ies" folds to "-y"; "-es" does not fold to the bare stem | "cancer registries" = "cancer registry" 83,433 ("cancer registrie" 0); "cohort studies" = "cohort study" 1,664,062; "bounding boxes" 116,770 vs "bounding box" 157,583; "meta analyses" 3,185,482 vs "meta analysis" 3,184,312 | One search for "-ies"/"-y"; "-es" forms are searched apart | "-ies" becomes "-y"; "-es" loses only the "s" |
| Matching ignores case | "AoU" = "aou" = "AOU" 22,667 | Case variants are one search | Case-folded |
| No wildcard inside quotes | quoted "genom\*" = "genom?" 186,151; unquoted genom\* 2,721,048, genom? 1,890,730, genome 1,881,409 | "\*" and "?" are escaped inside phrases | Literal |

## Operators and precedence

| Rule | Evidence (A = "MIDRC" 362, B = "GNSS" 97,594, C = "COVID-19" 3,116,237) |
|---|---|
| Unparenthesised `A OR B AND C` drops A: it returns exactly `B AND C` | 5,693 = `B AND C` 5,693, vs `(A OR B) AND C` 5,965 and `A OR (B AND C)` 6,053 |
| NOT binds loosest | `A OR B NOT C` = `(A OR B) NOT C` 91,989, vs `A OR (B NOT C)` 92,263 |
| Lowercase "or" and "not" are not operators | `A or B` 2 vs `A OR B` 97,954; `A not C` 268 vs `A NOT C` 88 |
| Lowercase "and" behaves like AND (implicit AND) | `A and C` = `A AND C` 274 |

So the builder (`adapters/query_builder.py`) writes operators in uppercase and parenthesises every
group: an OR group is `("x" OR "y")`, a gate is `((R) AND (F))`, and a clause with excludes is
`((S) OR ((R) AND (F))) NOT (E)`; a single positive group is `(S) NOT (E)`. A bare `NOT (E)` with no
positive part is never emitted. The result therefore never depends on precedence. A flag whose
tokens occur inside a risky alias under these rules ("Research Programs" inside "All of Us Research
Program", plural folded) is dropped from that gate, since `(R) AND (F)` would then match every `R`
paper.

**Identity check (65 queries).** Every emitted shape was checked by counts: the 10 released AI-model
queries, the All of Us clauses, the data-wording alternative and the recall complement. All 18
identities hold exactly. Example, Aurora: S 9 + RF 35 − S∧RF 0 = 44; 44 − P∧E 2 = 42, and the
released, fixed, agent-seed and extraction-seed clauses all count 42.

## Escaping

Inside a quoted phrase the agent backslash-escapes `^ : ~ [ ] { } ( ) ! | & + * ? /` and the
backslash itself, the same set as the extraction server plus the backslash. A bare "&" is a lexer
error and "\&" lexes. Escaping changes no count: the escaped and raw forms of "O\*NET", "(MIMIC)" and
"Washington; DC" are equal. An inner double quote is replaced with a space before escaping (see
above).

## Not measured (handled conservatively)

The safe direction for matching is an extra search, never a lost one, so these stay inside the
token and do not fold:

- an apostrophe other than one before a final "s" ("O'Brien");
- ":" between a letter and a digit, between two digits, or at the edge of a phrase;
- a dot between a digit and a letter ("ERA5.Land");
- "," or ";" between two digits;
- a dot after a digit at the end of a word ("version 2.0."), or before a word (".NET");
- plural folding of the endings "ss" and "us" ("GNSS", "Census"), of "-ies" after "a" or "e", and of
  "-ies", "-aes", "-oes" and "-ees" kept as written (the stemmer's branches; each keeps tokens apart);
- any character not listed above;
- `full_data_exact` and `title_abstract_only` were not measured.
