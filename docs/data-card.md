# Data card

**Source:** generated locally with NumPy; no external dataset, personal information, or real candidate records. Default: 1,200 observations, seed 42, one observation per unique fictional candidate. Dataset SHA-256 and observed missingness/class counts are recorded in each training report.

| Column | Type / allowed values | Role |
| --- | --- | --- |
| `candidate_id` | Nonmissing unique identifier | Audit only; excluded from learning |
| `data_origin` | `synthetic` | Scope validation only; excluded from learning |
| `planning` | Numeric 1–5 or missing | Invented behavioral score |
| `collaboration` | Numeric 1–5 or missing | Invented behavioral score |
| `curiosity` | Numeric 1–5 or missing | Invented behavioral score |
| `adaptability` | Numeric 1–5 or missing | Invented behavioral score |
| `analytical_skill` | Numeric 1–5 or missing | Invented skill score |
| `communication_skill` | Numeric 1–5 or missing | Invented skill score |
| `preferred_activity` | `analysis`, `discussion`, `experimentation`, or missing | Categorical input |
| `synthetic_work_style` | `analytical`, `collaborative`, `exploratory` | Training target; not needed for inference |

Scores are independently drawn from Beta(2,2), scaled to [1,5]. Activities are uniformly sampled. Latent class scores are:

- Analytical: `0.8 * planning + analytical_skill + 0.5 * (activity == analysis)`.
- Collaborative: `collaboration + 0.8 * communication_skill + 0.5 * (activity == discussion)`.
- Exploratory: `curiosity + 0.8 * adaptability + 0.5 * (activity == experimentation)`.

Independent Gaussian noise with standard deviation 0.65 is added to each latent score, and the maximum determines the label. Numeric inputs are rounded to three decimals. Approximately 4% of numeric cells and 3% of activity values are made missing independently **after** label creation. The exact realized fractions vary with seed. This missing-completely-at-random process is a simplification of real data collection.

Labels encode the generator's assumptions, not measured personality traits. They are not Big Five, MBTI, diagnoses, or employability categories. No protected characteristics or demographic proxies are intentionally included; their absence does not establish fairness. A declaration of synthetic origin cannot detect mislabeled real data. Do not enter personal candidate data.
