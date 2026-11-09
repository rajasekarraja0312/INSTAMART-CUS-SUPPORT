# Instamart Customer Intelligence & Support Copilot

## Overview

A single-repository Copilot built in four connected parts. All data is synthetic and fictional (produced by `generate_data.py`); all money is in **INR**. Every language-model call goes through the offline `mock_llm.py` (no API key, network or cost). `generate_data.py` and `mock_llm.py` are the supplied files and are **unmodified**.

| Part | What it does | Main files |
| --- | --- | --- |
| A | Audit and clean raw exports, store in SQLite, SQL + pandas analysis | `part_a_clean.py`, `part_a_validate.py`, `part_a_analysis.py`, `sql/queries.sql` |
| B | RFM + K-means segments, PCA, lapse prediction | `part_b.py` |
| C | Ticket triage: guardrails, redaction, retries, validation, budget, routing | `part_c.py` |
| D | TF-IDF policy RAG assistant and a tool-calling refund agent | `policy_rag.py`, `agent.py` |

Data flow: A's `data/clean/*.csv` feed B; B's `outputs/customer_segments.csv` feeds C's routing; A's `data/instamart.db` feeds D's agent.

## Reproduce

Run from the repository root (Python 3.11+):

1. `python -m venv .venv`
2. `source .venv/bin/activate`  (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py`   (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python part_a_clean.py`
6. `python part_a_validate.py`   (prints `ALL CHECKPOINTS PASSED`)
7. `python part_a_analysis.py`
8. `python part_b.py`
9. `python part_c.py`
10. `python policy_rag.py`
11. `python agent.py`

Seeds: data generator `random.Random(42)`; K-means, splits, forests and over-sampling use `random_state=42`; retry jitter uses `random.seed(42)`.

## Part A

### Audit of the raw data

Shapes: customers (1200, 5), products (40, 4), orders (13084, 9). Missing values (count / %): customers `age` 61 (5.08%), `gender` 30 (2.50%); orders `order_amount` 53 (0.41%, these are the literal text `N/A`, which pandas reads as NaN) and `delivery_minutes` 2,541 (19.42%); every other column has 0 missing. Order columns `order_amount` is read as text (it contains `error`), `delivery_minutes` as float.

| Audit figure | Value |
| --- | --- |
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| Non-numeric order_amount (53 `N/A` + 67 `error`) | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age | 61 |
| Age outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank gender | 30 |

### Cleaning rules and how many values each changed

Customers: (a) city stripped and title-cased: 24 spellings → 6 cities (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184). (b) gender mapped to male/female/unknown: female 610, male 560, unknown 30 (blank). (c) 6 impossible ages set to missing, then all 67 missing ages (61 + 6) filled with the city median via `groupby().transform("median")`; 0 missing, range 18–60. (d) `signup_date` converted to datetime.

Orders, in order:

| # | Rule | Rows/values changed |
| --- | --- | --- |
| 1 | `order_ts` → datetime | all rows |
| 2 | Remove exact duplicates | 119 removed |
| 3 | Per `order_id` keep the record with the latest `order_ts` (drops the earlier *pending* record) | 656 removed |
| 4 | `order_amount` → numeric (bad → NaN), fill from `round(price × qty × (1 − disc/100), 2)` | 112 filled |
| 5a | Negative `delivery_minutes` → missing | 105 set (the 106th sat on a row removed in rule 2/3) |
| 5b | Completed orders: fill missing delivery with the median of that product category among completed orders (`groupby.transform`) | 774 filled |
| 5c | Cancelled orders: delivery left missing on purpose (never delivered) | 1,199 stay missing |

Result: 12,309 rows (one per order_id); 11,110 completed, 1,199 cancelled, 0 pending; 0 missing `order_amount`; completed revenue **INR 2,998,668.80**. `part_a_validate.py` asserts 12 checkpoints and prints `ALL CHECKPOINTS PASSED`. Clean data is stored in `data/clean/*.csv` and in `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as `YYYY-MM-DD HH:MM:SS` text).

### SQL results (`sql/queries.sql`, run via `pd.read_sql_query`)

| Query | Result |
| --- | --- |
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows; sum INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTEs | 550 customers above the average spend of INR 2,552.06 |
| Q6 window ranks | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK ≠ ROW_NUMBER |
| Q7 running total | 183 daily rows; last cumulative INR 2,998,668.80 |

Why ties matter in Q6: when two customers in a city have the same total spend, RANK() gives both the same rank (and skips the next number) while ROW_NUMBER() still hands out distinct consecutive numbers, so the two columns diverge for every tied row after the first tie.

### pandas cross-check

`pd.pivot_table(..., margins=True)` grand total = INR 2,998,668.80, identical to the SQL total. Per city (sum / mean / count of completed `order_amount`):

| City | Sum (INR) | Mean (INR) | Count |
| --- | --- | --- | --- |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### Exploratory statistics (completed `order_amount`)

Mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00; fences −283.50 and 645.30; 0 outliers below, 897 above. Correlation matrix (order_amount, discount_pct, quantity, delivery_minutes): order_amount–discount_pct −0.095 (largest in absolute value), order_amount–quantity +0.087, all others about 0.

- **What is this data about?** Six months of quick-commerce orders (Apr–Sep 2025) across six Indian cities, 11,110 completed orders over grocery, dairy, snacks, personal care and electronics.
- **Distributions?** Strongly right-skewed: the mean (269.91) is double the median (135.00) because a few electronics baskets are large.
- **Outliers/anomalies?** 897 orders above INR 645.30 and none below; these are genuine high-priced electronics, not errors. The planted anomalies (duplicates, negative times, bad amounts) were fixed in cleaning.
- **Relationships?** All correlations are weak; the largest is order_amount with discount_pct (−0.095). A correlation this small does not establish causation, and it is almost certainly an artefact of how discounts were generated.

## Part B

### RFM (snapshot 2025-10-01 00:00:00)

1,175 customers with a completed order. Recency: median 19, min 0, max 180 days. Frequency: mean 9.46, max 33. Monetary: mean INR 2,552.06, max INR 14,766.55.

### K-means (StandardScaler on R, F, M; `n_init=10, random_state=42`)

| K | Inertia | Silhouette |
| --- | --- | --- |
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Elbow: inertia drops steeply until K=4 (1268 → 743) and only gently afterwards (743 → 617). Silhouette peaks at K=4 (0.526). Business interpretability also favours four actionable groups. **Chosen K = 4.** The inertia of K=8 is always lower than K=2 because more centroids can only sit closer to the points; inertia therefore falls monotonically with K and, taken alone, would always choose the largest K, so we need the elbow, the silhouette and interpretability to decide.

### Segment profile (K=4)

Cluster numbers are only labels and carry no order.

| Segment id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **Premium Big-Basket Buyers** (is_premium) | 138 | 24.1 | 6.8 | 7,129.70 | VIP perks, early access to electronics drops, dedicated support |
| 2 | Loyal Daily Shoppers | 419 | 8.8 | 19.4 | 3,410.02 | Streak rewards/subscriptions; protect with fast delivery |
| 0 | Occasional Shoppers | 436 | 28.9 | 3.8 | 1,057.62 | Bundle offers to raise frequency |
| 3 | Fading / At-Risk | 182 | 109.0 | 2.2 | 685.99 | Win-back coupon and reminder before they lapse |

Premium is the segment with the highest mean monetary value. PCA(2) on the scaled RFM matrix: explained variance ratios 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`.

### Lapse problem

Cutoff 2025-09-01. Cohort: customers with ≥1 completed order before the cutoff = **1,156**. Label `lapsed = 1` if no completed order between 2025-09-01 and 2025-09-30: **443 lapsed (38.32%)**. Features (all from orders dated before the cutoff plus the clean customer table): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days. Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` → 924 train / 232 test. Baseline "always not lapsed": accuracy 61.6%, lapsed-class F1 0.00.

### Preprocessing without leakage

`ColumnTransformer` inside a scikit-learn `Pipeline`: `StandardScaler` on numerics and `OneHotEncoder(drop="first")` on city, gender and favourite category (k−1 columns each). **The encoders and scalers were fitted on the training rows only; no encoder, scaler or resampler saw test rows.** Learned (training) mean of frequency 8.115 versus full-cohort mean 8.029: they differ because the scaler's mean is computed from the 924 training rows only, which is a subset of the cohort. Confirmed in code: 0 missing values in the train and test feature columns, before and after encoding. **No data dated 2025-09-01 or later entered the features, and the segment labels from the RFM clustering were not used as features.**

### Tuning and test evaluation (lapsed = positive class)

| Model | Best params | Best CV F1 | Test confusion [[TN, FP], [FN, TP]] | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | C = 100 | 0.651 | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | max_depth 8, min_samples_leaf 10 | 0.680 | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

Grids exactly as specified (LR: C ∈ {0.01, 0.1, 1, 10, 100}; RF: max_depth ∈ {3, 5, 8} × min_samples_leaf ∈ {5, 10, 20}; `cv=5, scoring="f1"`; the preprocessing sits inside the pipeline so it is re-fitted within every fold). **Selected model: random forest** (chosen on cross-validated F1 before looking at the test set), test F1 0.655, above the 0.55 requirement. Missing a customer who is about to lapse (false negative) costs the whole future revenue of that customer, whereas wrongly sending a coupon to a loyal one (false positive) costs only a small discount, so recall matters at least as much as precision; the forest has the higher precision at equal recall and the better CV F1, and it needs no feature-scaling assumptions for the non-linear frequency/monetary interaction.

### Class-balance experiment

Training set only was over-sampled (all original rows kept, lapsed rows re-drawn with replacement, `random_state=42`) to 570 / 570. The best logistic regression refitted on it, evaluated on the unchanged test set:

| Logistic regression | Precision | Recall | F1 | Confusion |
| --- | --- | --- | --- | --- |
| Original training data | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Balanced training data | 0.589 | 0.854 | 0.697 | [[90, 53], [13, 76]] |

Recall moved most (+0.20) while precision fell (−0.055). For the retention team this means they catch many more would-be lapsers (76 of 89 instead of 58) at the price of about 21 more wrongly targeted loyal customers, a good trade when a coupon is cheap relative to a lost customer.

### Interpretation, persistence and retention list

Random-forest top 5 features: `frequency` 0.280 (few orders → likely to lapse), `monetary` 0.271 (low spenders are less attached), `recency` 0.115 (long silence is a warning sign), `avg_delivery` 0.061 (slow deliveries hurt retention), `cancel_rate` 0.052 (frequent cancellations signal frustration). Logistic-regression coefficients (standardised features): `frequency` −1.698 (each extra order sharply lowers lapse odds), `recency` +0.040 (longer silence raises lapse odds slightly once frequency is accounted for), `monetary` +0.285 (given the same frequency, big-ticket buyers who order rarely are slightly more likely to lapse). The model is saved to `outputs/lapse_model.joblib`; reloading it reproduces the original test predictions exactly (verified in code). `outputs/retention_list.csv` (`customer_id, lapse_probability, actually_lapsed`) lists the top 10 of the 40 Premium test-set customers by predicted lapse probability; **1 of the 10 actually lapsed**. This retention list is illustrative and not a leakage-free evaluation, because the Premium flag comes from whole-window RFM that includes September, the window that defines the lapse label.

## Part C

### Guardrails and redaction

4 empty tickets were set to `skipped_empty` and 4 long tickets (734 estimated tokens > MAX_INPUT_TOKENS 400) to `flagged_long`; none was sent to the model. Redaction substitutions (email → `[EMAIL]`, phone → `[PHONE]`, order id → `[ORDER_ID]`, applied in that order over all 164 tickets):

| Email | Phone | Order id | Total | Left after re-applying the patterns |
| --- | --- | --- | --- | --- |
| 55 | 91 | 132 | 278 | 0 |

The user message is exactly the redacted ticket text, and an assertion in the code checks that none of the 156 prompts that were sent contains an email, phone number or order id.

### Prompts

Zero-shot (25 estimated tokens) contains `CLASSIFY`, lists the four categories and demands JSON-only `{"category": "...", "reason": "..."}`, with no `EXAMPLES:`. Few-shot (111 estimated tokens) adds a line `EXAMPLES:` followed by four labelled examples, including a fraud example.

### Retries and validation

`call_with_retry`: max 3 retries, wait `0.05 × 2**n × random.uniform(0.9, 1.15)` seconds, one log line per retry (`RETRY ticket=T0002 retry=0 wait=0.0530s`). `mock_llm.reset()` is called before every full run. A full run saw **19 TransientLLMErrors, all recovered** (every retry is retry 0, waits within 0.045–0.0575 s); no ticket ended in a permanent error. Validation: valid JSON with an allowed category → `classified_json`; otherwise `re.findall(r"[a-z]+", text.lower())` and exactly one allowed category word → `classified_keyword`; anything else (including JSON with category `shipping`) → `flagged_invalid_output` and human review.

Status counts for one full run (both prompts): **135 classified_json, 12 classified_keyword, 9 flagged_invalid_output, 4 skipped_empty, 4 flagged_long.**

### Zero-shot vs few-shot

| Run | Classified | Accuracy | Total tokens | Tokens / classified ticket |
| --- | --- | --- | --- | --- |
| Zero-shot | 147 | 85.7% (126/147) | 8,308 | 56.5 |
| Few-shot | 147 | 100.0% (147/147) | 21,724 | 147.8 |

Zero-shot confusion table (rows = true, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
| --- | --- | --- | --- | --- |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about 91 extra tokens per classified ticket (13,416 extra tokens over 147 tickets) but removes all 21 zero-shot errors, roughly 640 tokens per corrected ticket. It is worth paying when errors are expensive: all 11 misread fraud tickets went to billing, which would miss the 2-hour security-team deadline; for low-stakes tagging, zero-shot at 56.5 tokens per ticket would suffice.

### Token budget and routing

Zero-shot with `TOKEN_BUDGET = 6000` (`outputs/triage_budget_demo.csv`): the budget stopped the run after 6,012 tokens and **43 tickets ended `budget_exhausted`** (budget used: 6000). The final `outputs/triage_results.csv` (164 rows) is the few-shot run with `TOKEN_BUDGET = 100000` and the routing: skipped/long/invalid/exhausted → `human_review` (NORMAL); fraud → `security_team` (HIGH); premium customer → `premium_desk` (HIGH); else `support_queue` (NORMAL); a customer missing from the segment file counts as not premium.

| Route | Tickets |
| --- | --- |
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 (4 empty + 4 long + 9 invalid output) |
| premium_desk | 11 |

**Rule-validated by the pipeline (147 tickets):** answers parsed as JSON or recovered by the single-keyword rule. **Sent to humans (17 tickets):** empty, too long, invalid model output (never guessed) and, in the budget demo, budget-exhausted tickets.

## Part D

### Chunking, retrieval and grounding

40-word chunks with 10-word overlap: **15 chunks** (refund 4, delivery 3, escalation 3, payment 3, privacy 2). `TfidfVectorizer()` is fitted on the chunks, queries are transformed with the same fitted vectorizer and ranked by cosine similarity; for `refund policy late delivery` all three top chunks are from `refund_policy.txt` (scores 0.272, 0.251, 0.245). TF (term frequency) measures how often a word occurs inside a chunk, and IDF (inverse document frequency) down-weights words that appear in many chunks so rare, discriminating words count more. A limitation is that TF-IDF only matches exact words: a question like "money back" will not match a chunk that says "refund".

`answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message containing `POLICY ASSISTANT` and a user message `CONTEXT:\n[file]\nchunk…\nQUESTION:\n<question>`; it returns the text and the sources.

| Evaluation | Expected doc in top 3 (of 12) | Expected fact in answer (of 12) | Fallback on unanswerable (of 2) |
| --- | --- | --- | --- |
| Overlap 10 (15 chunks) | 12 | 12 | 2 |
| Overlap 0 (13 chunks) | 12 | 10 | 2 |

With zero overlap two answerable questions break: "What delivery time makes a completed order eligible for a full refund?" (the answer, "more than 45 minutes", is cut across a chunk boundary so the best sentence no longer contains it) and "How soon is money returned when a payment fails but is deducted?" (the key fact "48 hours" is separated from "deducted" by the boundary). Overlap keeps a sentence's neighbours in the same chunk.

### Tools and the ReAct loop

Five tools (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`) are described in `TOOLS_SPEC` and called through `TOOL_MAP`; `issue_refund` and `escalate_to_human` append a JSON line (timestamp, tool, order id, amount, run id) to `outputs/refund_ledger.jsonl`. `run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` runs the loop; each run is appended to `outputs/agent_runs.jsonl`.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
| --- | --- | --- | --- | --- |
| S1 | REFUND | 128.25 | 5 | 972 |
| S2 | REFUND | 135.00 | 5 | 972 |
| S3 | ESCALATE | 1999.00 | 5 | 959 |
| S4 | DENIED | 0.00 | 4 | 665 |
| S5 | DENIED | 0.00 | 3 | 382 |
| S6 | NOT_FOUND | 0.00 | 2 | 102 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,090 |

All refund amounts equal `expected_refund_inr`.

### Stop-safely demonstrations

| Demo | Status |
| --- | --- |
| S1 with `token_limit=200` | `FALLBACK_TOKEN_LIMIT` (stopped on the 3rd model call at 399 tokens, before running the requested tool) |
| S1 with `time_limit_seconds=0` | `FALLBACK_TIMEOUT` (0 model calls) |

S7 uses the default `token_limit=6000` and still finishes all six model calls (3,090 tokens) before `FALLBACK_MAX_ITERATIONS`. Every fallback returns the steps so far and a short human-handoff string. These demo runs are saved separately in `outputs/agent_demos.jsonl`.

### Crash and resume (S1)

After every tool observation a checkpoint is written to `state/<run_id>.json` (run id, scenario id, full message list, tokens used, step number, tools executed). A simulated crash was raised right after the 2nd tool execution (checkpoint already saved), then `resume_agent("S1-crash-resume")` continued. Tools executed across both parts, in order, none repeated:

1. `get_order`
2. `search_policy`  ← crash raised here
3. `calculate_refund`
4. `issue_refund`

Final status `REFUND`, INR 128.25.

### ReAct versus plan-and-execute

Measured ReAct cost: S1 5 model calls, S5 3, S6 2, S7 6; tokens grow with each call because the whole history (including the three retrieved policy chunks) is re-sent, so S1 needs 972 tokens. A plan-and-execute agent would ask the model once for the whole list (get_order, search_policy, calculate_refund, issue/escalate) and then run it without further calls: about 2 model calls for the happy path instead of 5 and a fraction of the tokens, since history is not re-sent. Its weakness is unexpected observations: for a NOT_FOUND order (S6) or a cancelled order (S5) the fixed plan would keep executing steps that make no sense until an error triggers a replan, costing extra calls and risking a wrong action. ReAct reads each observation before choosing the next step, so it stops after 2 calls on NOT_FOUND and after 3 on a cancelled order, and it can be bounded cleanly with the iteration, token and time limits shown above. For this refund workflow I would choose ReAct (possibly with a short plan as a hint): the flow is only 4–5 steps, so the extra tokens are small, while money is involved and reacting to every observation, with safe fallbacks, matters more than saving two model calls.

## Limitations and Next Steps

- All data and the model API are simulated: the metrics describe synthetic behaviour (three planted archetypes), not real Instamart customers, and the keyword-based mock classifier is far simpler than a real LLM.
- **The Premium flag is a post-hoc whole-window label (its RFM includes September, the window that defines the lapse label), so the retention list is illustrative and not a leakage-free evaluation.**
- The lapse model is validated on one 232-row test set from a single cutoff; a rolling-origin evaluation over several cutoffs and probability calibration would give more reliable estimates, and the over-sampling experiment should be repeated with class weights or a tuned decision threshold.
- TF-IDF retrieval matches exact words only; an embedding model would handle paraphrases such as "money back" versus "refund".
- The refund agent trusts the first tool call the mock emits and its ledger is append-only without idempotency: the crash-resume demo issues a second INR 128.25 line for the S1 order (tagged with its own run id). A production system should de-duplicate refunds per order id and run tool calls under a transaction.
- Imputing completed delivery times with the category median shrinks variance; the 774 imputed values should be flagged in production.
