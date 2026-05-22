# Project brief — Can a small, free AI tutor match a paid frontier AI for UPSC?

**For:** prayas.ai leadership, mentors, and non-technical stakeholders
**Owner:** Yeeshan — Data Scientist
**Last updated:** 2026-05-19

---

## The question we're answering

**Can a small, free, open-source AI — fine-tuned on prayas's UPSC data — answer UPSC questions as well as Google's paid Gemini API?**

If yes, prayas can run its AI tutor on its own machines for almost no per-query cost. If no, we keep paying Google's API rates but now know exactly the size of the gap and where to invest.

Either answer is useful. The point is to measure it scientifically, not guess.

---

## Why it matters

Right now, every time a prayas tutor session uses Google's `gemini-3-flash`, it costs around **₹0.80 per turn**. At our current scale (~6,000 active aspirants × ~30 turns/day), that adds up to **~₹1.5 lakh per day** in API spend — about ₹5.5 crore a year.

A small open-source model running on a regular laptop costs essentially nothing per query. If it works well enough, it changes the economics of the product.

But "well enough" is the question. A cheap tutor that gives wrong answers about Article 370 or the year of the Government of India Act is worse than no tutor — it teaches students wrong things they will write on the actual exam.

So we need to **measure** before we **decide**.

---

## What we're doing — in plain language

1. **Pick two small AIs to compare.** We've chosen `google/gemma-4-E4B-it` (Google, 4-billion-parameter, multilingual via 140+ pretrained languages) and `Qwen/Qwen3.5-4B` (Alibaba, 4-billion-parameter, explicitly multilingual across 201 languages with Hindi enumerated). Both are free, open source (Apache 2.0), and run on a regular Mac. Comparing two families tells us whether the result we get is portable, or only true for the one we picked.

2. **Teach both UPSC.** We use prayas's own data — past Prelims questions, past Mains questions with model answers, student answers graded by mentors, current-affairs articles — to fine-tune each small AI specifically on UPSC content. Same training data, same recipe; the only thing that varies is the base model. Each fine-tune takes ~5-7 hours on a laptop, so two adapters ≈ overnight.

3. **Test all four ways on 2,000 UPSC questions:**
   - **Way 1a:** Our fine-tuned Gemma.
   - **Way 1b:** Our fine-tuned Qwen.
   - **Way 2:** Google's `gemini-3-flash` with no UPSC priming.
   - **Way 3:** Google's `gemini-3-flash` shown three UPSC example questions before each test.

4. **Score every answer objectively.** Not by "looks good" — by reproducible numerical scores: Did it pick the right MCQ option? How similar is its Mains answer to the model answer? Did it adhere to UPSC's word-count rules? Did it fabricate facts that aren't in the source article?

5. **Compare statistically.** With proper confidence intervals and corrections for multiple comparisons, so we don't fool ourselves about which differences are real.

6. **Build a dashboard.** A simple web page where anyone at prayas can see the scores side-by-side and try their own questions against all three AIs.

---

## What questions we're testing

We cover six kinds of UPSC tasks — the four core comparison tasks plus two production-capability tests using prayas's own house-style prompts:

| Task | What the AI does | How we score it |
|---|---|---|
| **Prelims MCQ** | Picks A / B / C / D for an objective question | Accuracy + UPSC's actual negative-marking score |
| **Mains generation** | Writes a 150- or 250-word answer to a descriptive question | How close to the official model answer; word-count adherence; factual accuracy |
| **Mains grading** | Reads a student's answer and gives a score with strengths and improvements | How close its score is to a mentor's grade |
| **Current Affairs** | Reads a newspaper article and writes the prelims-relevant and Mains-relevant analysis | Factual faithfulness; entity accuracy; coverage of key points |
| **Prelims explanation generation** *(prayas production prompt)* | Given a question, options, and the correct answer, writes the explanation an aspirant would read after the quiz | Explanation similarity vs. gold; coverage of why each wrong option is wrong; factual accuracy of cited Articles / schemes |
| **Mains model-answer generation** *(prayas production prompt)* | Given a Mains question, writes the high-quality model answer prayas would publish — using the production prompt that defines house style | Same scoring as the Mains-generation task above, with extra emphasis on house-style structure (intro / body / conclusion + multi-dimensional framing) |

The last two are not new science — they test whether the fine-tuned model can run prayas's *actual production prompts* (once you supply them) instead of the generic prompts we use for the core comparison. Same model, same checkpoint — different prompts. This tells us whether the small AI is good enough to slot directly into the existing production pipeline.

All six tasks are in **both English and Hindi** because UPSC papers are bilingual.

---

## What "winning" looks like

We've written down our predictions *before* running the test (a scientific practice called "pre-registration"). The fine-tuned small AI wins if:

- It **beats** Google's prompted version on at least 3 of the 4 tasks at the headline metric, with statistically reliable margins.

Or, more modestly:

- It is **within 5 percentage points** of Google's prompted version on at least 3 of the 4 tasks — meaning "good enough to ship at much lower cost."

If both fail, we've learned the gap is currently unrecoverable at this size — a real, decision-relevant finding.

---

## Timeline

| Phase | What | Time |
|---|---|---|
| Setup & freeze data | Pull eval questions from prayas databases; lock them so we can't accidentally cheat | half a day |
| Pre-FT Hindi check | Run a 50-question Hindi probe on both base models, with a proper one-sided binomial significance test against random-chance baseline | 1 hour (done for Qwen; Gemma blocked — see [What's changed item 3](#whats-changed-since-the-original-plan-2026-05-14)) |
| Fine-tune two adapters | Run on a Mac M5, ~6 h each → overnight | one night |
| Score all 2,000 questions × 4 conditions | Mostly automated | 1-2 days |
| Statistical analysis + write up | Bootstrap CIs, hypothesis tests, populated report | half a day |
| Dashboard | Build the Streamlit web UI | 1-2 days |

**Total: roughly one to one-and-a-half weeks of focused work** to a publishable result.

---

## What scientific rigor we're applying

Three things prevent us from fooling ourselves:

1. **Locked-down evaluation set.** The 2,000 test questions are picked *before* training and never touched by the fine-tuning. A piece of code automatically refuses to start training if any test question slipped into the training data.

2. **Pre-registered predictions.** Before we see results, we've written down what we expect. If we end up agreeing with ourselves on every metric after the fact, that's a sign of motivated reasoning, not science.

3. **All scoring is objective.** Every one of the ~45 metrics is a deterministic mathematical function of the model output and the ground truth — same answer always gets the same score. No "AI grades AI" in v1 (see [What's changed item 2](#whats-changed-since-the-original-plan-2026-05-14)). That makes the result reproducible by anyone with the same code and removes a vendor dependency.

---

## What we know is fair to disclose as limitations

- We're using prayas's existing rubric-graded student answers as "ground truth" for the grading task. Those rubrics were originally produced by another AI, not by humans. We acknowledge this and plan a small human spot-check in a follow-up.
- The Hindi support of our small AI is in its "broadly trained" tier, not its "fluent native" tier. We measure Hindi performance separately so a weak result there doesn't get hidden in averages.
- The eval set is from prayas's own data. We don't know whether either Google or Google's training included any of it.
- Tracking covers ~50 metric × task × stratum cells; with multiple-comparison correction we keep statistical claims honest.

---

## If it works

- A version of the small AI ships inside the prayas product. Tutor responses become free at the margin.
- Annual ~₹5+ crore savings on API spend, at current scale.
- Privacy improves: student writing stops leaving prayas's servers for Google's.
- Latency improves: typically 2-3× faster than API calls.
- prayas owns the model — no vendor risk.

## If it doesn't work

- We know exactly the size of the gap, on which UPSC subjects, and at which difficulty.
- We have a baseline to measure future improvements against.
- We've built an evaluation pipeline that can re-test any future model in one day.
- The fallback is what we already use: Google's API.

Either way, we end the project with a reusable evaluation pipeline, a public-quality experiment report, and quantitative answers to "is the open-source path viable for our use case?"

---

## Where to read more

- [`experiment-report.md`](experiment-report.md) — full scientific protocol
- [`eval-design.md`](eval-design.md) — exact metrics and how they're computed
- [`architecture.md`](architecture.md) — engineering design of the test pipeline
- [`project-context.md`](project-context.md) — session-by-session decisions log
