# Lead Intelligence System

A command-line tool that qualifies sales leads from a CSV. It scores each lead from 1 to 10, decides **qualified**, **review** or **rejected**, ranks outreach priority, writes a personalized first message for qualified leads and produces a report a sales team can act on.

> **Status:** rubric and configuration are done. Items marked **[TODO]** are filled in as the code is built.

## Assumptions

- **Product:** a sales and customer data analytics platform sold to mid-market companies (about 50-5,000 employees).
- **Best-fit industries:** customer-heavy businesses such as retail, tech, marketing and media, gaming, finance, travel and hospitality.
- **Reference date:** recency is measured from a fixed date, 2024-01-20, for every file, so results are reproducible.
- **No answer key:** the data has no expected outcomes, so the rubric is my own judgment and every decision is explained.

## Data

The rubric was designed from the training file only (`leads.csv`, 29 leads). The testing and 100-lead files were not profiled, so the rubric is not tuned to them.

- 29 leads across 26 industries; company sizes 3 to 5,000 (median 180).
- Six sources, spread evenly: demo request, content download, LinkedIn outreach, sales call, referral, webinar.
- Last interaction 0 to 180 days before the reference date; no missing values.

## Qualification rubric

Each lead gets 0-10 points on four weighted factors: source and engagement intent (30%), company size fit (25%), recency (25%) and industry fit (20%). A weighted score of 7.0 or more is qualified, 5.5 to under 7.0 is review and below 5.5 is rejected **[TODO: confirm thresholds after scoring the training leads]**. A lead with a missing key field or low confidence goes to review instead of a guess. All weights, bands and industry groups are in `config.yaml`.

## How to run

**[TODO: confirm once `main.py` is built]**

```
pip install -r requirements.txt
export LLM_API_KEY=your-key-here        # Windows: set LLM_API_KEY=your-key-here
python main.py --input leads.csv --output output_report.json
```

Leads go to the LLM in batches (10 per call), not one call per lead. Failed calls are retried, then fall back to the rule-based score with a review flag.

## Output

`output_report.json` has, for every lead: score, decision, reasoning, priority rank and a confidence level. It also has the priority queue, the review queue, the rejected leads with reasons, summary statistics and 3-5 sample outreach messages **[TODO]**.

## Known limitations

- Weights and scores are judgment calls tied to the assumed product; a different product would change the industry order.
- No ground truth, so the tool cannot be checked against real conversions.
- The training file has 29 leads, one fewer than the brief's minimum of 30.
- Industries not in `config.yaml` are mapped to the nearest group by the LLM, which can be wrong; uncertain matches are flagged for review.
- Recency uses a fixed reference date and must be changed in `config.yaml` for newer data.
