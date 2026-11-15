# Lead Intelligence System

A command-line tool that qualifies sales leads from a CSV. For each lead it gives a score from 1 to 10, a decision (**qualified**, **review** or **rejected**), a plain-English reason and a priority rank. For qualified leads it also writes a personalized first message. It ends with a JSON report a sales team can act on.

## How to run

```
pip install -r requirements.txt            # pyyaml, python-dotenv
# put LLM_API_KEY=your-key in a .env file (never commit it)
python main.py --input leads.csv
python main.py --input leads_full_scale_100.csv --output output_report.json
```

Options: `--no-llm`, `--mock-llm ok` (offline test), `--no-messages`, `--tune`, `--list-models`. Model, batch size and retries are in `config.yaml`.

## Assumptions

- **Product:** a sales and customer data analytics platform, sold by Innova Solutions to mid-market companies (about 50-5,000 employees). Best-fit industries are customer-heavy ones such as retail, tech, marketing and media, finance and travel.
- **Reference date:** recency is measured from a fixed date, 2024-01-20, for every file, so results are reproducible.
- **No answer key:** the data has no expected outcomes, so the rubric is my own judgment and every decision is explained.

## Qualification rubric

Each lead gets 0-10 points on four weighted factors: source and engagement intent (30%), company size fit (25%), recency (25%) and industry fit (20%). A weighted score of 8.0 or more is qualified, 6.5 to under 8.0 is review and below 6.5 is rejected; I set these cut-offs after scoring the 29 training leads. A lead with a missing size, source or date, or a failed LLM call, goes to review instead of a guess. An unsure industry is only a note: the score decides, and a qualified lead gets a basic message. All weights, bands and industry groups are in `config.yaml`.

## How it works

1. Rules in code score source, size, recency and industry (a lookup table).
2. The LLM, called in **batches of 10 leads** (not one call per lead), maps unlisted industries to the nearest group and writes a short reasoning line. Rate limits and timeouts are retried with backoff; if a batch still fails, its leads keep the rule-based score and go to review. A bad key or model name stops the run with a clear message.
3. For qualified leads, the opening sentence and subject are fixed by lead source and the LLM writes the rest. A text with a banned claim or the wrong company gets one retry, then a template is used.

## Results on the 100-lead file

44 qualified, 38 review, 18 rejected. 10 scoring batches and 6 message batches, 0 failed; 41 messages written by the LLM and 3 basic template messages. `leads_testing.csv` (20 messy leads) ran without crashes, with missing data sent to review. See `output_report.json` for every decision, the review queue and sample messages.

## Known limitations

- Weights, thresholds and industry scores are judgment calls tied to the assumed product.
- The training file has 29 leads, one fewer than the brief's 30.
- 44% qualified is far above the real rate of about 5%; real data would need re-tuned thresholds.
- LLM output varies between runs, and a few messages are clumsy or mention a specific time such as "tomorrow".
- Referral messages say "mutual connection", though the data only says "Referral".
- No LinkedIn lead qualified, so the soft LinkedIn message style was not used in the final run.