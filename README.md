# CAP5619-Assignment2
In this single-student assignment, you will analyze SEC Form 8-K filings using large language models to extract new product releases and structure the data in a tabular format.

# SEC 8-K Product Announcement Extraction

Author: Hannah Fish

## Overview

This project uses Python and the Gemini API to extract named
new-product announcements from SEC Form 8-K filings.

The output includes:
- Company name
- Stock ticker
- Filing timestamp
- New product name
- Product description of fewer than 180 characters

## Data and scope

The sample contains 12 selected technology companies.

The original automated run:
- Attempted 160 filings
- Successfully processed 150 filings
- Recorded 10 failed attempts
- Extracted 128 candidate product records
- Covered filing dates from January 30, 2025, to October 29, 2026

Seven documented exclusions produced a corrected baseline of
121 records. Review was partial, so the retained records should
not be interpreted as a fully validated set of product launches.

Any manually added records and updated final counts are documented
in the report and review files.

## Methodology

1. Map ticker symbols to SEC Central Index Keys.
2. Discover recent 8-K filings through the SEC submissions API.
3. Retrieve the main filing and supported Exhibit 99 documents.
4. Convert HTML to text using BeautifulSoup.
5. Divide text into overlapping chunks.
6. Use Gemini for organization/product NER and announcement extraction.
7. Validate structured responses and supporting quotations.
8. Export CSV, pipe-separated, evidence, and audit files.

Model used for the original run: gemini-3.1-flash-lite.
Temperature: 0.

## Repository contents

- Assignment2_submission.py: extraction pipeline without report creation
- requirements.txt: required Python packages
- results/: submitted CSV and matching pipe-separated table
- report/: methodology report
- review/: evidence and documented corrections

## Setup

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure these environment variables locally:

- GEMINI_API_KEY: your own Gemini API key
- SEC_USER_AGENT: your name and contact email
- GEMINI_MODEL: optional model override

No API key is included in this repository.

## Running the code

```bash
python Assignment2.py
```

The script targets 150 successfully processed filings.
A fresh run writes to output_submission_run by default.

Running the script may make Gemini API requests and incur charges.
It is not necessary to rerun the extraction to inspect the submitted
results.

The script defaults to the original extraction prompt.
STRICT_PROMPT=1 enables optional revised rules for a separately
documented experiment.

## Reproducibility

Newly discovered recent filings may differ from the original sample.
An original candidate manifest, if available, can be selected with
the CANDIDATE_MANIFEST environment variable.

Automated output is separate from the manually corrected submission
files. A new run does not automatically reproduce manual corrections.

## Limitations

- Manual review was partial.
- Structured JSON does not guarantee correct announcement classification.
- Existing-product mentions and internal improvements can be misclassified.
- PDF/image attachments and non-Exhibit-99 attachments are outside scope.
- Repeated announcements across different filings are not automatically
  deduplicated.
- Some extraction attempts failed validation or timed out.

See the methodology report for review findings and limitations.
