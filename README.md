# FrontierAtlas Paper Metadata Extractor

`extract_paper_meta.py` parses arXiv preprint PDFs with PyMuPDF and writes `extracted_papers.json`
(title, authors, date, abstract, official GitHub repo, models, datasets, SOTA claims).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# download the 5 test papers from arxiv.org and extract
python extract_paper_meta.py --pdf-dir pdfs --download

# use PDFs you already have (named <arxiv_id>.pdf in ./pdfs)
python extract_paper_meta.py --pdf-dir pdfs --out extracted_papers.json

# optional: refine models/datasets/sota_claims with an LLM (needs ANTHROPIC_API_KEY)
python extract_paper_meta.py --pdf-dir pdfs --llm
```

Missing fields come back as `null` or `[]`; each field extractor is isolated, so one failure never
stops a paper, and problems are logged as warnings.

## How the official GitHub repo is chosen

Every `github.com/<owner>/<repo>` URL in the PDF is collected (wrapped URLs are rejoined, `.git`, `/tree/...`
and footnote digits are stripped) and scored. A link gains points for appearing on the first page
(abstract or footnote), for sitting next to release language such as "our code is available at", and for a
repo name that echoes the paper title. It loses points for appearing after the References heading
(a citation, not a release) and is rejected if it is on a denylist of common dependencies
(`pytorch/pytorch`, `huggingface/transformers`, and similar). The highest-scoring link wins; if
nothing clears the threshold, `github_url` is `null`.

## Validate the output

```bash
python validate_output.py extracted_papers.json
```

Checks required fields, the arXiv ID and date formats, that `github_url` is a clean repo link, and that every
SOTA claim has `benchmark`, `metric`, `value` and `comparison_target`. Exits with code 1 on any failure.

## Notes on `extracted_papers.json`

The JSON in this repo was compiled from the papers' arXiv pages (and the CVPR open-access PDF for VGGT's
numbers), because the build environment could not download from arxiv.org. Publication dates were checked
against the arXiv abstract pages. Abstracts are marked `[Paraphrase]`; running the script on the real PDFs
produces the verbatim abstract, so run it and overwrite the file:

```bash
python extract_paper_meta.py --pdf-dir pdfs --download
python validate_output.py
```

Models, datasets and claims in the script come from lexicon and regex heuristics, so review them, or use `--llm`.
