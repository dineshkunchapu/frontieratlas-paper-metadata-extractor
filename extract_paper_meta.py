#!/usr/bin/env python3
"""FrontierAtlas paper metadata extractor.

Parses arXiv preprint PDFs with PyMuPDF and emits extracted_papers.json:
header metadata, official GitHub repo, models, datasets, SOTA claims.

Usage:
    python extract_paper_meta.py --pdf-dir pdfs --download
    python extract_paper_meta.py --pdf-dir pdfs --ids 2503.18382 2503.11651
    python extract_paper_meta.py --pdf-dir pdfs --llm      # optional LLM refinement

Every field extractor is wrapped so one failure never aborts a paper;
missing fields come back as null / [] and the problem is logged.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    sys.exit("PyMuPDF missing: pip install -r requirements.txt")

log = logging.getLogger("extract_paper_meta")

DEFAULT_IDS = ["2503.18382", "2510.13998", "2607.18859", "2507.23370", "2503.11651"]

# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
SUPERSCRIPTS = "¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§¶∗⋆♠♣♦♥"
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


def strip_latex(s: str) -> str:
    s = re.sub(r"\$\^\{?[^$}]*\}?\$", "", s)               # $^3$ -> ''
    s = re.sub(r"\\(?:textbf|textit|emph|mathbf|mathrm|text)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)                    # inline math -> content
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    return s.replace("{", "").replace("}", "")


def clean_text(s: str) -> str:
    s = strip_latex(s)
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)            # de-hyphenate line breaks
    s = s.replace("\u00ad", "")
    s = re.sub(r"\s*\n\s*", " ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def clean_author(name: str) -> str:
    name = EMAIL_RE.sub("", name)
    name = re.sub(f"[{re.escape(SUPERSCRIPTS)}]", "", name)
    name = re.sub(r"\d+", "", name)
    name = strip_latex(name)
    return re.sub(r"\s{2,}", " ", name).strip(" ,;&")


# --------------------------------------------------------------------------
# PDF reading
# --------------------------------------------------------------------------
class Paper:
    def __init__(self, pdf_path: Path, arxiv_id: str):
        self.arxiv_id = arxiv_id
        self.doc = fitz.open(pdf_path)
        self.pages = [p.get_text("text") for p in self.doc]
        self.full_text = "\n".join(self.pages)
        self.first_page = self.pages[0] if self.pages else ""

    def first_page_lines(self):
        """Yield (text, max_font_size, y0) per line on page 1, in reading order."""
        out = []
        for block in self.doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                out.append((text, max(s["size"] for s in spans), line["bbox"][1]))
        out.sort(key=lambda t: t[2])
        return out


# --------------------------------------------------------------------------
# Header metadata
# --------------------------------------------------------------------------
AFFIL_WORDS = re.compile(
    r"universit|institute|inc\b|corp|research|laborator|lab\b|college|school|"
    r"department|google|meta\b|microsoft|baidu|bytedance|oxford|@|china|usa|team|"
    r"academy|centre|center|group", re.I)


def extract_title(p: Paper) -> str | None:
    lines = [l for l in p.first_page_lines() if len(l[0]) > 2 and not l[0].lower().startswith("arxiv:")]
    if not lines:
        return None
    top = max(l[1] for l in lines)
    parts = [l[0] for l in lines if abs(l[1] - top) < 1.0]
    return clean_text(" ".join(parts)) or None


def extract_authors(p: Paper, title: str | None) -> list[str]:
    lines = p.first_page_lines()
    if not lines:
        return []
    top = max(l[1] for l in lines)
    candidates = []
    started = False
    for text, size, _ in lines:
        if abs(size - top) < 1.0:
            started = True
            continue
        if not started:
            continue
        if re.match(r"abstract", text, re.I):
            break
        if AFFIL_WORDS.search(text) or EMAIL_RE.search(text):
            if candidates:          # affiliations follow the names; stop there
                break
            continue
        candidates.append(text)
    blob = " ".join(candidates)
    blob = re.sub(r"\band\b", ",", blob)
    names = []
    for chunk in re.split(r"[,;]", blob):
        n = clean_author(chunk)
        if 3 <= len(n) <= 60 and re.search(r"[A-Za-z]", n) and len(n.split()) <= 5:
            names.append(n)
    return list(dict.fromkeys(names))


def extract_abstract(p: Paper) -> str | None:
    m = re.search(r"\bAbstract\b[\s\-—:.]*\n?(.*?)(?=\n\s*(?:1\.?\s+Introduction|I\.\s+INTRODUCTION|Keywords?|Index Terms|1\s*\n\s*Introduction)\b)",
                  p.first_page + "\n" + (p.pages[1] if len(p.pages) > 1 else ""),
                  re.S | re.I)
    return clean_text(m.group(1)) if m else None


def extract_date(p: Paper) -> str | None:
    m = re.search(r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[[\w.\-]+\]\s*(\d{1,2}\s+\w{3,9}\s+\d{4})", p.full_text[:4000])
    if m:
        for fmt in ("%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(m.group(1), fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return fetch_date_from_arxiv_api(p.arxiv_id)


def fetch_date_from_arxiv_api(arxiv_id: str) -> str | None:
    try:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        with urllib.request.urlopen(url, timeout=15) as r:
            xml = r.read().decode()
        m = re.search(r"<published>(\d{4}-\d{2}-\d{2})", xml)
        return m.group(1) if m else None
    except Exception as e:  # offline is fine
        log.warning("%s: arXiv API date lookup failed (%s)", arxiv_id, e)
        return None


# --------------------------------------------------------------------------
# GitHub: official repo vs third-party links
# --------------------------------------------------------------------------
GH_RE = re.compile(r"github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)", re.I)
# Well-known dependencies / frameworks that are cited, not released by authors.
DEPENDENCY_DENYLIST = {
    "pytorch/pytorch", "huggingface/transformers", "huggingface/datasets",
    "tensorflow/tensorflow", "numpy/numpy", "scipy/scipy", "opencv/opencv",
    "facebookresearch/detectron2", "facebookresearch/dinov2", "google/jax",
    "openai/gpt-2", "microsoft/deepspeed", "nvidia/apex", "vllm-project/vllm",
    "ggerganov/llama.cpp", "lm-sys/fastchat", "princeton-nlp/swe-bench",
    "swe-bench/swe-bench", "paddlepaddle/paddle", "scikit-learn/scikit-learn",
}
RELEASE_CUES = re.compile(
    r"(our|the|we)\s+(code|implementation|source|models?|repository|project)|"
    r"(code|models?|implementation|resources?)\s+(is|are|will be)?\s*(publicly\s+)?(available|released|open[- ]sourced?)|"
    r"open[- ]source|released at|available at|project page", re.I)


def normalise_repo(owner: str, repo: str) -> str:
    repo = re.sub(r"\.git$", "", repo).rstrip(".,;:)")
    repo = re.sub(r"\.\d$", "", repo)          # footnote marker glued on: "PaddleOCR.1"
    return f"https://github.com/{owner}/{repo}"


def find_references_start(text: str) -> int:
    idx = [m.start() for m in re.finditer(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text)]
    return idx[-1] if idx else int(len(text) * 0.85)


def extract_github(p: Paper, title: str | None, authors: list[str]) -> tuple[str | None, bool]:
    text = re.sub(r"(github\.com/[^\s]*)\n\s*([^\s]+)", r"\1\2", p.full_text)  # rejoin wrapped URLs
    ref_start = find_references_start(text)
    first_page_end = len(p.first_page)
    paper_tokens = {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", title or "")}
    author_tokens = {w.lower() for a in authors for w in a.split()}

    scores: dict[str, float] = {}
    for m in GH_RE.finditer(text):
        url = normalise_repo(m.group(1), m.group(2))
        key = url.lower().removeprefix("https://github.com/")
        if key in DEPENDENCY_DENYLIST:
            scores[url] = scores.get(url, 0) - 10
            continue
        s = 0.0
        pos = m.start()
        context = text[max(0, pos - 200): pos + 60]
        if pos < first_page_end:
            s += 4                       # abstract / first-page footnote
        if RELEASE_CUES.search(context):
            s += 3                       # "our code is available at ..."
        if pos > ref_start:
            s -= 6                       # bibliography entry => cited, not released
        if re.search(r"\[\d+\]|et al\.|\b(19|20)\d\d\b", context) and pos > ref_start:
            s -= 2
        repo_l = m.group(2).lower().removesuffix(".git")
        if repo_l in paper_tokens or any(t in repo_l for t in paper_tokens if len(t) > 4):
            s += 2                       # repo name echoes the paper title
        s += 1                           # base: appears at all (repeats add up below)
        scores[url] = scores.get(url, 0) + s
    if not scores:
        return None, False
    best, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score < 3:                   # only weak / reference-section evidence
        return None, False
    return best, True


# --------------------------------------------------------------------------
# Models / datasets / SOTA claims (heuristic; --llm refines)
# --------------------------------------------------------------------------
MODEL_PATTERNS = [
    r"\bGPT-?[0-9][\w.\-]*(?:\s?mini|\s?turbo)?\b", r"\bClaude[\s\-]?[0-9.]*\s?(?:Opus|Sonnet|Haiku)?[\s\-]?[0-9.]*\b",
    r"\bGemini[\s\-]?[0-9.]+\s?(?:Pro|Flash)?\b", r"\bQwen[\w.\-]*\b", r"\bLLaMA[\s\-]?[0-9.]*\w*\b",
    r"\bLlama[\s\-]?[0-9.]*\w*\b", r"\bGemma[\s\-]?[0-9]*[\w\-]*\b", r"\bDeepSeek[\w.\-]*\b",
    r"\bMiniMax[\w.\-]*\b", r"\bGLM[\s\-]?[0-9.]+\b", r"\bBitNet[\w.\-]*\b", r"\bUniMERNet\b",
    r"\bPP-\w+(?:-[LSM])?\b", r"\bDUSt3R\b", r"\bMASt3R\b", r"\bVGGSfM\b", r"\bCoTracker\w*\b",
    r"\bDINO\w*\b", r"\bMiniLM\b", r"\bGOT-?OCR[\w.]*\b", r"\bNougat\b", r"\bDonut\b", r"\bPix2tex\b",
    r"\bMistral[\w.\-]*\b", r"\bPhi-?[0-9][\w.\-]*\b",
]
DATASET_PATTERNS = [
    r"\bSWE-?bench(?:[\s\-](?:Verified|Lite|Full))?\b", r"\bHumanEval\b", r"\bMBPP\b", r"\bMATH-?500\b", r"\bGSM8K\b",
    r"\bMMLU\b", r"\bGLUE\b", r"\bMNLI\b", r"\bQNLI\b", r"\bSST-?2\b", r"\bCNN/?Daily ?Mail\b", r"\bCNNDM\b",
    r"\bFALCON\b", r"\bUniMERNet-1M\b", r"\bUniMER(?:Net)? Benchmark\b",r"\barXiv-4M\b", r"\bScanNet\b", r"\bCO3D\b",
    r"\bRealEstate10K\b", r"\bCO3Dv2\b", r"\bIMC\b", r"\bTAP-?Vid\b", r"\bETH3D\b", r"\bDTU\b", r"\bKITTI\b",
    r"\bImageNet(?:-1k)?\b", r"\bCOCO\b", r"\bMegaDepth\b", r"\bBLEU\b(?=\s+on)",
]
SOTA_CUE = re.compile(r"(outperform|surpass|improv|faster|speed-?up|reduc|saving|state-of-the-art|SOTA|beat|compar)", re.I)
VALUE_RE = re.compile(r"(\+?\d+(?:\.\d+)?\s?(?:%|×|x|X)(?:\s+faster)?)")
METRIC_HINTS = [
    ("Pass@1", r"pass@1|resolve rate|resolved rate"), ("Accuracy", r"accuracy"), ("BLEU", r"\bBLEU\b"),
    ("F1 Score", r"\bF1\b"), ("Speedup Factor", r"faster|speed-?up|latency"), ("Memory Reduction", r"memory"),
]


def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        k = re.sub(r"[\s\-]", "", x).lower()
        if k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out


def extract_entities(text: str, patterns: list[str]) -> list[str]:
    hits = []
    for pat in patterns:
        hits += [m.group(0).strip() for m in re.finditer(pat, text)]
    counts = Counter(hits)
    return _dedupe([h for h, _ in counts.most_common()])


def extract_sota_claims(abstract: str | None, results_text: str) -> list[dict]:
    claims = []
    source = (abstract or "")
    for sent in re.split(r"(?<=[.!?])\s+", source):
        if not SOTA_CUE.search(sent):
            continue
        for v in VALUE_RE.findall(sent):
            metric = next((n for n, pat in METRIC_HINTS if re.search(pat, sent, re.I)), "Score")
            target = re.search(r"(?:over|than|against|versus|vs\.?|outperforms?|beats?)\s+([A-Z][\w\-.]+(?:\s[A-Z][\w\-.]+)?)", sent)
            bench = next(iter(extract_entities(sent, DATASET_PATTERNS)), None) or "Overall"
            claims.append({"benchmark": bench, "metric": metric, "value": re.sub(r"(\d)\s+(%|×|x|X)", r"\1\2", v).strip(),
                           "comparison_target": target.group(1) if target else None})
    return claims


TASK_MAP = {"document-understanding": r"document|OCR|formula|table recognition", "computer-vision": r"image|visual|3D|vision|pose|depth",
            "natural-language-processing": r"language model|LLM|text|summariz", "code-generation": r"software engineering|SWE-bench|issue resolution|program repair|code generation",
            "agents": r"\bagent", "model-compression": r"quantiz|distill|1\.58|ternary|low-bit"}
METHOD_MAP = {"deep-learning": r"neural|transformer|deep", "distillation": r"distill", "quantization": r"quantiz|1\.58|ternary",
              "ocr": r"\bOCR\b|formula recognition", "test-time-scaling": r"test-time|ensemble", "multi-agent": r"multi-agent|agent"}


def tag(text: str, mapping: dict[str, str]) -> list[str]:
    return [k for k, pat in mapping.items() if re.search(pat, text, re.I)]


# --------------------------------------------------------------------------
# Optional LLM refinement (needs ANTHROPIC_API_KEY; not exercised in tests)
# --------------------------------------------------------------------------
def llm_refine(record: dict, text: str) -> dict:
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (
            "From this paper text, return ONLY JSON with keys models (list), datasets (list), "
            "sota_claims (list of {benchmark, metric, value, comparison_target}). Include only entities "
            "used in the experiments or main claims, not merely cited.\n\n" + text[:60000])
        msg = client.messages.create(model=os.getenv("EXTRACT_LLM_MODEL", "claude-sonnet-5-5"),
                                     max_tokens=3000, messages=[{"role": "user", "content": prompt}])
        data = json.loads(re.search(r"\{.*\}", msg.content[0].text, re.S).group(0))
        for k in ("models", "datasets", "sota_claims"):
            if data.get(k):
                record[k] = data[k]
    except Exception as e:
        log.warning("%s: LLM refinement skipped (%s)", record["arxiv_id"], e)
    return record


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def safe(fn, default, *a):
    try:
        return fn(*a)
    except Exception as e:
        log.warning("%s failed: %s", fn.__name__, e)
        return default


def process(pdf_path: Path, arxiv_id: str, use_llm: bool) -> dict:
    rec = {"arxiv_id": arxiv_id, "title": None, "authors": [], "publication_date": None,
           "arxiv_pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf", "arxiv_page_url": f"https://arxiv.org/abs/{arxiv_id}",
           "github_url": None, "is_official_code": False, "tasks": [], "methods": [], "models": [], "datasets": [],
           "sota_claims": [], "abstract": None}
    try:
        p = Paper(pdf_path, arxiv_id)
    except Exception as e:
        log.error("%s: cannot open %s (%s)", arxiv_id, pdf_path, e)
        return rec
    body = p.full_text[: find_references_start(p.full_text)]
    rec["title"] = safe(extract_title, None, p)
    rec["authors"] = safe(extract_authors, [], p, rec["title"])
    rec["abstract"] = safe(extract_abstract, None, p)
    rec["publication_date"] = safe(extract_date, None, p)
    rec["github_url"], rec["is_official_code"] = safe(extract_github, (None, False), p, rec["title"], rec["authors"])
    rec["models"] = safe(extract_entities, [], body, MODEL_PATTERNS)
    rec["datasets"] = safe(extract_entities, [], body, DATASET_PATTERNS)
    rec["sota_claims"] = safe(extract_sota_claims, [], rec["abstract"], body)
    ctx = (rec["abstract"] or "") + " " + (rec["title"] or "")
    rec["tasks"], rec["methods"] = tag(ctx, TASK_MAP), tag(ctx, METHOD_MAP)
    return rec if not use_llm else llm_refine(rec, body)


def download(arxiv_id: str, dest: Path) -> None:
    req = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}.pdf", headers={"User-Agent": "frontieratlas-extractor/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", default="pdfs")
    ap.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    ap.add_argument("--out", default="extracted_papers.json")
    ap.add_argument("--download", action="store_true", help="fetch missing PDFs from arxiv.org")
    ap.add_argument("--llm", action="store_true", help="refine models/datasets/sota_claims with an LLM")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for aid in args.ids:
        pdf = pdf_dir / f"{aid}.pdf"
        if not pdf.exists() and args.download:
            try:
                download(aid, pdf)
            except Exception as e:
                log.error("%s: download failed (%s)", aid, e)
        if not pdf.exists():
            log.error("%s: PDF not found at %s", aid, pdf)
            continue
        results.append(process(pdf, aid, args.llm))
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %d records to %s", len(results), args.out)


if __name__ == "__main__":
    main()
