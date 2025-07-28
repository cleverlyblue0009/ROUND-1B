#!/usr/bin/env python3
"""
Round 1B – Persona-Driven Document Intelligence (Hackathon Implementation)

Input (JSON) schema (example):
{
  "challenge_info": {...},              # ignored in output (kept for logging)
  "documents": [
    {"filename": "Doc1.pdf", "title": "Doc1"},
    {"filename": "Doc2.pdf", "title": "Doc2"}
  ],
  "persona": { "role": "Travel Planner" },
  "job_to_be_done": { "task": "Plan a 4-day trip..." }
}

Output schema (output.json):
{
  "metadata": {
    "input_documents": [...],
    "persona": "...",
    "job_to_be_done": "...",
    "processing_timestamp": "ISO8601"
  },
  "extracted_sections": [
    {
      "document": "...pdf",
      "section_title": "...",
      "importance_rank": 1,
      "page_number": 3
    },
    ...
  ],
  "subsection_analysis": [
    {
      "document": "...pdf",
      "refined_text": "...snippet...",
      "page_number": 3
    },
    ...
  ]
}

Requires:
  outline_core.py  (exports: extract_outline_blocks(pdf_path) -> (title_block, blocks))
  scoring.py       (exports: build_query, build_keywords, combined_scores, keyword_set)
"""

import os
import sys
import json
import re
import time
from datetime import datetime
from typing import Any, Dict, List

# local modules
try:
    from outline_core import extract_outline_blocks, LineBlock
except ImportError as e:
    print("ERROR: Cannot import outline_core.py. Make sure it is in the same directory.", file=sys.stderr)
    raise

try:
    import scoring
except ImportError as e:
    print("ERROR: Cannot import scoring.py. Make sure it is in the same directory.", file=sys.stderr)
    raise


# --------------------------------------------------------------------------------------
# Read & normalize hackathon input.json
# --------------------------------------------------------------------------------------
def read_task_hackathon(task_path: str) -> Dict[str, Any]:
    """
    Load hackathon-style input JSON and normalize to flat fields:
      persona: str
      job: str
      documents: List[str] (filenames only)
      challenge_info: dict (optional)
    """
    with open(task_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    persona = raw.get("persona", {}).get("role") or "(unknown persona)"
    job = raw.get("job_to_be_done", {}).get("task") or "(no task specified)"

    # documents list -> filenames
    docs_raw = raw.get("documents", [])
    documents = []
    for d in docs_raw:
        if isinstance(d, dict):
            fn = d.get("filename")
            if fn:
                documents.append(fn)
        elif isinstance(d, str):
            documents.append(d)

    return {
        "persona": persona,
        "job": job,
        "documents": documents,
        "challenge_info": raw.get("challenge_info", {})
    }


# --------------------------------------------------------------------------------------
# Collect PDF absolute paths (validate existence)
# --------------------------------------------------------------------------------------
def resolve_pdf_paths(input_dir: str, filenames: List[str]) -> List[str]:
    """
    Convert filenames from task JSON into absolute paths in input_dir.
    Silently skip missing files but warn on stderr.
    """
    pdfs = []
    for name in filenames:
        p = name if os.path.isabs(name) else os.path.join(input_dir, name)
        if os.path.isfile(p) and p.lower().endswith(".pdf"):
            pdfs.append(p)
        else:
            print(f"Warning: PDF '{name}' not found in {input_dir}. Skipped.", file=sys.stderr)
    return pdfs


def discover_pdfs(input_dir: str) -> List[str]:
    """Fallback: all PDFs in input_dir."""
    return [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf")
    ]


# --------------------------------------------------------------------------------------
# Build sections: heading text + concatenated body until next heading
# --------------------------------------------------------------------------------------
def build_sections_from_blocks(title_block, blocks: List[LineBlock], doc_name: str) -> List[Dict[str, Any]]:
    """
    Converts LineBlocks into logical sections for scoring.
    Each section:
      doc, heading, page, text
    """
    sections = []
    current_heading = None
    current_text: List[str] = []
    current_pages = set()

    def flush():
        nonlocal current_heading, current_text, current_pages
        if current_heading:
            sections.append({
                "doc": doc_name,
                "heading": current_heading.text,
                "page": min(current_pages) if current_pages else current_heading.page,
                "text": " ".join(current_text).strip()
            })
        current_heading = None
        current_text = []
        current_pages = set()

    for b in blocks:
        if b.tag == "TITLE":
            continue
        if b.tag == "HEADING":
            flush()
            current_heading = b
            current_text = []
            current_pages = {b.page}
        else:  # BODY
            if current_heading is not None:
                current_text.append(b.text)
                current_pages.add(b.page)
            else:
                # body text before first heading -> ignore
                pass

    flush()

    # fallback: no headings found -> whole doc
    if not sections:
        text_all = " ".join([b.text for b in blocks if b.tag != "TITLE"]).strip()
        sections.append({
            "doc": doc_name,
            "heading": doc_name,
            "page": 1,
            "text": text_all
        })

    return sections


# --------------------------------------------------------------------------------------
# Rank sections by persona+job relevance
# --------------------------------------------------------------------------------------
def rank_sections(persona: str, job: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    query = scoring.build_query(persona, job)
    kw = scoring.build_keywords(persona, job)
    headings = [s["heading"] for s in sections]
    texts = [s["text"] for s in sections]
    scores = scoring.combined_scores(query, headings, texts, kw)
    for s, sc in zip(sections, scores):
        s["score"] = float(sc)
    sections.sort(key=lambda x: x["score"], reverse=True)
    for i, s in enumerate(sections, start=1):
        s["importance_rank"] = i
    return sections


# --------------------------------------------------------------------------------------
# Sub-section drilldown (snippets)
# --------------------------------------------------------------------------------------
SNIP_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+|[\r\n]+|•")

def extract_subsections(section: Dict[str, Any],
                        persona: str,
                        job: str,
                        max_snips: int = 3) -> List[Dict[str, Any]]:
    """
    Break section text into candidate snippets and score them quickly
    against the persona+job query. Returns top snippets.
    """
    text = section["text"]
    if not text:
        return []

    # candidate splits
    parts = []
    for seg in SNIP_SPLIT_RE.split(text):
        seg = seg.strip()
        if not seg:
            continue
        # de-bullet
        seg = re.sub(r"^[•\-\*\u2022●◦]\s*", "", seg)
        parts.append(seg)

    if not parts:
        return []

    # score snippets using TF-IDF vs query
    query = scoring.build_query(persona, job)
    scores = scoring.tfidf_scores(query, parts)
    order = scores.argsort()[::-1]  # high->low

    out = []
    base_page = section["page"]  # cheap fallback; we don't track fine-grained snippet pages
    for idx in order[:max_snips]:
        out.append({
            "document": section["doc"],
            "refined_text": parts[idx],
            "page_number": base_page
        })
    return out


# --------------------------------------------------------------------------------------
# Main Round 1B processing
# --------------------------------------------------------------------------------------
def process_round1b(input_dir: str,
                    output_dir: str,
                    task_file: str = "input.json",
                    topk_sections: int = 20,
                    max_snips_per_section: int = 3) -> None:
    start = time.time()

    # --- load task JSON ---
    task_path = os.path.join(input_dir, task_file)
    if not os.path.isfile(task_path):
        print(f"ERROR: Task file '{task_file}' not found in {input_dir}.", file=sys.stderr)
        sys.exit(1)

    task_data = read_task_hackathon(task_path)
    persona = task_data["persona"]
    job = task_data["job"]

    # --- resolve PDFs ---
    if task_data["documents"]:
        pdfs = resolve_pdf_paths(input_dir, task_data["documents"])
    else:
        pdfs = discover_pdfs(input_dir)

    if not pdfs:
        print("ERROR: No PDFs to process.", file=sys.stderr)
        sys.exit(1)

    # --- per-PDF section extraction ---
    all_sections: List[Dict[str, Any]] = []
    for pdf_path in pdfs:
        doc_name = os.path.basename(pdf_path)
        try:
            title, blocks = extract_outline_blocks(pdf_path)
        except Exception as e:
            print(f"ERROR: Failed to parse '{doc_name}': {e}", file=sys.stderr)
            continue
        secs = build_sections_from_blocks(title, blocks, doc_name)
        all_sections.extend(secs)

    if not all_sections:
        print("ERROR: No sections extracted from any PDF.", file=sys.stderr)
        sys.exit(1)

    # --- rank sections ---
    ranked = rank_sections(persona, job, all_sections)

    # --- select top K ---
    top_sections = ranked[: min(len(ranked), topk_sections)]

    # --- sub-section analysis ---
    all_sub = []
    for sec in top_sections:
        subs = extract_subsections(sec, persona, job, max_snips=max_snips_per_section)
        all_sub.extend(subs)

    # --- output JSON (hackathon expected format) ---
    out = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in pdfs],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": [
            {
                "document": s["doc"],
                "section_title": s["heading"],
                "importance_rank": s["importance_rank"],
                "page_number": s["page"]
            }
            for s in top_sections
        ],
        "subsection_analysis": all_sub
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"Round 1B: processed {len(pdfs)} PDFs in {elapsed:.2f}s -> {out_path}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Round 1B Persona-Driven Document Intelligence")
    ap.add_argument("--input", default=None, help="Input directory (default: env INPUT_DIR or ./input)")
    ap.add_argument("--output", default=None, help="Output directory (default: env OUTPUT_DIR or ./output)")
    ap.add_argument("--task", default="input.json", help="Input JSON filename in input dir")
    ap.add_argument("--topk", type=int, default=20, help="Max sections to return")
    ap.add_argument("--snips", type=int, default=3, help="Max snippets per section")
    args = ap.parse_args()

    input_dir = args.input or os.environ.get("INPUT_DIR") or os.path.join(os.getcwd(), "input")
    output_dir = args.output or os.environ.get("OUTPUT_DIR") or os.path.join(os.getcwd(), "output")

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory '{input_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    process_round1b(
        input_dir=input_dir,
        output_dir=output_dir,
        task_file=args.task,
        topk_sections=args.topk,
        max_snips_per_section=args.snips,
    )


if __name__ == "__main__":
    main()
