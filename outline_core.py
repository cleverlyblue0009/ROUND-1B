# outline_core.py
import re
import fitz
from dataclasses import dataclass
from typing import List, Tuple, Optional

# --- thresholds (match your tuned Round 1A) ---
MAX_WORDS_HEADING = 8
UPPERCASE_RATIO_MIN = 0.6
MIN_HEADING_SCORE = 2.5
GAP_THRESH = 8

BULLET_START_TOKENS = ("•", "-", "–", "*", "·", "o", "●", "◦")
PUNCT = set(",.;:!?()[]{}'\"")

@dataclass
class LineBlock:
    text: str
    page: int
    bbox: Tuple[float, float, float, float]
    font_size: float
    is_bold: bool
    caps_ratio: float
    font_rank: int
    tag: str = "BODY"   # BODY | HEADING | TITLE

def _clean_space(t:str)->str:
    import re
    return re.sub(r"\s+", " ", t).strip()

def _caps_ratio(t:str)->float:
    letters=[c for c in t if c.isalpha()]
    if not letters: return 0.0
    return sum(c.isupper() for c in letters)/len(letters)

def _is_bullet(t:str)->bool:
    t=t.lstrip()
    if t.startswith(BULLET_START_TOKENS): return True
    return bool(re.match(r"^\d+[\.\)]\s+", t))

def _extract_page_lines(page):
    out=[]
    d=page.get_text("dict")
    for blk in d.get("blocks",[]):
        if blk.get("type",0)!=0: continue
        for ln in blk.get("lines",[]):
            spans=ln.get("spans",[])
            if not spans: continue
            parts=[]
            max_sz=0.0
            is_bold=False
            x0=y0=x1=y1=None
            for s in spans:
                txt=s.get("text","")
                if txt: parts.append(txt)
                sz=float(s.get("size",0))
                if sz>max_sz: max_sz=sz
                fn=s.get("font","").lower()
                if "bold" in fn or "black" in fn or "semibold" in fn: is_bold=True
                sx0,sy0,sx1,sy1=s.get("bbox",(0,0,0,0))
                x0=sx0 if x0 is None else min(x0,sx0)
                y0=sy0 if y0 is None else min(y0,sy0)
                x1=sx1 if x1 is None else max(x1,sx1)
                y1=sy1 if y1 is None else max(y1,sy1)
            text=_clean_space(" ".join(parts))
            if not text: continue
            out.append((text,(x0,y0,x1,y1),is_bold,max_sz))
    return out

def _score_line(text, caps, font_rank, is_bold, gap_above):
    # base
    s=0.0
    if _is_bullet(text): return -5.0
    if text and text[0].islower(): s-=2.0
    if text.lower().startswith(("in ","on ","at ","for ","of ","to ","by ")): s-=1.5
    if font_rank==0: s+=1.0
    elif font_rank==1: s+=0.75
    elif font_rank==2: s+=0.5
    if is_bold: s+=0.25
    w=len(text.split())
    if w<=MAX_WORDS_HEADING: s+=1.5
    if w<=3: s+=0.5
    if caps>=UPPERCASE_RATIO_MIN: s+=1.0
    elif caps<0.2: s-=0.5
    punct=sum(ch in PUNCT for ch in text)
    if punct>=3: s-=0.5
    if gap_above>GAP_THRESH: s+=0.5
    return s

def extract_outline_blocks(pdf_path:str):
    doc=fitz.open(pdf_path)
    # collect raw
    raw=[]
    for pidx in range(doc.page_count):
        page=doc.load_page(pidx)
        for text,bbox,is_bold,max_sz in _extract_page_lines(page):
            raw.append({
                "text":text,
                "page":pidx+1,
                "bbox":bbox,
                "is_bold":is_bold,
                "font_size":max_sz
            })
    if not raw:
        return None, []
    # rank font sizes
    uniq=sorted({round(r["font_size"],2) for r in raw}, reverse=True)
    rankmap={sz:i for i,sz in enumerate(uniq)}
    # build block objects
    blocks=[]
    for r in raw:
        text=r["text"]
        caps=_caps_ratio(text)
        blocks.append(LineBlock(
            text=text,
            page=r["page"],
            bbox=r["bbox"],
            font_size=r["font_size"],
            is_bold=r["is_bold"],
            caps_ratio=caps,
            font_rank=rankmap[round(r["font_size"],2)],
        ))
    # compute gaps & score
    title=None
    for i,b in enumerate(blocks):
        prev=blocks[i-1] if i>0 else None
        if prev is None or prev.page!=b.page:
            gap_above=9999
        else:
            gap_above=b.bbox[1]-prev.bbox[3]
        s=_score_line(b.text,b.caps_ratio,b.font_rank,b.is_bold,gap_above)
        if b.page==1 and b.caps_ratio>0.7 and (title is None or b.font_size>title.font_size):
            title=b
        if s>=MIN_HEADING_SCORE: b.tag="HEADING"
    if title: title.tag="TITLE"
    return title, blocks
