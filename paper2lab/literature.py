"""「搜」外部文献检索 —— 多源联邦检索（6 个公开学术索引，全部免费、无需 key）。

**为什么要这个模块**：`retrieval.py` 原来那版「外部检索」是让模型凭记忆写几篇文献，
本质是**让模型背课文** —— 它写出来的东西无法证实存在，属于最该防的幻觉来源。
本模块换掉这件事：只去**真数据库**查，返回的每一条都带 DOI / 落地页，
点开就能核；查不到就如实说查不到。

**为什么是「多源」而不是「接一个库」**：单接 arXiv 就查不到期刊和会议，
单接 Crossref 就查不到预印本。这里并发查 6 个库再合并去重，
覆盖范围是**累加的**，不押在某一家身上：

    openalex     全学科、全出版社（含 arXiv / PubMed 的收录）  ~2.6 亿
    crossref     出版社登记的 DOI 元数据（期刊、会议、图书）   ~1.6 亿
    arxiv        物理 / 计算机 / 数学预印本                      ~260 万
    europepmc    生命科学与医学（含 PubMed 全量与预印本）        ~4500 万
    datacite     数据集 / 软件 / 样本的 DOI（科学数据这一块）    ~5000 万
    doaj         开放获取期刊（能直接读到全文）                  ~2000 万

去重后总量 4 亿条量级。**边界要讲清楚**：这是"跨公开学术索引"，
不是"整个互联网"——新闻、博客、厂商文档不在范围内（那类内容需要商业搜索
API，要钱且要 key，与本项目"零成本可跑"的定位冲突）。

**三条硬边界**（与 `scifuncs` 的红线同源）：

1. **不编文献**。每条 `LitHit` 都来自某次真实 HTTP 响应；没给 DOI 就留空，
   **绝不为了好看拼一个 DOI 出来**。
2. **不冒充核验**。文献真实存在 ≠ 它支持这条结论。所有返回项一律
   `verified=False`，报告里必须写"候选依据，需人读摘要确认"。
3. **不硬凑**。相关度不达标的结果直接丢（见 `MIN_TITLE_HIT`），
   宁可返回空列表，也不拿"碰巧有 hello world 论文"充数。
"""
from __future__ import annotations

import hashlib
import html as _html
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

__all__ = [
    "PROVIDERS", "PROVIDER_DESC", "LitHit", "SearchResult",
    "search", "search_many", "for_findings", "enabled", "clear_cache",
    "CACHE_DIR", "MODE",
]

#: 检索方式的自述 —— 写进结果里，避免对外被含糊成"语义检索"或"全网搜索"。
MODE = ("多源联邦检索：6 个公开学术索引并发查询 → DOI/标题去重 → "
        "标题与摘要命中率重排（关键词检索，非向量语义检索）")

#: 默认开启。全免费、无 key，所以没理由默认关；但保留开关，
#: 内网 / 离线环境可以 `PAPER2LAB_LIT=off` 一键关掉（关掉后如实标"未检索"）。
_ENV_ON = ("1", "on", "true", "yes", "auto")

#: 单个源的 socket 超时与整体 wall-clock 预算（秒）。
#: 预算比"源数 × 单源超时"小得多 —— 一个源卡住不该拖死整条审计链。
DEFAULT_TIMEOUT = float(os.environ.get("PAPER2LAB_LIT_TIMEOUT") or 4.5)
DEFAULT_BUDGET = float(os.environ.get("PAPER2LAB_LIT_BUDGET") or 7.0)

#: 相关度下限：标题至少命中 40% 的查询词，否则视为"没检索到"。
#: 0.34 实测会把「只因为摘要里出现一次 learning rate」的论文放进来 —— 那属于噪音。
MIN_TITLE_HIT = 0.40

#: 缓存目录与有效期。审计同一篇论文时不必重复打人家的 API（也省人家的额度）。
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "lit"
CACHE_TTL = int(os.environ.get("PAPER2LAB_LIT_TTL") or 7 * 86400)

#: OpenAlex 无 key 时是"共享 IP 的每日免费额度"，可能被别人用完；
#: 给了 key 就是自己的 $1/天。**缺 key 不算配置错误**，降级到其余 5 个源即可。
MAILTO = os.environ.get("PAPER2LAB_LIT_MAILTO") or "paper2lab@example.com"
UA = ("Paper2Lab/1.3 (academic consistency auditor; +https://github.com/"
      "la1034681-blip/paper2lab) Python-urllib")

#: 源清单。顺序 = 对外展示顺序（覆盖广的排前面）。
PROVIDERS: tuple = ("openalex", "crossref", "europepmc", "arxiv", "datacite", "doaj")

PROVIDER_DESC: dict = {
    "openalex": "全学科全出版社（含预印本与 PubMed 收录，~2.6 亿条）",
    "crossref": "出版社登记的 DOI 元数据（期刊/会议/图书，~1.6 亿条）",
    "europepmc": "生命科学与医学（含 PubMed 全量，~4500 万条）",
    "arxiv": "物理/计算机/数学预印本（~260 万条）",
    "datacite": "数据集/软件/样本的 DOI（科学数据，~5000 万条）",
    "doaj": "开放获取期刊（多数能直接读全文，~2000 万条）",
}

#: 查询词里的停用词 —— 它们几乎出现在每篇论文标题里，命中它们不代表相关。
_STOP = frozenset("""
a an the of for and or in on at to with by from via using based into over under
is are was were be been being this that these those we our it its as not no
study analysis approach method framework model models results
without towards toward when while how what why using both more than such also may can
""".split())

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")
_TAG = re.compile(r"<[^>]+>")

#: 「论文内部的表格 / 附录」被当成独立条目的假文献。
#: 实测：Crossref 里真有 `Table 4: Proposed model performance tuning…`
#: `Supplemental Information 1: Combinations of number of hidden…` 这类 DOI ——
#: 它们是**正文的零件**，不是文献，混进候选列表只会把好东西挤下去。
_NOISE_TITLE = re.compile(
    r"^\s*(?:"
    r"(?:table|figure|fig|scheme|appendix)\s*(?:\d+|[A-Z]\b|:)"
    r"|(?:supplemental|supplementary)\b"
    r")", re.I)


# ---------------------------------------------------------------- 基础工具

def enabled() -> bool:
    """外部检索是否启用（默认启用）。"""
    return str(os.environ.get("PAPER2LAB_LIT") or "on").strip().lower() in _ENV_ON


def _get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/json, application/atom+xml, text/xml, */*",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _json(url: str, timeout: float) -> Any:
    raw = _get(url, timeout)
    return json.loads(raw.decode("utf-8", "replace"))


def _clean(text: Any, limit: int = 1200) -> str:
    """去标签 + 反转义 + 压空白。摘要里常带 JATS/HTML 标签，直接给人看会乱。"""
    s = _html.unescape(_TAG.sub(" ", str(text or "")))
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def _terms(text: str) -> set:
    """查询词 -> 比对用的词集合。

    英文按词切（去停用词）；中文没有空格，按 **2 字滑窗**切 —— 这是中文检索里
    最朴素也够用的做法，避免引入分词依赖。全角/半角不做归一（学术标题以英文为主）。
    """
    s = str(text or "")
    out: set = set()
    for w in _WORD.findall(s):
        w = w.lower().strip("-")
        if len(w) >= 2 and w not in _STOP:
            out.add(w)
    for run in _CJK.findall(s):
        if len(run) <= 2:
            out.add(run)
        else:
            out.update(run[i:i + 2] for i in range(len(run) - 1))
    return out


def _hit_ratio(terms: set, text: str) -> float:
    """查询词在 text 里的命中比例（0~1）。terms 为空时返回 0，不返回 1。

    英文必须按**词边界**比，不能用子串：`"rate" in "generate"` 为真，
    那样「generate」就会被算成命中了「learning rate」—— 相关度虚高之后，
    门槛就形同虚设了（实测这是排序把不相关论文排上来的主因）。
    中文没有词边界，按 2 字串直接比。
    """
    if not terms:
        return 0.0
    low = str(text or "").lower()
    if not low:
        return 0.0
    hit = 0
    for t in terms:
        if _CJK.search(t):
            if t in low:
                hit += 1
        elif re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low):
            hit += 1
    return hit / len(terms)


def _norm_doi(doi: Any) -> str:
    """把各家五花八门的 DOI 写法收敛成 `10.xxxx/yyyy`。

    实测踩到的脏数据：OpenAlex 有时把**全文链接**当 DOI 给
    （`10.1051/itmconf/20268403033/pdf`）。DOI 本身不含这些后缀，
    不剥掉就会同一条记录在去重时被当成两条，而且点开是 404。
    """
    d = str(doi or "").strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    d = re.sub(r"^doi:\s*", "", d)
    d = re.sub(r"/(?:pdf|full|abstract|html|epub|fulltext|metadata)$", "", d)
    d = re.sub(r"\.pdf$", "", d)
    return d.rstrip("/. ")


def _norm_title(title: Any) -> str:
    """去重用的标题指纹：只留字母数字，抹平标点差异。"""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(title or "").lower())


def _int_or_none(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 六个源
#
# 每个 `_fetch_*` 只做一件事：把这家 API 的返回**翻译成同一种字典**。
# 解析失败就地抛异常，由 `search()` 统一记成"这个源这次没成"，不影响别的源。

def _fetch_openalex(query: str, n: int, timeout: float) -> list:
    url = (f"https://api.openalex.org/works?search={urllib.parse.quote(query)}"
           f"&per-page={max(1, min(n, 50))}&mailto={urllib.parse.quote(MAILTO)}")
    key = os.environ.get("OPENALEX_API_KEY") or os.environ.get("PAPER2LAB_OPENALEX_KEY")
    if key:
        url += f"&api_key={urllib.parse.quote(key)}"
    data = _json(url, timeout)
    out = []
    for w in (data.get("results") or []):
        if not isinstance(w, dict):
            continue
        loc = w.get("primary_location") or {}
        src = (loc.get("source") or {})
        out.append({
            "title": w.get("display_name") or w.get("title") or "",
            "authors": [((a.get("author") or {}).get("display_name") or "")
                        for a in (w.get("authorships") or [])][:8],
            "year": _int_or_none(w.get("publication_year")),
            "venue": src.get("display_name") or "",
            "doi": _norm_doi(w.get("doi")),
            "url": loc.get("landing_page_url") or w.get("id") or "",
            "oa_url": (w.get("open_access") or {}).get("oa_url") or "",
            "citations": _int_or_none(w.get("cited_by_count")),
            "kind": w.get("type") or "",
            "abstract": _openalex_abstract(w.get("abstract_inverted_index")),
            "source": "openalex",
        })
    return out


def _openalex_abstract(inv: Any) -> str:
    """OpenAlex 的摘要是**倒排索引**（词 -> 位置列表），要按位置重排还原成句子。"""
    if not isinstance(inv, dict):
        return ""
    pairs = []
    for word, positions in inv.items():
        for p in (positions or []):
            try:
                pairs.append((int(p), word))
            except (TypeError, ValueError):
                continue
    if not pairs:
        return ""
    pairs.sort()
    return _clean(" ".join(w for _, w in pairs), 1200)


def _fetch_crossref(query: str, n: int, timeout: float) -> list:
    fields = ("DOI,title,author,issued,container-title,is-referenced-by-count,"
              "type,abstract,URL")
    url = (f"https://api.crossref.org/works?query.bibliographic={urllib.parse.quote(query)}"
           f"&rows={max(1, min(n, 50))}&select={fields}"
           f"&mailto={urllib.parse.quote(MAILTO)}")
    data = _json(url, timeout)
    out = []
    for it in ((data.get("message") or {}).get("items") or []):
        if not isinstance(it, dict):
            continue
        titles = it.get("title") or []
        year = None
        for key in ("issued", "published-print", "published-online", "created"):
            parts = (it.get(key) or {}).get("date-parts") or []
            if parts and parts[0]:
                year = _int_or_none(parts[0][0])
                break
        authors = []
        for a in (it.get("author") or [])[:8]:
            name = " ".join(x for x in (a.get("given"), a.get("family")) if x)
            if name:
                authors.append(name)
        out.append({
            "title": (titles[0] if titles else "") or "",
            "authors": authors,
            "year": year,
            "venue": ((it.get("container-title") or [""])[0] or ""),
            "doi": _norm_doi(it.get("DOI")),
            "url": it.get("URL") or (f"https://doi.org/{_norm_doi(it.get('DOI'))}"
                                     if it.get("DOI") else ""),
            "oa_url": "",
            "citations": _int_or_none(it.get("is-referenced-by-count")),
            "kind": it.get("type") or "",
            "abstract": _clean(it.get("abstract"), 1200),
            "source": "crossref",
        })
    return out


def _fetch_europepmc(query: str, n: int, timeout: float) -> list:
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search"
           f"?query={urllib.parse.quote(query)}&format=json"
           f"&pageSize={max(1, min(n, 50))}&resultType=core")
    data = _json(url, timeout)
    out = []
    for it in (((data.get("resultList") or {}).get("result")) or []):
        if not isinstance(it, dict):
            continue
        doi = _norm_doi(it.get("doi"))
        src, pid = it.get("source") or "", it.get("id") or ""
        out.append({
            "title": it.get("title") or "",
            "authors": [a.strip() for a in str(it.get("authorString") or "").split(",")
                        if a.strip()][:8],
            "year": _int_or_none(it.get("pubYear")),
            "venue": (it.get("journalTitle")
                      or ((it.get("bookOrReportDetails") or {}).get("publisher")
                          if isinstance(it.get("bookOrReportDetails"), dict) else "")
                      or ""),
            "doi": doi,
            "url": (f"https://doi.org/{doi}" if doi else
                    (f"https://europepmc.org/article/{src}/{pid}" if src and pid else "")),
            "oa_url": "",
            "citations": _int_or_none(it.get("citedByCount")),
            "kind": it.get("pubType") or "",
            "abstract": _clean(it.get("abstractText"), 1200),
            "source": "europepmc",
        })
    return out


def _fetch_arxiv(query: str, n: int, timeout: float) -> list:
    url = ("http://export.arxiv.org/api/query"
           f"?search_query=all:{urllib.parse.quote(query)}"
           f"&start=0&max_results={max(1, min(n, 50))}&sortBy=relevance")
    raw = _get(url, timeout)
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(raw)
    out = []
    for e in root.findall("a:entry", ns):
        title = (e.findtext("a:title", "", ns) or "").strip()
        if not title:
            continue
        link = (e.findtext("a:id", "", ns) or "").strip()
        doi = _norm_doi(e.findtext("arxiv:doi", "", ns))
        out.append({
            "title": re.sub(r"\s+", " ", title),
            "authors": [(a.findtext("a:name", "", ns) or "").strip()
                        for a in e.findall("a:author", ns)][:8],
            "year": _int_or_none((e.findtext("a:published", "", ns) or "")[:4]),
            "venue": "arXiv",
            "doi": doi,
            "url": link,
            "oa_url": link,
            "citations": None,          # arXiv 不给被引数，**不猜**
            "kind": "preprint",
            "abstract": _clean(e.findtext("a:summary", "", ns), 1200),
            "source": "arxiv",
        })
    return out


def _fetch_datacite(query: str, n: int, timeout: float) -> list:
    url = (f"https://api.datacite.org/dois?query={urllib.parse.quote(query)}"
           f"&page[size]={max(1, min(n, 50))}")
    data = _json(url, timeout)
    out = []
    for it in (data.get("data") or []):
        a = (it or {}).get("attributes") or {}
        titles = a.get("titles") or []
        title = (titles[0] or {}).get("title") if titles else ""
        if not title:
            continue
        doi = _norm_doi(a.get("doi") or it.get("id"))
        out.append({
            "title": title,
            "authors": [c.get("name") or "" for c in (a.get("creators") or [])][:8],
            "year": _int_or_none(a.get("publicationYear")),
            "venue": a.get("publisher") or "",
            "doi": doi,
            "url": a.get("url") or (f"https://doi.org/{doi}" if doi else ""),
            "oa_url": "",
            "citations": _int_or_none(a.get("citationCount")),
            "kind": (a.get("types") or {}).get("resourceTypeGeneral") or "",
            "abstract": _clean((a.get("descriptions") or [{}])[0].get("description")
                               if a.get("descriptions") else "", 1200),
            "source": "datacite",
        })
    return out


def _fetch_doaj(query: str, n: int, timeout: float) -> list:
    url = (f"https://doaj.org/api/search/articles/{urllib.parse.quote(query)}"
           f"?pageSize={max(1, min(n, 50))}")
    data = _json(url, timeout)
    out = []
    for it in (data.get("results") or []):
        b = (it or {}).get("bibjson") or {}
        title = b.get("title") or ""
        if not title:
            continue
        ids = {str(i.get("type") or "").lower(): str(i.get("id") or "")
               for i in (b.get("identifier") or []) if isinstance(i, dict)}
        links = [str(l.get("url")) for l in (b.get("link") or [])
                 if isinstance(l, dict) and l.get("url")]
        doi = _norm_doi(ids.get("doi"))
        out.append({
            "title": title,
            "authors": [str(a.get("name") or "") for a in (b.get("author") or [])
                        if isinstance(a, dict)][:8],
            "year": _int_or_none(b.get("year")),
            "venue": ((b.get("journal") or {}).get("title") or ""),
            "doi": doi,
            "url": (f"https://doi.org/{doi}" if doi else (links[0] if links else "")),
            "oa_url": links[0] if links else "",
            "citations": None,          # DOAJ 不给被引数，**不猜**
            "kind": "journal-article",
            "abstract": _clean(b.get("abstract"), 1200),
            "source": "doaj",
        })
    return out


_FETCHERS = {
    "openalex": _fetch_openalex,
    "crossref": _fetch_crossref,
    "europepmc": _fetch_europepmc,
    "arxiv": _fetch_arxiv,
    "datacite": _fetch_datacite,
    "doaj": _fetch_doaj,
}


# ---------------------------------------------------------------- 结果模型

@dataclass
class LitHit:
    """一条检索到的真实文献记录。

    `verified=False` 是**故意写死的默认值**：这条记录在数据库里真实存在（可点 DOI），
    但"它是否支持本次审计的某条结论"没有任何机器能替人判断 —— 两者是两件事，
    不能因为前者为真就把后者也标成真（这是本项目一贯的"非自证"口径）。
    """
    title: str = ""
    authors: list = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    oa_url: str = ""
    citations: Optional[int] = None
    kind: str = ""
    abstract: str = ""
    found_in: list = field(default_factory=list)
    score: float = 0.0
    why: str = ""
    verified: bool = False

    def to_dict(self) -> dict:
        d = {
            "title": self.title,
            "authors": [a for a in self.authors if a],
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "url": self.url,
            "oa_url": self.oa_url,
            "citations": self.citations,
            "kind": self.kind,
            "abstract": self.abstract,
            "found_in": list(self.found_in),
            "score": round(float(self.score), 3),
            "why": self.why,
            # 两层含义分开标，别让下游混用：
            #   exists  —— 数据库里真有这条（可核）
            #   verified —— 它是否支持本次结论（**没人为它背书**）
            "exists": True,
            "verified": False,
        }
        return d


@dataclass
class SearchResult:
    query: str = ""
    items: list = field(default_factory=list)
    providers: dict = field(default_factory=dict)
    reached: int = 0
    cached: bool = False
    elapsed_s: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": True,
            "query": self.query,
            "items": [h.to_dict() if isinstance(h, LitHit) else h for h in self.items],
            "providers": self.providers,
            "reached": self.reached,
            "of": len(self.providers),
            "cached": self.cached,
            "elapsed_s": round(self.elapsed_s, 2),
            "mode": MODE,
            "note": self.note,
            "boundary": ("文献记录真实存在（可点 DOI 核），但「是否支持本次结论」"
                         "需人读摘要确认 —— 本结果不代表任何结论已被证实。"),
        }


# ---------------------------------------------------------------- 缓存

def _cache_path(query: str, providers: Iterable[str], limit: int,
                year_from: Optional[int]) -> Path:
    raw = "|".join([query.strip().lower(), ",".join(sorted(providers)),
                    str(limit), str(year_from or "")])
    return CACHE_DIR / (hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20] + ".json")


def _cache_read(path: Path) -> Optional[dict]:
    try:
        if not path.exists() or (time.time() - path.stat().st_mtime) > CACHE_TTL:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("items") is not None else None
    except Exception:      # noqa: BLE001 —— 缓存坏了就当没有，不影响检索
        return None


def _cache_write(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:      # noqa: BLE001
        pass


def clear_cache() -> int:
    """清空检索缓存，返回删掉的条数（测试与排障用）。"""
    n = 0
    try:
        for p in CACHE_DIR.glob("*.json"):
            p.unlink()
            n += 1
    except Exception:      # noqa: BLE001
        pass
    return n


# ---------------------------------------------------------------- 合并与重排

def _merge(raw_hits: list, terms: set, year_from: Optional[int]) -> list:
    """把各源的记录合并成一份。

    **两阶段去重**（只做一阶段会留下重复）：
      ① 按 DOI 合并 —— 有 DOI 就是同一篇的铁证；
      ② 再按标题指纹合并 —— 同一篇论文常出现「arXiv 版没 DOI、会议版有 DOI」，
         只按 DOI 去重会把它算成两条，用户会以为查到两篇不同的文章。
         合并方向一律**无 DOI → 有 DOI**（保住可点的 DOI 链接）。
    """
    def _bucket_add(cur: dict, h: dict) -> None:
        srcs = cur.setdefault("found_in", [])
        if h.get("source") and h["source"] not in srcs:
            srcs.append(h["source"])
        for f in ("abstract", "venue", "oa_url", "doi", "url"):
            if not cur.get(f) and h.get(f):
                cur[f] = h[f]
        if cur.get("citations") is None and h.get("citations") is not None:
            cur["citations"] = h["citations"]
        if not cur.get("authors") and h.get("authors"):
            cur["authors"] = h["authors"]

    buckets: dict = {}
    for h in raw_hits:
        # 标题里的 `&amp;` 之类实体要还原 —— Crossref 的标题是半 HTML 的
        title = re.sub(r"\s+", " ", _html.unescape(str(h.get("title") or ""))).strip()
        if not title or _NOISE_TITLE.match(title):
            continue
        h = dict(h, title=title)
        key = _norm_doi(h.get("doi")) or ("t:" + _norm_title(title))
        if not key or key == "t:":
            continue
        cur = buckets.get(key)
        if cur is None:
            buckets[key] = dict(h)
            buckets[key]["found_in"] = [h.get("source")]
            continue
        _bucket_add(cur, h)

    # 第二阶段：把「无 DOI 的同标题记录」并进「有 DOI 的那条」
    by_title: dict = {}
    for key, h in buckets.items():
        if key.startswith("t:"):
            continue
        by_title.setdefault("t:" + _norm_title(h.get("title")), key)
    for key in [k for k in buckets if k.startswith("t:")]:
        target = by_title.get(key)
        if target is None:
            by_title[key] = key          # 这篇本来就没 DOI，自成一条
            continue
        _bucket_add(buckets[target], buckets.pop(key))

    out: list = []
    for h in buckets.values():
        year = h.get("year")
        if year_from and year and int(year) < int(year_from):
            continue
        title_hit = _hit_ratio(terms, h.get("title"))
        abs_hit = _hit_ratio(terms, h.get("abstract"))
        # 相关度不达标直接丢 —— 宁可返回空，也不拿"碰巧存在的论文"充数
        if title_hit < MIN_TITLE_HIT and abs_hit < 0.5:
            continue
        n_src = len(h.get("found_in") or [])
        cites = h.get("citations")
        score = (3.0 * title_hit + 1.0 * abs_hit
                 + 0.8 * (n_src - 1)
                 + 0.35 * math.log1p(cites or 0)
                 + (0.3 if year and year >= 2020 else 0.0))
        bits = []
        if title_hit:
            bits.append(f"标题命中 {round(title_hit * len(terms))}/{len(terms)} 个查询词")
        if n_src > 1:
            bits.append(f"{n_src} 个源同时收录（{', '.join(h['found_in'])}）")
        if cites is not None:
            bits.append(f"被引 {cites}")
        h["score"] = score
        h["why"] = "；".join(bits) or "摘要命中"
        h["found_in"] = h.get("found_in") or []
        out.append(LitHit(
            title=str(h.get("title") or "").strip(),
            authors=[a for a in (h.get("authors") or []) if a],
            year=h.get("year"), venue=h.get("venue") or "",
            doi=h.get("doi") or "", url=h.get("url") or "",
            oa_url=h.get("oa_url") or "", citations=h.get("citations"),
            kind=h.get("kind") or "", abstract=h.get("abstract") or "",
            found_in=h["found_in"], score=score, why=h["why"],
        ))
    out.sort(key=lambda x: (-x.score, -(x.citations or 0), -(x.year or 0)))
    return out


def search(query: str, *, limit: int = 8, providers: Optional[Iterable[str]] = None,
           year_from: Optional[int] = None, timeout: Optional[float] = None,
           budget: Optional[float] = None, use_cache: bool = True) -> dict:
    """在 6 个公开学术索引里并发查同一个 query，合并去重后按相关度返回。

    返回 `{"ok", "query", "items", "providers", "reached", "cached", "note"}`。
    **一个源挂了不影响别的源** —— 失败原因逐源记在 `providers` 里，不假装成功。
    """
    q = re.sub(r"\s+", " ", str(query or "")).strip()
    if not q:
        return {"ok": False, "query": "", "items": [], "providers": {},
                "error": "query 为空 —— 没有检索词就不发请求（不拿空查询去试探人家的库）",
                "mode": MODE}
    if not enabled():
        return {"ok": True, "query": q, "items": [], "providers": {},
                "reached": 0, "of": len(PROVIDERS), "cached": False,
                "note": "外部文献检索已关闭（PAPER2LAB_LIT=off）；本次只做论文内部检索",
                "mode": MODE}

    want = [p for p in (providers or PROVIDERS) if p in _FETCHERS]
    if not want:
        return {"ok": False, "query": q, "items": [], "providers": {},
                "error": f"providers 里没有可用的源；可用: {', '.join(PROVIDERS)}",
                "mode": MODE}

    limit = max(1, min(int(limit or 8), 30))
    # 每个源多取一些，给去重和重排留余量（去重后常常只剩一半）
    per_source = max(limit, min(limit * 2, 30))
    timeout = float(timeout or DEFAULT_TIMEOUT)
    budget = float(budget or DEFAULT_BUDGET)

    cpath = _cache_path(q, want, limit, year_from)
    if use_cache:
        hit = _cache_read(cpath)
        if hit is not None:
            hit["cached"] = True
            hit["note"] = (hit.get("note") or "") + "（本机缓存命中）"
            return hit

    terms = _terms(q)
    t0 = time.time()
    raw_hits: list = []
    providers: dict = {}
    reached = 0

    # 注意：**不用 with** —— 上下文退出会 join 所有线程，某个源卡住就把预算吃光。
    # 每个源自己带 socket 超时，这里再给一个整体预算兜底。
    pool = ThreadPoolExecutor(max_workers=len(want), thread_name_prefix="p2l-lit")
    try:
        futs = {pool.submit(_FETCHERS[p], q, per_source, timeout): p for p in want}
        try:
            for fut in as_completed(futs, timeout=budget):
                name = futs[fut]
                try:
                    got = fut.result()
                except Exception as ex:      # noqa: BLE001 —— 单个源失败是正常情况
                    providers[name] = {"ok": False, "n": 0,
                                       "error": f"{type(ex).__name__}: {ex}"}
                    continue
                reached += 1
                providers[name] = {"ok": True, "n": len(got)}
                raw_hits.extend(got)
        except Exception:      # noqa: BLE001 —— as_completed 超时
            pass
        # 超时还没回来的源：如实标出来，不当作"0 条结果"
        for fut, name in futs.items():
            if name in providers:
                continue
            # 超时那一瞬间刚好赶回来的，别误记成超时（它其实已经拿到了）
            if fut.done() and not fut.cancelled():
                try:
                    got = fut.result()
                except Exception as ex:      # noqa: BLE001
                    providers[name] = {"ok": False, "n": 0,
                                       "error": f"{type(ex).__name__}: {ex}"}
                else:
                    reached += 1
                    providers[name] = {"ok": True, "n": len(got)}
                    raw_hits.extend(got)
                continue
            providers[name] = {"ok": False, "n": 0, "error": "整体预算内未返回（超时）"}
            fut.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    items = _merge(raw_hits, terms, year_from)[:limit]
    elapsed = time.time() - t0

    if reached == 0:
        note = (f"{len(want)} 个检索源都没连上（{elapsed:.1f}s）—— 如实记为未检索，"
                "本次结论没有外部文献依据")
    elif not items:
        note = (f"已查 {reached}/{len(want)} 个源、拿到 {len(raw_hits)} 条原始记录，"
                "但没有一条相关度达标 —— 按「宁可说没有，也不拿不相关的充数」丢弃")
    else:
        note = (f"已查 {reached}/{len(want)} 个源、拿到 {len(raw_hits)} 条原始记录，"
                f"去重重排后返回 {len(items)} 条；均为候选依据，需人读摘要确认")

    payload = SearchResult(query=q, items=items, providers=providers,
                           reached=reached, cached=False, elapsed_s=elapsed,
                           note=note).to_dict()
    if use_cache:
        _cache_write(cpath, payload)
    return payload


def search_many(queries: Iterable[str], *, limit: int = 8, **kw) -> dict:
    """一次查多个 query（审计场景：一篇论文的 2~3 个参数各查一次）。

    合并后按 `found_in` + 分数排，并对同一 DOI 只留一条。
    """
    qs = []
    for q in queries or []:
        q = re.sub(r"\s+", " ", str(q or "")).strip()
        if q and q not in qs:
            qs.append(q)
    if not qs:
        return {"ok": False, "items": [], "queries": [], "error": "没有可用的检索词",
                "mode": MODE}

    merged: dict = {}
    per_q: dict = {}
    reached_max = 0
    providers: dict = {}
    t0 = time.time()
    for q in qs[:4]:          # 上限 4 个 —— 再多就是在替用户乱猜研究方向
        r = search(q, limit=limit, **kw)
        per_q[q] = len(r.get("items") or [])
        reached_max = max(reached_max, int(r.get("reached") or 0))
        for k, v in (r.get("providers") or {}).items():
            prev = providers.get(k)
            providers[k] = v if prev is None else (
                {"ok": bool(prev.get("ok") or v.get("ok")),
                 "n": max(int(prev.get("n") or 0), int(v.get("n") or 0))})
        for it in (r.get("items") or []):
            key = _norm_doi(it.get("doi")) or ("t:" + _norm_title(it.get("title")))
            if key in merged:
                prev = merged[key]
                for s in (it.get("found_in") or []):
                    if s not in prev["found_in"]:
                        prev["found_in"].append(s)
                prev["score"] = max(float(prev.get("score") or 0),
                                    float(it.get("score") or 0))
                prev["queries"] = prev.get("queries", []) + [q]
            else:
                nxt = dict(it)
                nxt["queries"] = [q]
                merged[key] = nxt

    items = sorted(merged.values(),
                   key=lambda x: (-float(x.get("score") or 0), -(x.get("year") or 0)))
    # 与 `_merge` 同一个坑：同一篇论文的「arXiv 版没 DOI / 会议版有 DOI」会各来一条。
    # 跨 query 合并时同样按标题指纹收口，无 DOI 的并进有 DOI 的。
    by_title: dict = {}
    for key, it in merged.items():
        if not key.startswith("t:"):
            by_title.setdefault(_norm_title(it.get("title")), key)
    for key in [k for k in merged if k.startswith("t:")]:
        target = by_title.get(_norm_title(merged[key].get("title")))
        if not target or target == key:
            continue
        keep, drop = merged[target], merged.pop(key)
        for s in (drop.get("found_in") or []):
            if s not in keep["found_in"]:
                keep["found_in"].append(s)
        for q in (drop.get("queries") or []):
            if q not in keep.setdefault("queries", []):
                keep["queries"].append(q)
        if not keep.get("abstract") and drop.get("abstract"):
            keep["abstract"] = drop["abstract"]
        keep["score"] = max(float(keep.get("score") or 0), float(drop.get("score") or 0))
        items = [x for x in items if x is not drop]

    items.sort(key=lambda x: (-float(x.get("score") or 0), -(x.get("year") or 0)))
    return {
        "ok": True,
        "queries": qs[:4],
        # 上限给 3×limit 而不是放开 —— 审计报告里挂一屏"可能相关"的文献，
        # 反而让人以为系统在拿文献凑数。
        "items": items[:max(limit * 2, 6)],
        "per_query": per_q,
        "providers": providers,
        "reached": reached_max,
        "of": len(PROVIDERS),
        "elapsed_s": round(time.time() - t0, 2),
        "mode": MODE,
        "note": ("每个参数单独查一次后合并去重；跨 query 同时命中的排在前面"
                 "（说明它不止切题一个侧面）"),
        "boundary": "候选依据，非结论证据；引用前须人读摘要确认。",
    }


def for_findings(findings: Any, paper_title: str = "", *, max_params: int = 3,
                 limit: int = 4, **kw) -> dict:
    """给一批审计结论找外部文献依据（审计链里「搜」的外挂部分）。

    查询词怎么定：**参数名 + 论文标题里的实词**。
    为什么不用整条结论：结论里混着本项目的内部措辞（"未固定""动态值"），
    拿去搜学术库会把结果带偏 —— 文献库里没人这么写。
    """
    findings = list(findings or [])
    if not findings:
        return {"ok": True, "items": [], "queries": [], "by_param": {},
                "note": "本次没有结论，无需外部检索", "mode": MODE}

    title_words = [w for w in _WORD.findall(paper_title or "")
                   if w.lower() not in _STOP][:5]

    # 可疑项优先：对不上的结论最需要外部依据
    def _rank(f):
        st = str(getattr(getattr(f, "status", None), "value", "") or "")
        return (0 if st and st != "consistent" else 1,
                str(getattr(f, "param_key", "")))

    picked, seen = [], set()
    for f in sorted(findings, key=_rank):
        key = str(getattr(f, "param_key", "") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        picked.append(f)
        if len(picked) >= max_params:
            break

    queries, by_param = [], {}
    for f in picked:
        key = str(getattr(f, "param_key", "") or "")
        disp = str(getattr(f, "display_name", "") or key).replace("_", " ")
        q = " ".join([w for w in (disp.split() + title_words) if w])[:180]
        if not q:
            continue
        queries.append(q)
        by_param[key] = q

    if not queries:
        return {"ok": True, "items": [], "queries": [], "by_param": {},
                "note": "结论里没有可用于检索的参数名", "mode": MODE}

    res = search_many(queries, limit=limit, **kw)
    # 每条结果挂回它来自哪个参数的查询（人要知道"这条是给哪个结论找的"）
    for it in res.get("items") or []:
        qs = it.get("queries") or []
        it["for_params"] = [k for k, q in by_param.items() if q in qs]
    res["by_param"] = by_param
    return res
