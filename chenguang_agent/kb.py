"""Load the demo store knowledge base from data/."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INTENTS = ("售前咨询", "物流查询", "售后政策", "投诉情绪", "未识别")


@dataclass
class KnowledgeBase:
    store_name: str
    products: list[dict[str, str]]
    faqs: list[dict[str, str]]
    eval_cases: list[dict[str, str]]
    escalation_script: str
    complaint_keywords: list[str]
    pregnancy_keywords: list[str]
    acute_keywords: list[str]
    efficacy_keywords: list[str]
    tracking_keywords: list[str]
    vocabulary: list[str]
    faq_aliases: dict[str, list[str]]
    kb_version: str = ""


def load_kb(data_dir: Path | None = None) -> KnowledgeBase:
    root = data_dir or DATA_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"找不到知识库目录: {root}")

    store_name = _store_name((root / "store.md").read_text(encoding="utf-8"))
    policy = _sections((root / "guardrails.md").read_text(encoding="utf-8"))
    aliases_doc = json.loads((root / "faq_aliases.json").read_text(encoding="utf-8"))
    vocabulary = _vocabulary(root / "retrieval_terms.txt")
    faqs = _read_csv(root / "faq.csv")
    products = _read_csv(root / "products.csv")
    cases = _read_csv(root / "eval_cases.csv")
    faq_aliases = {key: list(value) for key, value in aliases_doc.get("aliases", {}).items()}

    kb = KnowledgeBase(
        store_name=store_name,
        products=products,
        faqs=faqs,
        eval_cases=cases,
        escalation_script=_script(policy),
        complaint_keywords=_keywords(policy, "投诉关键词"),
        pregnancy_keywords=_keywords(policy, "孕哺关键词"),
        acute_keywords=_keywords(policy, "急性不适关键词"),
        efficacy_keywords=_keywords(policy, "功效承诺关键词"),
        tracking_keywords=_keywords(policy, "物流轨迹关键词"),
        vocabulary=vocabulary,
        faq_aliases=faq_aliases,
        kb_version=kb_fingerprint(root),
    )
    _validate(kb)
    return kb


# Files that change what the bot is allowed to say. Eval cases are not part of the fingerprint.
_VERSION_FILES = (
    "VERSION",
    "store.md",
    "products.csv",
    "faq.csv",
    "faq_aliases.json",
    "guardrails.md",
    "retrieval_terms.txt",
)


def kb_fingerprint(root: Path) -> str:
    version_path = root / "VERSION"
    version = "unversioned"
    if version_path.is_file():
        text = version_path.read_text(encoding="utf-8").strip()
        if text:
            version = text.splitlines()[0].strip() or "unversioned"
    digest = hashlib.sha256()
    for name in _VERSION_FILES:
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"{version}+{digest.hexdigest()[:12]}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def _sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    title: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("# "):
            if title is not None:
                sections[title] = "\n".join(buffer).strip()
            title = line[2:].strip()
            buffer = []
        else:
            buffer.append(line)
    if title is not None:
        sections[title] = "\n".join(buffer).strip()
    return sections


def _section(sections: dict[str, str], prefix: str) -> str:
    for title, body in sections.items():
        if title.startswith(prefix):
            return body
    raise KeyError(f"guardrails.md 缺少章节: {prefix}")


def _keywords(sections: dict[str, str], prefix: str) -> list[str]:
    for line in _section(sections, prefix).splitlines():
        line = line.strip()
        if line and not line.startswith("原始") and not line.startswith("已经"):
            return [part.strip() for part in line.split("|") if part.strip()]
    raise ValueError(f"{prefix} 没有关键词")


def _script(sections: dict[str, str]) -> str:
    body = _section(sections, "转人工固定话术")
    match = re.search(r"「(.+?)」", body)
    if not match:
        raise ValueError("转人工固定话术缺少「」")
    return match.group(1).strip()


def _store_name(text: str) -> str:
    match = re.search(r"\|\s*店名\s*\|\s*(.+?)\s*\|", text)
    if not match:
        raise ValueError("store.md 缺少店名")
    return match.group(1).strip()


def _vocabulary(path: Path) -> list[str]:
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def _validate(kb: KnowledgeBase) -> None:
    if len(kb.eval_cases) != 15:
        raise ValueError(f"评测集应为 15 条，实际 {len(kb.eval_cases)} 条")
    faq_ids = {row["q_id"] for row in kb.faqs}
    for qid in kb.faq_aliases:
        if qid not in faq_ids:
            raise ValueError(f"faq_aliases 引用了不存在的 {qid}")
    blob = "\n".join(row["标准问"] + row["相似问"] for row in kb.faqs)
    missing = [term for term in kb.vocabulary if term not in blob]
    if missing:
        raise ValueError("这些检索词不在任何标准问/相似问里: " + "、".join(missing))
    intents = {row["意图"] for row in kb.faqs}
    unknown = intents - set(INTENTS)
    if unknown:
        raise ValueError(f"FAQ 出现未知意图: {unknown}")
    for row in kb.eval_cases:
        if row["期望意图"] not in INTENTS:
            raise ValueError(f"{row['case_id']} 的期望意图无法识别")
    if not kb.escalation_script:
        raise ValueError("转人工话术为空")
