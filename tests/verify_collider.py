"""对撞引擎单元测试（阶段四）——纯规则，不调模型。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import collider

CHECKS = 0
FAILURES = []


def check(ok: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    assert isinstance(ok, bool), f"[{label}] ok 不是布尔值: {ok!r}"
    assert isinstance(label, str), f"[{label}] label 不是字符串"
    if not ok:
        FAILURES.append(label)
        print(f"  ✗ {label}")
    else:
        print(f"  ✓ {label}")


def _one(pv=None, cv=None, numeric=None):
    return collider.collide(pv or {}, cv or {}, numeric)


# ---------------------------------------------------------------- 1. 一致

print("1. 一致")
r = _one(
    {"learning_rate": {"value": "1e-4", "source": "Page 3", "doc": "lr=1e-4"}},
    {"learning_rate": {"value": "1e-4", "source": "train.py:42", "doc": "lr = 1e-4"}},
)
check(len(r) == 1, "一致时报一条")
check(r[0].status.value == "consistent", f"状态一致: {r[0].status.value}")
check(r[0].risk.value == "none", f"风险 none: {r[0].risk.value}")
check(r[0].evidence_origin == "both", f"来源 both: {r[0].evidence_origin}")

# 数值写法不同但归一后相等
r = _one(
    {"batch_size": {"value": "32", "source": "P4", "doc": ""}},
    {"batch_size": {"value": "32.0", "source": "cfg.py:3", "doc": ""}},
)
check(len(r) == 1 and r[0].status.value == "consistent", f"32 == 32.0: {r[0].status.value}")

# 百分比与小数等价
r = _one(
    {"dropout": {"value": "0.1", "source": "P4", "doc": ""}},
    {"dropout": {"value": "10%", "source": "cfg.py:3", "doc": ""}},
)
check(len(r) == 1 and r[0].status.value == "consistent", f"0.1 == 10%: {r[0].status.value}")

# 同义词归并
r = _one(
    {"lr": {"value": "1e-4", "source": "P3", "doc": ""}},
    {"learning_rate": {"value": "1e-4", "source": "train.py:1", "doc": ""}},
)
check(len(r) == 1 and r[0].param_key == "learning_rate", f"lr 归并到 learning_rate: {r[0].param_key}")

# ---------------------------------------------------------------- 2. 不一致

print("2. 不一致")
r = _one(
    {"learning_rate": {"value": "1e-4", "source": "Page 3", "doc": "lr=1e-4"}},
    {"learning_rate": {"value": "0.001", "source": "train.py:11", "doc": "lr = 0.001"}},
)
check(len(r) == 1, "不一致时报一条")
check(r[0].status.value == "inconsistent", f"状态不一致: {r[0].status.value}")
check(r[0].risk.value == "low", f"风险 low: {r[0].risk.value}")
check(r[0].paper == "1e-4", f"论文值保留: {r[0].paper}")
check(r[0].code == "0.001", f"代码值保留: {r[0].code}")
check(r[0].paper_source == "Page 3", f"论文出处: {r[0].paper_source}")
check(r[0].code_source == "train.py:11", f"代码出处: {r[0].code_source}")
check(r[0].paper_doc == "lr=1e-4", f"论文原文片段: {r[0].paper_doc}")
check("论文 1e-4 与代码 0.001 不相等" in r[0].note, f"说明含值: {r[0].note}")

# 排序: 高风险在前
r = _one(
    {"batch_size": {"value": "32", "source": "P4", "doc": ""}},
    {"batch_size": {"value": "64", "source": "cfg.py:3", "doc": ""}},
)
r2 = _one(
    {"learning_rate": {"value": "1e-4", "source": "P3", "doc": ""}},
    {"learning_rate": {"value": "1e-3", "source": "train.py:1", "doc": ""}},
)
check(r2[0].param_key == "learning_rate", f"高风险在前: {r2[0].param_key}")

# ---------------------------------------------------------------- 3. 单侧缺失

print("3. 单侧缺失")
r = _one({"learning_rate": {"value": "1e-4", "source": "P3", "doc": ""}}, {})
check(len(r) == 1, "代码缺失时报一条")
check(r[0].status.value == "missing_in_code", f"状态: {r[0].status.value}")
check(r[0].evidence_origin == "paper", f"来源 paper: {r[0].evidence_origin}")

r = _one({}, {"learning_rate": {"value": "0.001", "source": "train.py:1", "doc": ""}})
check(len(r) == 1 and r[0].status.value == "missing_in_paper", f"论文缺失: {r[0].status.value}")
check(r[0].evidence_origin == "code", f"来源 code: {r[0].evidence_origin}")

# 两侧都缺 -> 静默，不报噪声
r = _one({}, {})
check(len(r) == 0, f"两侧都缺不报: {len(r)} 条")

# 空字符串也算缺
r = _one({"learning_rate": {"value": "", "source": "P3", "doc": ""}}, {})
check(len(r) == 0, f"空值不报: {len(r)} 条")

# ---------------------------------------------------------------- 4. 数值合理性

print("4. 数值合理性")
n = collider.numeric_check("learning_rate", "999")
check(n["plausible"] is False, f"999 离谱: {n['note']}")
n = collider.numeric_check("learning_rate", "1e-4")
check(n["plausible"] is True, f"1e-4 合理: {n['note']}")
n = collider.numeric_check("batch_size", "32")
check(n["plausible"] is True, f"batch_size=32 合理: {n['note']}")
n = collider.numeric_check("dropout", "0.1")
check(n["plausible"] is True, f"dropout=0.1 合理: {n['note']}")
n = collider.numeric_check("optimizer", "adam")
check(n["plausible"] is True, f"优化器名跳过: {n['note']}")
n = collider.numeric_check("learning_rate", "")
check(n["plausible"] is True, f"空值跳过: {n['note']}")
# 未知参数不做判断
n = collider.numeric_check("frobnicate", "42")
check(n["plausible"] is True, f"未知参数跳过: {n['note']}")

# 数值合理性在对撞里如实写进 note
r = _one(
    {"learning_rate": {"value": "1e-4", "source": "P3", "doc": ""}},
    {"learning_rate": {"value": "999", "source": "train.py:1", "doc": ""}},
    numeric={"learning_rate": collider.numeric_check("learning_rate", "999")},
)
check("超出" in r[0].numeric_note, f"数值备注带上: {r[0].numeric_note}")

# ---------------------------------------------------------------- 5. 置信度

print("5. 置信度")
r = _one(
    {"learning_rate": {"value": "1e-4", "source": "P3", "doc": "", "confidence": "已确认"}},
    {"learning_rate": {"value": "1e-4", "source": "train.py:1", "doc": "", "confidence": "疑似"}},
)
check(r[0].confidence.value == "inferred", f"一侧疑似则整体疑似: {r[0].confidence.value}")

r = _one(
    {"learning_rate": {"value": "1e-4", "source": "P3", "doc": "", "confidence": "已确认"}},
    {"learning_rate": {"value": "1e-3", "source": "train.py:1", "doc": "", "confidence": "已确认"}},
)
check(r[0].confidence.value == "confirmed", f"两侧已确认则确认: {r[0].confidence.value}")

# ---------------------------------------------------------------- 6. 仲裁

print("6. 仲裁")
c1 = collider.Collision(
    param_key="learning_rate", display_name="Learning Rate",
    category=collider.Category.PARAM,
    status=collider.Status.INCONSISTENT,
    confidence=collider.Confidence.CONFIRMED, risk=collider.Risk.LOW,
    paper="1e-4", paper_source="Page 3", paper_doc="lr=1e-4",
    code="0.001", code_source="train.py:11", code_doc="lr = 0.001",
    evidence_origin="both",
)
a = collider.arbitrate(c1)
check(a.verdict == "维持冲突", f"冲突维持: {a.verdict}")
check(a.requires_human is False, f"普通冲突不需人工: {a.requires_human}")

c2 = collider.Collision(
    param_key="learning_rate", display_name="Learning Rate",
    category=collider.Category.PARAM,
    status=collider.Status.MISSING_IN_CODE,
    confidence=collider.Confidence.INFERRED, risk=collider.Risk.MEDIUM,
    paper="1e-4", paper_source="Page 3", paper_doc="",
    evidence_origin="paper",
)
a = collider.arbitrate(c2)
check(a.verdict == "降级为疑似", f"单侧降级: {a.verdict}")
check(a.requires_human is True, f"单侧需人工: {a.requires_human}")

c3 = collider.Collision(
    param_key="learning_rate", display_name="Learning Rate",
    category=collider.Category.PARAM,
    status=collider.Status.INCONSISTENT,
    confidence=collider.Confidence.CONFIRMED,
    risk=collider.Risk.LOW,
    paper="999", paper_source="Page 3", paper_doc="lr=999",
    code="0.001", code_source="train.py:11", code_doc="lr = 0.001",
    numeric_note="999 超出数值合理区间 [1e-09, 1]",
    evidence_origin="both",
)
a = collider.arbitrate(c3)
check(a.verdict == "升级", f"离谱值升级: {a.verdict}")
check(a.requires_human is True, f"离谱值需人工: {a.requires_human}")

c4 = collider.Collision(
    param_key="batch_size", display_name="Batch Size",
    category=collider.Category.PARAM,
    status=collider.Status.CONSISTENT,
    confidence=collider.Confidence.CONFIRMED, risk=collider.Risk.NONE,
    paper="32", paper_source="P4", paper_doc="",
    code="32", code_source="cfg.py:3", code_doc="",
    evidence_origin="both",
)
a = collider.arbitrate(c4)
check(a.verdict == "一致，无需仲裁", f"一致不仲裁: {a.verdict}")

# ---------------------------------------------------------------- 7. 红队

print("7. 红队")
pv = {"learning_rate": {"value": "1e-4", "source": "Page 3", "doc": "lr=1e-4"}}
cv = {"learning_rate": {"value": "0.001", "source": "", "doc": "lr = 0.001"}}
r = _one(pv, cv)
notes = collider.red_team(r, pv, cv)
check(any(n.flaw == "证据出处缺失" for n in notes), f"红队抓出处缺失: {[n.flaw for n in notes]}")
check(any(n.severity == "high" for n in notes), f"严重度 high: {[n.severity for n in notes]}")

# 单侧定案
r = _one(pv, {})
notes = collider.red_team(r, pv, {})
check(any(n.flaw == "单侧定案" for n in notes), f"红队抓单侧定案: {[n.flaw for n in notes]}")

# 数值离谱
r = _one(
    {"learning_rate": {"value": "999", "source": "P3", "doc": ""}},
    {"learning_rate": {"value": "0.001", "source": "train.py:1", "doc": ""}},
    numeric={"learning_rate": collider.numeric_check("learning_rate", "999")},
)
notes = collider.red_team(r, {"learning_rate": {"value": "999", "source": "P3", "doc": ""}},
                          {"learning_rate": {"value": "0.001", "source": "train.py:1", "doc": ""}})
check(any(n.flaw == "数值离谱" for n in notes), f"红队抓数值离谱: {[n.flaw for n in notes]}")

# 没有漏洞时安静
r = _one(
    {"learning_rate": {"value": "1e-4", "source": "P3", "doc": ""}},
    {"learning_rate": {"value": "1e-4", "source": "train.py:1", "doc": ""}},
)
notes = collider.red_team(r,
                          {"learning_rate": {"value": "1e-4", "source": "P3", "doc": ""}},
                          {"learning_rate": {"value": "1e-4", "source": "train.py:1", "doc": ""}})
check(len(notes) == 0, f"无漏洞时安静: {len(notes)} 条")

# ---------------------------------------------------------------- 8. 序列化

print("8. 序列化")
c1 = collider.Collision(
    param_key="learning_rate", display_name="Learning Rate",
    category=collider.Category.PARAM,
    status=collider.Status.INCONSISTENT,
    confidence=collider.Confidence.CONFIRMED, risk=collider.Risk.LOW,
    paper="1e-4", paper_source="Page 3", paper_doc="lr=1e-4",
    code="0.001", code_source="train.py:11", code_doc="lr = 0.001",
    numeric_note="999 超出合理区间", note="备注", risk_reason="理由",
    evidence_origin="both",
)
d = c1.to_dict()
check(d["param_key"] == "learning_rate", f"to_dict param: {d['param_key']}")
check(d["status"] == "inconsistent", f"to_dict status: {d['status']}")
check(d["category"] == "param", f"to_dict category: {d['category']}")
check(d["paper"] == "1e-4", f"to_dict paper: {d['paper']}")
back = collider.Collision.from_dict(d)
check(back.param_key == "learning_rate", f"from_dict 回绕: {back.param_key}")
check(back.status.value == "inconsistent", f"from_dict status: {back.status.value}")

a = collider.arbitrate(c1)
ad = a.to_dict()
check(ad["verdict"] == "升级", f"仲裁 to_dict: {ad['verdict']}")
check(ad["collision"]["param_key"] == "learning_rate", f"仲裁嵌套 to_dict: {ad['collision']['param_key']}")

# ---------------------------------------------------------------- 9. 角色层

print("9. 角色层")
from paper2lab.agent_roles import Analyst, CodeVerifier, NumericVerifier

a = Analyst()
v = CodeVerifier()
n = NumericVerifier()
check(a.name == "analyst", f"analyst name: {a.name}")
check(v.name == "verifier", f"verifier name: {v.name}")
check(n.name == "numeric", f"numeric name: {n.name}")

class _J:
    def __init__(self, p, c):
        self.paper = p
        self.code = c
        self.paper_view = {}
        self.code_view = {}

class _P:
    def __init__(self):
        self.params = {"learning_rate": _Ev("1e-4", "Page 3", "learning rate = 1e-4")}
        self.params_all = {}
        self.image_params = {}

class _C:
    def __init__(self):
        self.params = {"learning_rate": [_Ev("0.001", "train.py:11", "lr = 0.001")]}
        self.aux_only_keys = set()

class _Ev:
    def __init__(self, raw, loc, snip):
        self.raw_value = raw
        self.location = loc
        self.snippet = snip
        self.normalized = None

j = _J(_P(), _C())
pv = a(j)
cv = v(j)
check(pv["learning_rate"]["value"] == "1e-4", f"analyst 抽到论文值: {pv.get('learning_rate')}")
check(pv["learning_rate"]["source"] == "Page 3", f"analyst 出处: {pv['learning_rate']['source']}")
check(cv["learning_rate"]["value"] == "0.001", f"verifier 抽到代码值: {cv.get('learning_rate')}")
check(cv["learning_rate"]["source"] == "train.py:11", f"verifier 出处: {cv['learning_rate']['source']}")

# 数值验证员不自己抽值，只读前两个 Agent 挂在 job 上的视图
# （节点 node_roles 就是这么串的：先 analyst/code，再把结果挂到 job 上喂 numeric）
j.paper_view = pv
j.code_view = cv
nv = n(j)
check(nv.get("learning_rate", {}).get("plausible") is True, f"数值合理: {nv}")

class _P2:
    params = {"learning_rate": _Ev("999", "Page 3", "lr=999")}
    params_all = {}
    image_params = {}
j2 = _J(_P2(), _C())
j2.paper_view = a(j2)
j2.code_view = v(j2)
nv2 = n(j2)
check(nv2.get("learning_rate", {}).get("plausible") is False, f"离谱值被抓: {nv2}")

# ---------------------------------------------------------------- 10. 规范化标记（模型版）
# 口径: 标记只在**两侧都有**时才参与判定；只一侧有 = 不许单方面定案。
# 规则版永远不填这两个字段，所以它接进来看起来"什么都没变"是**对的**。

print("10. 规范化标记（模型版）")
_pn = {"scheduler": {"value": "余弦退火", "source": "P4", "doc": "",
                     "confidence": "已确认", "normalized": "cosine"}}
_cn = {"scheduler": {"value": "CosineAnnealingLR", "source": "train.py:42", "doc": "",
                     "confidence": "已确认", "normalized": "cosine"}}
_r = collider.collide(_pn, _cn)
check(_r[0].status == collider.Status.CONSISTENT,
      f"两侧标记等价 -> 判一致: {_r[0].status.value}")
check("规范化" in _r[0].note, f"note 写明经规范化后等价: {_r[0].note}")
check(_r[0].paper == "余弦退火" and _r[0].code == "CosineAnnealingLR",
      "原值仍如实保留（没抹掉写法差异）")

_cn2 = {"scheduler": {"value": "CosineAnnealingLR", "source": "train.py:42", "doc": "",
                      "confidence": "已确认"}}
check(collider.collide(_pn, _cn2)[0].status == collider.Status.INCONSISTENT,
      "只有一侧有标记 -> 仍判冲突（不许单方面定案）")
check(collider.Collision.from_dict(_r[0].to_dict()).paper_normalized == "cosine",
      "标记能序列化回绕")
_old = {k: v for k, v in _r[0].to_dict().items() if not k.endswith("_normalized")}
check(collider.Collision.from_dict(_old).paper_normalized == "",
      "老结构（缺这两个键）仍可反序列化")

# ---------------------------------------------------------------- 汇总

print(f"\n{'=' * 50}")
print(f"通过 {CHECKS - len(FAILURES)}/{CHECKS}")
if FAILURES:
    print(f"失败 {len(FAILURES)}: {'；'.join(FAILURES)}")
    sys.exit(1)
print("全部通过")