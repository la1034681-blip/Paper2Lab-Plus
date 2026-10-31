"""执行视图的派生文案 —— 把「某件工具这一次实际做出了什么」从审计结果里算出来。

**为什么单独成模块**（而不是写在 `app.py` 里）：
界面里「执行记录」表的"结果"列若只写「✅ 执行」，所有工具就都成了一排
"✅ 执行"，用户自然看不出两者在跑不同的东西。所以要用**产出**说话。这段派生逻辑
必须能被单元验证覆盖，而 `app.py` 一 import 就会跑整个 Streamlit 脚本、测不了，
因此抽到这里。

**口径提醒**：本模块只做**展示派生**，不参与任何判定——它读的都是审计结果里已有的字段。
"""
from __future__ import annotations


def agent_panel_title(cov: dict) -> str:
    """「抽取 Agent」面板的标题 —— **必须带会随材料变的数字**。

    踩过的坑（用户实测反馈"换什么材料这面板都没变"）：标题原先只写
    「N 步尝试 · 补回 M 类参数」，而"决定性内核已抽全"的样本恒为「0 步 · 0 类」——
    于是 demo、01_DDPM 这类材料折叠时看到的**一字不差**，面板看起来永远是同一句。
    因此标题固定带上「待补 X 类」（这个数会随材料变），展开后再给细节。
    """
    ag = (cov or {}).get("agent") or {}
    targets = len(ag.get("targets") or [])
    filled = len(ag.get("filled") or {})
    steps = len(ag.get("steps") or [])
    return (f"🤖 抽取 Agent（第 3 环节内部）｜ 待补 {targets} 类 → "
            f"补回 {filled} 类 ｜ {steps} 步尝试")


def tool_outcome(audit, key: str) -> str:
    """某件工具**这一次实际做出了什么**。取不到信息时返回 "—"，绝不编。"""
    cov = audit.coverage or {}
    sc = audit.selfcheck or {}
    try:
        if key == "parse_paper":
            thin = cov.get("thin_page_list") or []
            ev = sum(len(v) for v in (audit.paper.params_all or {}).values())
            # 注: 这里用「取到值的参数种类 / 提到过的参数种类」这个**真比率**，
            # 而不是 coverage 里的 decl_matched/decl_total —— 那两个量不是分子分母关系
            # （前者=抽取到的参数证据条数，后者=正文里"xx = 1.5"式疑似声明句式数）。
            return (f"解析 {cov.get('pages_parsed', 0)}/{cov.get('pages_total', 0)} 页 · "
                    f"取到值的参数种类 {cov.get('kinds_valued', 0)}/"
                    f"{cov.get('kinds_seen', 0)} · 参数证据 {ev} 处"
                    + (f" · {len(thin)} 页文字层稀薄待读图" if thin else ""))
        if key == "analyze_images":
            im = cov.get("image") or {}
            if not im.get("enabled"):
                notes = im.get("notes") or []
                why = str(notes[0])[:40] if notes else "未配置读图通道"
                return f"未启用读图 —— {why}"
            return (f"读图 {im.get('regions_read', 0)}/{im.get('candidates', 0)} 个区域 · "
                    f"采信 {im.get('kept', 0)} 条 · 丢弃 {im.get('dropped', 0)} 条")
        if key == "fill_missing":
            am = audit.paper.agent_meta or {}
            targets = am.get("targets") or []
            filled = am.get("filled") or {}
            steps = am.get("steps") or []
            if not targets:
                return f"无待补目标 · {am.get('stopped_reason') or '参数种类已全覆盖'}"
            blocked = sum(1 for s in steps if s.get("status") == "rejected")
            return (f"待补 {len(targets)} 类 → 补回 {len(filled)} 类 · 共尝试 {len(steps)} 步"
                    + (f" · 挡下 {blocked} 条不可信候选" if blocked else ""))
        if key == "hypothesize":
            # 阶段二 2A：说清"提了几条、其中几条必须实机"—— 只写「✅ 执行」等于没说
            hyps = getattr(audit, "hypotheses", None) or []
            total = int(getattr(audit, "hypothesis_total", 0) or len(hyps))
            if not hyps:
                return "未提出可验证假设（两侧没对不上，或材料只到论文章节）"
            need = sum(1 for h in hyps if getattr(h, "needs_rerun", False))
            return (f"提出 {total} 条假设 · 展示 {len(hyps)} 条"
                    + (f" · 其中 {need} 条必须实机验证" if need else " · 全部静态可判"))
        if key == "design_experiment":
            # 阶段二 2D：说清"设计了几个实验、真跑了几次工具、跑出什么" —— 只写「✅ 执行」等于没说
            exps = getattr(audit, "experiments", None) or []
            total = int(getattr(audit, "experiment_total", 0) or len(exps))
            if not exps:
                return "未设计实验（没有可设计的假设卡）"
            st = getattr(audit, "experiment_stats", None) or {}
            calls = st.get("calls")
            head = f"设计 {total} 个实验 · 展示 {len(exps)} 个"
            if calls is None:
                calls = sum(len(getattr(e, "records", None) or []) for e in exps)
            tails: list = []
            if calls:
                tails.append(f"真跑 {calls} 次工具")
            if st.get("disagree"):
                tails.append(f"复算对不上 {st['disagree']}")
            if st.get("machine"):
                tails.append(f"需实机 {st['machine']}")
            return head + (" · " + " · ".join(tails) if tails else "")
        if key == "read_figures":
            # 阶段二 2E：说清"读了几条、来自哪条通道、有没有真的读图"
            clues = getattr(audit, "figure_clues", None) or []
            total = int(getattr(audit, "figure_total", 0) or len(clues))
            st = getattr(audit, "figure_stats", None) or {}
            if not clues:
                return "未读到图表线索（正文里没有独立成行的图注）"
            head = f"读到 {total} 条图表线索 · 展示 {len(clues)} 条"
            tails: list = []
            cap = sum(1 for c in clues if getattr(c, "source", "") == "caption")
            vlm = sum(1 for c in clues if getattr(c, "source", "") == "vlm")
            if cap:
                tails.append(f"图注原文 {cap}")
            if vlm:
                tails.append(f"读图 {vlm}（两遍取交集）")
            elif not st.get("enabled"):
                tails.append("读图未启用（无视觉 key），仅图注原文")
            pages = st.get("pages") or []
            if pages:
                tails.append(f"涉及 {len(pages)} 页")
            return head + (" · " + " · ".join(tails) if tails else "")
        if key == "chem_check":
            # 阶段二 2E-化学：这一步**算什么、抓出几条硬错**，一句话说清。
            # 它是独立工具（2026-10-25 从图表解析拆出），产出必须自己能交代，不能借别家的。
            items = list(getattr(audit, "chem_items", None) or [])
            total = int(getattr(audit, "chem_total", 0) or len(items))
            if not total:
                return "未检测到化学身份（正文里没有分子式 / CAS / SMILES / 分子量声明）"
            bad = sum(1 for c in items
                      if str(getattr(c, "status", "")) in ("inconsistent", "invalid"))
            fb = bool(getattr(audit, "chem_fallback", False))
            return (f"核验 {total} 条化学身份 · 可判定 {len(items)} 条"
                    + (f" · {bad} 条对不上 / 不合规" if bad else " · 全部对得上")
                    + ("（该步被跳过，程序兜底补算）" if fb else ""))
        if key == "analyze_code":
            ok, total = cov.get("code_ok", 0), cov.get("code_files", 0)
            failed = len(cov.get("code_failed_list") or [])
            unsup = len(cov.get("unsupported") or [])
            return (f"扫描 {total} 个文件 · 成功 {ok}"
                    + (f" · 失败 {failed}" if failed else "")
                    + (f" · 暂不支持 {unsup} 个" if unsup else ""))
        if key == "align":
            risky = sum(1 for f in audit.findings if f.status.value != "consistent")
            return f"产出 {len(audit.findings)} 条结论（风险 {risky} 条）"
        if key == "llm_analyze":
            sh = (audit.readiness or {}).get("llm_shards") or {}
            if not sh or not sh.get("total"):
                return "无待归因结论"
            return (f"归因 {sh.get('total', 0)} 条 · {sh.get('mode')}"
                    f"（{sh.get('workers', '—')} 路）· 失败 {sh.get('failed', 0)}")
        if key == "selfcheck":
            return (f"证据自证 {sc.get('passed', 0)}/{sc.get('checked', 0)} 通过 · "
                    f"审计可信度 {cov.get('confidence_score', '—')}/100")
        if key == "rootcause":
            tr = list(getattr(audit, "rootcause", None) or [])
            if not tr:
                return "无待追查的项"
            located = sum(1 for t in tr
                          if getattr(t, "confidence", "none") != "none")
            return (f"追查 {len(tr)} 项 · 可定位 {located} 项"
                    + (f" · 其余 {len(tr) - located} 项明说「无法定位」"
                       if located < len(tr) else ""))
        if key == "patch":
            ps = list(getattr(audit, "patches", None) or [])
            if not ps:
                return "无待出补丁的结论"
            ok = sum(1 for p in ps if getattr(p, "status", "") == "proposed")
            return (f"给方案 {ok} 条 / 拒绝 {len(ps) - ok} 条"
                    "（拒绝理由进报告与本面板）")
        if key == "compute":
            # 阶段二 2C：**② 算 / ③ 判**两段分开记 —— 只说"算了几条"看不出分工。
            cr = getattr(audit, "compute", None) or {}
            rels = cr.get("relations") or []
            if not rels:
                return "未算任何关系式（输入不齐，未用默认值补数）"
            flagged = int(cr.get("flagged", 0) or 0)
            skipped = int(cr.get("skipped", 0) or 0)
            done = sum(1 for r in rels if r.get("judged_by"))
            return (f"② 算数 → ③ 判定 {done} 条 · 一致/对不上 {flagged} 条"
                    + (f" · 未算 {skipped} 条" if skipped else ""))
        if key == "report":
            return (f"报告含 {len(audit.findings)} 条结论"
                    + (" + 人工确认记录" if getattr(audit, "clarifications", None) else ""))
    except Exception:      # noqa: BLE001
        return "—"
    return "—"
