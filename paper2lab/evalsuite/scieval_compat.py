"""与书生·司南（SciEvalKit）评测契约的对齐层 —— 让 Paper2Lab 能被"同款跑法"评测。

为什么需要这一层
----------------
书生（上海 AI 实验室）的科学智能评测工具链是 **SciEvalKit**
（`github.com/InternScience/SciEvalKit`，榜首挂 OpenCompass 司南榜单）。
它给"自定义智能体"留了一个标准插槽 —— `agent_runner.py`：

    agent.name / agent.model_version / agent.run(sample) -> result.final_answer
    dataset.dataset_name / len(dataset)
    dataset.build_agent_sample(idx) -> EvalSample
    dataset.score_agent_sample(idx, final_answer, **judge_kwargs) -> dict

只要满足这些，任何智能体都能被它拉进去批量跑题、自动聚合成榜单分数。

但整套 SciEvalKit 依赖很重（torch / vllm / transformers / tensorflow-hub ...），
本地跑不动、也没这个必要。所以本模块**逐字复刻**接入所必需的那几类契约
（签名与语义完全一致），使 Paper2Lab 在**零重依赖**下即可被同样的流程评测；
将来要上平台时，只需把 `agent_adapter.Paper2LabAgent` 拷进它的
`scieval/agents/` 并在 `_build_agent_from_config` 的 `AGENT_MODULES` 里登记一行。

对齐清单（来源：SciEvalKit v1 / 2025-12）
----------------------------------------
    scieval/agents/base.py      EvalSample / AgentBase
    scieval/agents/records.py   EvalResult / StepResult / ToolCalling / TrajectoryStore / EvalRecord

与官方实现的差异（明确记录，避免"看起来一样其实不一样"）：
    * `TrajectoryStore` 官方用 `scieval.smp.dump/load`（msgpack/parquet 混合），
      这里改用标准库 json —— 数据形态等价，且不引入 pandas 依赖。
    * 不实现 `smolagents / seed18agent / deepseek32agent` 那三个内置 agent。
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: 契约来源（写入产物与报告，便于答辩时交代口径）
SCHEMA_SOURCE = "SciEvalKit (InternScience) v1 / 2025-12"

#: 对齐的签名清单。测试用它反查本模块 —— 将来官方改接口会在此暴露。
#: 参数名逐字一致（`kwargs` 表示 **kwargs 兜底参数）。
CONTRACT_SIGNATURES: Dict[str, List[str]] = {
    "EvalSample.__init__": ["self", "prompt", "images", "files", "metadata"],
    "AgentBase.__init__": ["self", "name", "model_version", "kwargs"],
    "AgentBase.run": ["self", "sample"],
    "EvalResult.__init__": ["self", "success", "final_answer"],
    "TrajectoryStore.__init__": ["self", "root_dir"],
    "EvalRecord.__init__": ["self", "index", "final_answer", "score", "metadata"],
}


# ---------------------------------------------------------------- 小工具
def _dump(obj: Any, path: str) -> None:
    """JSON 落盘（对齐 scieval.smp.dump 的"写出到文件"语义）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load(path: str) -> Any:
    """JSON 读取（对齐 scieval.smp.load）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 输入契约
class EvalSample:
    """一道题（发给 agent 的题目）。

    与 SciEvalKit `scieval/agents/base.py::EvalSample` 逐字一致。
    `files` 用来携带本题的附件路径（Paper2Lab 用它传论文 PDF / 代码包）。
    """

    def __init__(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        files: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        self.prompt = prompt
        self.images = images or []
        self.files = files or {}
        self.metadata = metadata or {}


# ---------------------------------------------------------------- 输出契约
class ToolCalling:
    def __init__(self, tool_name: Any, tool_input: Any, tool_output: Optional[str] = None):
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.tool_output = tool_output

    def add_response(self, response: str) -> None:
        self.tool_output = response

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
        }


class StepResult:
    def __init__(self, role: str, content: Optional[List[Dict[str, Any]]]):
        self.role = role
        self.content = content if content is not None else []
        self.tool_calling: List[ToolCalling] = []

    def add_tool_calling(self, tool_call: ToolCalling) -> None:
        self.tool_calling.append(tool_call)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calling": [tc.to_dict() for tc in self.tool_calling],
        }


class EvalResult:
    """一次作答（agent 返回给评测器的东西）。

    `final_answer` 是主字段 —— 评测器只认它。
    """

    def __init__(self, success: bool, final_answer: str):
        self.success = success
        self.final_answer = final_answer
        self.steps: List[StepResult] = []

    def add_step(self, step: StepResult) -> None:
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "final_answer": self.final_answer,
            "steps": [s.to_dict() for s in self.steps],
        }


class EvalRecord:
    """一条评测记录（分数 + 元数据）。"""

    def __init__(
        self,
        index: int,
        final_answer: str,
        score: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.index = index
        self.final_answer = final_answer
        self.score = score
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "final_answer": self.final_answer,
            "score": self.score,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------- 落盘
class TrajectoryStore:
    """按题号增量落盘（断点续跑靠它）。与官方同名方法一一对应。"""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def traj_path(self, idx: int) -> str:
        return os.path.join(self.root_dir, f"sample_{idx}_traj.json")

    def eval_path(self, idx: int) -> str:
        return os.path.join(self.root_dir, f"sample_{idx}_eval.json")

    def load_traj(self, idx: int) -> Optional[Dict[str, Any]]:
        path = self.traj_path(idx)
        if not os.path.exists(path):
            return None
        return _load(path)

    def load_eval(self, idx: int) -> Optional[Dict[str, Any]]:
        path = self.eval_path(idx)
        if not os.path.exists(path):
            return None
        return _load(path)

    def save_traj(self, idx: int, result: EvalResult) -> None:
        _dump(result.to_dict(), self.traj_path(idx))

    def save_eval(self, idx: int, record: Any) -> None:
        if hasattr(record, "to_dict"):
            record = record.to_dict()
        _dump(record, self.eval_path(idx))


# ---------------------------------------------------------------- agent 基类
class AgentBase(ABC):
    """智能体基类。`name` + `model_version` + `run()` 三件齐活即可被评测。"""

    name = "agent"

    def __init__(self, name: Optional[str] = None, model_version: Optional[str] = None, **kwargs):
        self.name = name or getattr(self, "name", self.__class__.__name__.lower())
        self.model_version = model_version or "default"

    @abstractmethod
    def run(self, sample: EvalSample) -> EvalResult:
        raise NotImplementedError


# ---------------------------------------------------------------- 聚合
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def run_agent_eval(
    agent: AgentBase,
    dataset: Any,
    work_dir: str,
    nproc: int = 1,
    reuse: bool = False,
    mode: str = "all",
    judge_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """跑一套评测并聚合。

    与 SciEvalKit `agent_runner.run_agent_eval` 的语义一致：
      产出 <work_dir>/agent_eval/<dataset>/<agent.name>/<agent.model_version>/<eval_id>/
      下的逐题记录 + 汇总 summary.json，返回 {指标: 均值}。

    这里把并发默认压到 1（审计是 CPU 密集，且需保证可复现）。
    """
    judge_kwargs = judge_kwargs or {}
    dataset_name = getattr(dataset, "dataset_name", dataset.__class__.__name__)
    root_dir = os.path.join(work_dir, "agent_eval", dataset_name,
                            agent.name, agent.model_version)
    os.makedirs(root_dir, exist_ok=True)
    store = TrajectoryStore(root_dir)

    do_infer = mode in ("all", "infer")
    do_eval = mode in ("all", "eval")

    results: List[Tuple[int, Dict[str, Any], str]] = []
    tasks = list(range(len(dataset)))

    def _one(idx: int) -> Tuple[int, Dict[str, Any], str]:
        final_answer = ""
        traj = store.load_traj(idx) if reuse else None
        if do_infer:
            if traj and traj.get("success"):
                final_answer = traj.get("final_answer", "")
            else:
                sample = dataset.build_agent_sample(idx)
                res = agent.run(sample)
                store.save_traj(idx, res)
                final_answer = res.final_answer
        elif traj:
            final_answer = traj.get("final_answer", "")

        if not do_eval:
            return idx, {}, final_answer

        cached = store.load_eval(idx) if reuse else None
        if cached is not None:
            return idx, cached.get("score", cached), cached.get("final_answer", final_answer)

        score = dataset.score_agent_sample(idx, final_answer, **judge_kwargs)
        rec = EvalRecord(index=idx, final_answer=final_answer, score=score,
                         metadata={"dataset": dataset_name})
        store.save_eval(idx, rec)
        return idx, score, final_answer

    if nproc > 1:
        with ThreadPoolExecutor(max_workers=nproc) as ex:
            futs = [ex.submit(_one, i) for i in tasks]
            for f in as_completed(futs):
                results.append(f.result())
    else:
        for i in tasks:
            results.append(_one(i))

    results.sort(key=lambda x: x[0])

    agg: Dict[str, List[float]] = {}
    for _, score, _ in results:
        for k, v in score.items():
            if _is_number(v):
                agg.setdefault(k, []).append(float(v))
    summary = {k: (sum(v) / len(v) if v else 0.0) for k, v in agg.items()}

    _dump({"schema_source": SCHEMA_SOURCE, "dataset": dataset_name,
           "agent": agent.name, "model_version": agent.model_version,
           "summary": summary}, os.path.join(root_dir, "summary.json"))
    return summary
