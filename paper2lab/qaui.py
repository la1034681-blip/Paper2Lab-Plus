"""问答窗口 · 界面（Streamlit 侧）。

为什么用**弹窗**而不是新增一个导航页：
  · 用户问的是「我刚看到的这个东西」，眼下的上下文必须还在背后 ——
    切页会把结论页顶掉，回来还得找位置；
  · 弹窗可以随时关掉，关掉时页面状态一点没变（不改判定、不改数据）。

入口放两处，都是**页面最上面、不用滚就能看见**的位置：
  ① 指挥台顶栏右上角（首屏那颗会呼吸的高亮按钮）；
  ② 结果页侧栏最顶（换到别的页也跑不掉）。

铁律：问答出错**绝不影响审计页** —— 所有渲染都包在 try 里，
最坏情况是这个按钮点了没反应，而不是整页白掉。
"""
from __future__ import annotations

import html as _html

import streamlit as st

from .llm import begin_run as _begin_run

from . import qa
from .models import Status

OPEN_KEY = "qa_open"          # 弹窗开着没有
HIST_KEY = "qa_hist"          # [{"role": "user"/"assistant", "content": str, "note": str}]
PEND_KEY = "qa_pend"          # 由快捷按钮塞进来的待提问（等下一轮渲染时消费）

#: 相对时间显示用（不引第三方依赖，问答里没必要）
_ENTRY_LABEL = "💬 问 一 问"


def _e(v) -> str:
    return _html.escape("" if v is None else str(v))


def entry_css() -> str:
    """入口按钮 + 弹窗配色。只在需要时输出（指挥台页 / 结果页）。"""
    return """<style>
/* ---- 入口：会呼吸的高亮胶囊。刻意做成整页唯一一个「亮色」元素，
        一眼就能找到；不是靠位置猜，而是靠颜色对比 ---- */
div[class*="st-key-qa_open_head"] button,
div[class*="st-key-qa_open_side"] button{
  background:linear-gradient(135deg,#2f6df6 0%,#7b4dff 100%) !important;
  color:#ffffff !important;border:0 !important;border-radius:999px !important;
  font-weight:800 !important;letter-spacing:.04em !important;
  box-shadow:0 0 0 1px rgba(140,175,255,.45),0 6px 20px rgba(70,110,255,.45) !important;
  animation:qaGlow 2.8s ease-in-out infinite;
}
div[class*="st-key-qa_open_head"] button:hover,
div[class*="st-key-qa_open_side"] button:hover{
  background:linear-gradient(135deg,#4a82ff 0%,#9366ff 100%) !important;
  transform:translateY(-1px);
}
@keyframes qaGlow{
  0%,100%{box-shadow:0 0 0 1px rgba(140,175,255,.40),0 4px 14px rgba(70,110,255,.35);}
  50%    {box-shadow:0 0 0 3px rgba(140,175,255,.18),0 8px 26px rgba(70,110,255,.60);}
}
/* ---- 弹窗：跟着整站走深色，否则一块白板突然盖在深色大屏上很割裂 ---- */
div[data-testid="stDialog"] > div{background:#0b1220 !important;
  border:1px solid #1e2b47 !important;border-radius:16px !important;
  box-shadow:0 24px 60px rgba(0,0,0,.6) !important;}
div[data-testid="stDialog"] h2{color:#e8f0ff !important;font-size:19px !important;}
div[data-testid="stDialog"] p,
div[data-testid="stDialog"] li,
div[data-testid="stDialog"] strong{color:#c3d1e8 !important;}
div[data-testid="stDialog"] [data-testid="stCaptionContainer"] p{color:#5f7099 !important;}
div[data-testid="stDialog"] hr{border-color:#1b2742 !important;}
/* 消息气泡 */
div[data-testid="stDialog"] [data-testid="stChatMessage"]{
  background:#101a2e !important;border:1px solid #1b2742 !important;
  border-radius:12px !important;padding:10px 14px !important;margin-bottom:8px !important;}
div[data-testid="stDialog"] [data-testid="stChatMessage"] p,
div[data-testid="stDialog"] [data-testid="stChatMessage"] li{font-size:14px !important;
  line-height:1.72 !important;color:#cdd9ee !important;}
/* 输入区。
   这里刻意**不用 `st.chat_input`**：它放进弹窗后，文字会贴着外框上沿被裁掉一截
   （实测：无输入时占位文字、有输入时首行，都只剩下半截 —— 换了背景色也没用，
   是它自己的高度计算在弹窗里失真）。改成一个表单 + 文本域，高度由我们说了算。 */
div[data-testid="stDialog"] .stForm{border:0 !important;padding:0 !important;}
div[data-testid="stDialog"] [data-testid="stTextArea"] textarea{
  background:#070d19 !important;color:#dbe6f8 !important;
  border:1px solid #22304d !important;border-radius:10px !important;
  font-size:14px !important;line-height:1.7 !important;caret-color:#7b9dff !important;}
div[data-testid="stDialog"] [data-testid="stTextArea"] textarea:focus{
  border-color:#3b5286 !important;box-shadow:none !important;}
div[data-testid="stDialog"] [data-testid="stTextArea"] textarea::placeholder{
  color:#54648a !important;}
/* 发送键做成和入口同一套亮色，一眼知道"就按这里" */
div[data-testid="stDialog"] [data-testid="stFormSubmitButton"] button{
  background:linear-gradient(135deg,#2f6df6 0%,#7b4dff 100%) !important;
  color:#fff !important;border:0 !important;font-weight:700 !important;
  border-radius:9px !important;}
div[data-testid="stDialog"] [data-testid="stFormSubmitButton"] button:hover{
  background:linear-gradient(135deg,#4a82ff 0%,#9366ff 100%) !important;}
/* 弹窗里的普通按钮（清空 / 结束）—— 同理，默认是浅色 */
div[data-testid="stDialog"] .stButton>button{
  background:#0f1a2d !important;color:#9fb3d6 !important;
  border:1px solid #22304d !important;border-radius:9px !important;}
div[data-testid="stDialog"] .stButton>button:hover{
  color:#dbe6f8 !important;border-color:#3b5286 !important;background:#132039 !important;}
/* 快捷问题：小一号、半透明，别抢主按钮的注意力 */
div[data-testid="stDialog"] div[class*="st-key-qa_chip"] button{
  background:#0f1a2d !important;color:#9fb3d6 !important;border:1px solid #22304d !important;
  border-radius:999px !important;font-weight:600 !important;font-size:12px !important;
  padding:4px 10px !important;min-height:0 !important;}
div[data-testid="stDialog"] div[class*="st-key-qa_chip"] button:hover{
  color:#dbe6f8 !important;border-color:#3b5286 !important;}
</style>"""


def render_entry(key: str = "qa_open_head", label: str = _ENTRY_LABEL,
                 help_text: str = "就这一轮的审计结果继续追问，多轮对话，随时可关") -> bool:
    """画一颗入口按钮，返回是否被点击。

    只负责「画 + 报点击」，开不开弹窗由调用方决定 ——
    这样同一个入口能放在顶栏、侧栏、任何地方，行为一致。
    """
    try:
        st.html(entry_css())
    except Exception:      # noqa: BLE001
        pass
    return bool(st.button(label, key=key, use_container_width=True,
                          help=help_text))


def _ctx_line(audit, meta: dict) -> str:
    """弹窗顶部那条上下文 —— 让用户一眼确认「它在拿哪份材料回答我」。"""
    if audit is None:
        head = "还没有审计结论 · 可以先问平台怎么用，跑完一轮后才能问这份材料"
        detail = ""
    else:
        r = getattr(audit, "readiness", None) or {}
        total = r.get("total")
        fs = getattr(audit, "findings", None) or []
        n = len(fs)
        # ⚠ 不能用 `"consistent" not in str(status)` 判：`inconsistent` 里就含
        #   `consistent`；而且 `class Status(str, Enum)` 的 str() 是 "Status.INCONSISTENT"，
        #   小写匹配一次都命中不了 —— 实测因此把 13 条全报成"对不上"。按枚举比。
        bad = len([f for f in fs if getattr(f, "status", None) != Status.CONSISTENT])
        title = (getattr(getattr(audit, "paper", None), "title", "") or "未识别标题")
        head = (f"《{_e(title)[:46]}》　对表 {n} 条（{bad} 条对不上）　"
                f"就绪度 {'—' if total is None else str(total)}/100")
        detail = _e(meta.get("files") or "")
    return (f"<div style='background:#0e1728;border:1px solid #1b2742;border-radius:10px;"
            f"padding:8px 12px;font-size:12px;color:#8b9ab8;margin-bottom:6px'>"
            f"<div style='color:#c3d1e8;font-weight:700'>{head}</div>"
            + (f"<div style='color:#4f5f80;margin-top:2px'>{detail}</div>" if detail else "")
            # 顶部的模型名（原「● deepseek-chat」那枚绿点徽标）**已整段删除**
            # （2026-10-31 用户要求："把左上角显示的模型名称给去除掉"）。
            # 为什么删得掉：模型名在下面「高级设置 · 模型 API」的三个框里本来就写着，
            #   弹窗里再复述一遍是噪音；而且它会让人把「哪条通道答的」和
            #   「哪条通道配的」混在一起 —— 一旦两者不一致就误判成程序出错。
            # ⚠ `render_dialog(api_key=...)` 参数**不能删**：`qa.answer` 还要用它。
            + "<div style='margin-top:4px'>数据来源：本轮审计结果（不改判定）</div></div>")


def _welcome(audit) -> str:
    if audit is None:
        return ("还没有审计结论。你可以先问我「这个平台怎么用」「审计都查什么」，"
                "或者关掉窗口点「开始科研审计」跑一轮再来。")
    return ("我只看这一轮审计算出来的东西，不会另做结论。\n\n"
            "可以问：某个数为什么对不上、复现要先改哪一处、这次有没有漏查、"
            "假设卡和实验卡各自说明了什么。")


def render_dialog(audit, meta: dict | None = None, *, api_key: str = "",
                  base_url: str = "", model: str = "", report: str = "") -> None:
    """打开（或重画）问答弹窗。调用前请先判断 OPEN_KEY。

    多轮怎么实现：历史存在 `session_state`，每一轮把最近 8 条压进提示词
    —— 不用额外的会话服务，刷新/关窗都不影响已经产生的审计数据。
    """
    meta = meta or {}

    def _on_close() -> None:
        """用户点右上角 X 关窗时，把开关真正关掉。

        ⚠ 2026-10-31 真机踩到：Streamlit 关弹窗会触发一次重跑，但**不会**替我们
        改 `session_state`。原来 `OPEN_KEY` 一直是 True，于是重跑时又把弹窗画了
        一遍 —— 用户看到的是「这个窗关不掉」，而且它一直盖住页面，背后的大屏和
        左侧导航**全都点不到**（探针后面 10 项全挂就是这个原因）。
        本版 Streamlit 支持 `on_dismiss` 回调，在这里落一次地面。
        """
        st.session_state[OPEN_KEY] = False

    @st.dialog("💬 问一问 · 就这份材料深入答疑", width="large",
               on_dismiss=_on_close)
    def _dlg() -> None:
        st.html(entry_css())
        st.html(_ctx_line(audit, meta))

        hist = st.session_state.setdefault(HIST_KEY, [])
        asked: str = ""

        # ---- 快捷问题：第一次进来给四个「问了就有东西」的方向
        if len(hist) <= 1:
            st.caption("快捷提问（点一下直接问）")
            chips = st.columns(2, gap="small")
            for i, text in enumerate(qa.QUICK):
                with chips[i % 2]:
                    if st.button(text, key=f"qa_chip_{i}", use_container_width=True):
                        asked = text

        # ---- 历史（有内容时才限高；空态不限高，否则欢迎语下面拖着一个大空洞）
        # ⚠ `height=None` 不是"不限高"，Streamlit 会直接抛 StreamlitInvalidHeightError
        #   （实测把弹窗整块打成红底报错）。要"自然高度"就**别传 height**。
        box = st.container(height=300, border=False) if hist \
            else st.container(border=False)
        with box:
            if not hist:
                st.markdown(_welcome(audit))
            for turn in hist:
                with st.chat_message(turn.get("role", "assistant")):
                    st.markdown(turn.get("content") or "")
                    if turn.get("note"):
                        st.caption(turn["note"])

        # ---- 输入：表单而不是 chat_input（原因见 entry_css 里的注释）
        with st.form("qa_form", clear_on_submit=True, border=False):
            typed = st.text_area("要问的问题", height=88,
                                 placeholder="就这份材料继续追问，例如：论文写 1e-4、代码写 0.001，到底哪个是对的？",
                                 label_visibility="collapsed")
            _sc = st.columns([1.1, 3.4], gap="small")
            with _sc[0]:
                sent = st.form_submit_button("发送 ⏎", use_container_width=True)
            with _sc[1]:
                st.caption("Ctrl/⌘ + Enter 也能发送；回答只解释，不改判定。")
        if sent and (typed or "").strip():
            asked = typed.strip()

        # ---- 一轮：写入问题 -> 调模型 -> 落历史 -> 重画
        if asked:
            # 2026-10-30 晚：新的一次提问**作废上一次**（包括还在生成的那一轮）。
            #   本地模型单槽、上下文有限，两问同时进来必然挤爆，用户只看到"调用失败"。
            _begin_run()
            hist.append({"role": "user", "content": asked})
            with st.chat_message("assistant"):
                with st.spinner("在翻这一轮的审计数据…"):
                    try:
                        body, note = qa.answer(audit, asked, hist, api_key=api_key,
                                               base_url=base_url, model=model,
                                               meta=meta, report=report)
                    except qa.RunCancelled:
                        # 2026-10-30 晚：「点一次新的就把上一次终止」——
                        #   这一轮被新的一次点击作废，**不是失败**，静默丢弃：
                        #   撤掉刚写进去的那条提问，不留半截对话、不报错。
                        hist.pop()
                        st.rerun()
                st.markdown(body)
                st.caption(note)
            hist.append({"role": "assistant", "content": body, "note": note})
            st.rerun()

        # ---- 底部：清空 / 结束
        foot = st.columns([1, 1, 3], gap="small")
        with foot[0]:
            if st.button("🧹 清空对话", key="qa_clear", use_container_width=True):
                st.session_state[HIST_KEY] = []
                st.rerun()
        with foot[1]:
            if st.button("✕ 结束问答", key="qa_close", use_container_width=True):
                st.session_state[OPEN_KEY] = False
                st.rerun()
        with foot[2]:
            st.caption("关掉窗口不会丢结论，重新打开还能接着问；"
                       "对话只用于解释，不改任何判定与分数。")

    _dlg()
