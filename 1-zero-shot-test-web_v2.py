# 1-zero-shot-test-web_v2.py — Streamlit + LangChain (OpenAI Vision) + Footer OpenAI Icon
import os
import platform
from datetime import datetime
import traceback
import io
import base64
from pathlib import Path
from typing import List, Optional
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
# Mensagens multimodais (texto + imagem) em LangChain:
try:
    from langchain_core.messages import HumanMessage
except Exception:  # fallback para versões antigas
    from langchain.schema import HumanMessage

# Private Functions
from functions.readfiles import load_llm_models_yaml as ldllm

# ======= Setup =======
load_dotenv(find_dotenv())
st.set_page_config(page_title="LLM - Zero Shot (Streamlit)", page_icon="🤖", layout="wide")

APP_TITLE = "Prompt Engineer Project - tests - Images"
PDF_TITLE  = APP_TITLE


# ======= Utils =======
def _read_logo_base64() -> str | None:
    """Procura o logo em caminhos comuns e retorna o base64, se existir."""
    here = Path(__file__).parent.resolve()
    candidates = [
        here / "logos" / "prompteng1.png",
        Path.cwd() / "logos" / "prompteng1.png",
        here / "prompteng1.png",
    ]
    tried = []
    for p in candidates:
        tried.append(str(p))
        if p.exists():
            try:
                return base64.b64encode(p.read_bytes()).decode("utf-8")
            except Exception:
                pass
    st.info("⚠️ Logo not found. Tried paths:<br>" + "<br>".join(tried), icon="⚠️")
    return None

def _to_image_data_url(uploaded_file) -> Optional[str]:
    """Converte um arquivo carregado em data:image/...;base64,..."""
    if uploaded_file is None:
        return None
    raw = uploaded_file.read()
    if not raw:
        return None
    mime = uploaded_file.type or "image/png"
    if not mime.startswith("image/"):
        return None
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _build_mm_content(prompt_text: str, image_data_urls: List[str], image_http_urls: List[str]):
    """Constrói o conteúdo multimodal (texto + imagens)"""
    blocks = []
    if prompt_text.strip():
        blocks.append({"type": "text", "text": prompt_text.strip()})
    for url in image_data_urls:
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    for url in image_http_urls:
        u = url.strip()
        if u:
            blocks.append({"type": "image_url", "image_url": {"url": u}})
    return blocks

def _extract_token_usage(response):
    """Extrai uso de tokens da resposta da API"""
    input_toks = output_toks = total_toks = None
    md = getattr(response, "response_metadata", {}) or {}
    usage = md.get("token_usage") or md.get("usage") or {}
    input_toks  = usage.get("prompt_tokens")     or usage.get("input_tokens")
    output_toks = usage.get("completion_tokens") or usage.get("output_tokens")
    total_toks  = usage.get("total_tokens")
    if total_toks is None and (input_toks and output_toks):
        total_toks = input_toks + output_toks
    return input_toks, output_toks, total_toks

logo_b64 = _read_logo_base64()


# ======= CSS (global) =======
st.markdown(
    """
    <style>
      .header-wrap{ display:flex; align-items:center; gap:14px; margin: 6px 0 4px 0; }
      .header-title{ color:#c00000; font-size:48pt; font-weight:700; line-height:1.0; margin:0; padding:0; }
      .header-logo{ height:150px; width:auto; display:block; }
      .page-divider{ border:0; height:2px; background:linear-gradient(to right,#c9d7e8,#7aa6d9,#c9d7e8); margin:8px 0 16px 0; }

      .blink-error { animation: blinker 1s linear infinite; color:#fff; background:#d32f2f;
                     padding:6px 10px; border-radius:8px; display:inline-block; font-weight:700; letter-spacing:.3px; }
      @keyframes blinker { 50% { opacity:0; } }

      .prompt-label, .model-label { font-size:14pt; color:#c00000; font-weight:700; margin: 6px 0 4px 0; }
      .stSelectbox * { font-size:14pt !important; }
      .stTextInput input, .stTextArea textarea { font-size:14pt !important; }
      .blue-section { color:#ff0000 !important; margin:12px 0 6px 0; font-weight:700; }
      .thumb{ border:1px solid #d0d7de; border-radius:8px; padding:4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======= Header (logo + título) =======
if logo_b64:
    st.markdown(
        f"""
        <div class="header-wrap">
          <img class="header-logo" src="data:image/png;base64,{logo_b64}" alt="Prompt Engineering Logo">
          <div class="header-title">{APP_TITLE}</div>
        </div>
        <hr class="page-divider">
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="header-wrap">
          <div class="header-title">{APP_TITLE}</div>
        </div>
        <hr class="page-divider">
        """,
        unsafe_allow_html=True,
    )

st.title("🤖 Zero-Shot LLM (Streamlit)")
st.caption(f"System: {platform.system()} | Python: {platform.python_version()} | OS name: {os.name}")


# ======= State =======
st.session_state.setdefault("exec_log", "")
st.session_state.setdefault("current_output", "")
st.session_state.setdefault("is_running", False)
st.session_state.setdefault("last_error", "")
st.session_state.setdefault("log_pdf_bytes", b"")
st.session_state.setdefault("usage_rows", [])


# ======= Inputs (texto + imagens) =======
default_q = "Analyze the image(s) and answer: what is this and what key details stand out?"

st.markdown('<div class="prompt-label">Enter your question (prompt)</div>', unsafe_allow_html=True)
question = st.text_input("", value=default_q)

available_models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-mini",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3",
    "o3-mini"
]
vision_capable = {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini", "o3", "o3-mini"}

st.markdown('<div class="model-label">LLM Model</div>', unsafe_allow_html=True)
model_name = st.selectbox("", options=available_models, index=5)

st.markdown('<h3 class="blue-section">Images:</h3>', unsafe_allow_html=True)
up_files = st.file_uploader(
    "Upload 1–6 images (PNG/JPEG/WebP). They will be sent to the LLM.",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

url_text = st.text_area(
    "Optional: paste image URLs (one per line).",
    height=90,
    placeholder="https://example.com/image1.png\nhttps://example.com/image2.jpg"
)

if model_name not in vision_capable and (up_files or url_text.strip()):
    st.info("ℹ️ The selected model may not support **vision**. Only the **text** will be sent.", icon="ℹ️")

# Thumbnails de preview
if up_files:
    cols = st.columns(min(3, len(up_files)))
    for i, uf in enumerate(up_files):
        with cols[i % len(cols)]:
            st.image(uf, use_column_width=True, caption=f"Upload {i+1}", output_format="PNG")

# ======= Botões =======
col1, col2, col3 = st.columns([1, 1, 1], vertical_alignment="bottom")
run_clicked   = col1.button("Run", disabled=st.session_state.is_running)
clear_clicked = col2.button("Clear Log")

dl_txt_placeholder = col3.empty()
dl_txt_placeholder.download_button(
    label="⬇️ Download Log (.txt)",
    data=st.session_state.exec_log or "Log is still empty.",
    file_name=f"llm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain",
    disabled=not bool(st.session_state.exec_log),
)


# ======= Execução =======
if clear_clicked:
    st.session_state["exec_log"] = ""
    st.session_state["current_output"] = ""
    st.session_state["last_error"] = ""
    st.session_state["log_pdf_bytes"] = b""
    st.session_state["usage_rows"] = []
    st.toast("Log cleared!", icon="🧹")


if run_clicked:
    if not question.strip():
        st.warning("Please enter a prompt.")
    else:
        st.session_state.is_running = True
        llm = ChatOpenAI(model=model_name, max_retries=0, timeout=60)

        data_urls = []
        if up_files and model_name in vision_capable:
            for uf in up_files[:6]:
                du = _to_image_data_url(uf)
                if du:
                    data_urls.append(du)

        http_urls = []
        if url_text.strip() and model_name in vision_capable:
            http_urls = [ln.strip() for ln in url_text.strip().splitlines() if ln.strip()]

        if (data_urls or http_urls) and model_name in vision_capable:
            content_blocks = _build_mm_content(question, data_urls, http_urls)
            messages = [HumanMessage(content=content_blocks)]
            payload_for_log = f"{question}\n\n[Attached images: {len(data_urls)+len(http_urls)}]"
        else:
            messages = [HumanMessage(content=question)]
            payload_for_log = question

        with st.status(f"Querying the LLM ({model_name})...", expanded=False):
            started = datetime.now()
            try:
                response = llm.invoke(messages)
                content = getattr(response, "content", str(response))
                st.session_state["last_error"] = ""
                ok = True
            except Exception as e:
                response = None
                content = f"[ERROR] {''.join(traceback.format_exception_only(type(e), e)).strip()}"
                st.session_state["last_error"] = content
                ok = False
            finished = datetime.now()
            duration_s = (finished - started).total_seconds()

        st.session_state["current_output"] = content
        sep = "\n" + ("-" * 80) + "\n"
        block = (
            f"Timestamp: {started.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Model: {model_name}\n"
            f"Prompt:\n{payload_for_log}\n\n"
            f"Response:\n{content}\n"
            f"Duration: {duration_s:.2f}s"
        )
        st.session_state["exec_log"] += (sep if st.session_state["exec_log"] else "") + block

        in_tok, out_tok, tot_tok = (None, None, None)
        if response is not None:
            in_tok, out_tok, tot_tok = _extract_token_usage(response)

        seq = len(st.session_state["usage_rows"]) + 1
        st.session_state["usage_rows"].append({
            "No.": seq,
            "Model": model_name,
            "Input tokens": in_tok or "-",
            "Output tokens": out_tok or "-",
            "Total tokens": tot_tok or ((in_tok or 0)+(out_tok or 0)),
            "Time (s)": f"{duration_s:.2f}",
        })

        if ok:
            st.success("✅ Execution finished — result received.")
        else:
            st.error("An error occurred. See details in the footer.")

        st.session_state.is_running = False


# ======= Output =======
st.markdown('<h3 class="blue-section">LLM Response:</h3>', unsafe_allow_html=True)
st.text_area("Output", key="current_output", height=220, help="Latest run response.")


# ======= Table: Token usage =======
st.markdown('<h3 class="blue-section">Token Usage:</h3>', unsafe_allow_html=True)
def render_usage_table_component(rows):
    rows = rows[-5:] if rows else []
    def cell(v): return "-" if (v is None or v == "") else v
    body_rows = "\n".join([
        f"""
        <tr class='row-{i % 2}'><td>{cell(r.get('No.'))}</td><td>{cell(r.get('Model'))}</td>
        <td>{cell(r.get('Input tokens'))}</td><td>{cell(r.get('Output tokens'))}</td>
        <td>{cell(r.get('Total tokens'))}</td><td>{cell(r.get('Time (s)'))}</td></tr>
        """ for i, r in enumerate(rows)
    ]) if rows else "<tr><td colspan='6'>No records yet</td></tr>"
    html = f"""
    <style>
      table{{width:100%;border-collapse:collapse;font-size:14pt}}
      th,td{{border:1px solid #d0d7de;padding:8px}}
      thead tr{{background:#ADD8E6;font-weight:bold}}
      tbody tr:nth-child(even){{background:#f2f8fc}}
    </style>
    <table>
      <thead><tr><th>No.</th><th>Model</th><th>Input</th><th>Output</th><th>Total</th><th>Time(s)</th></tr></thead>
      <tbody>{body_rows}</tbody>
    </table>"""
    return html
components.html(render_usage_table_component(st.session_state["usage_rows"]), height=280, scrolling=False)


# ======= Execution Log =======
st.subheader("Execution Log")
st.text_area("History", key="exec_log", height=340, disabled=True)


# ======= Errors footer =======
st.markdown("---")
if st.session_state["last_error"]:
    st.markdown('<span class="blink-error">⚠️ EXECUTION ERROR</span>', unsafe_allow_html=True)
    st.text_area("Error details", value=st.session_state["last_error"], height=140, disabled=True)

# ======= OpenAI Footer Icon =======
st.markdown(
    """
    <style>
        .footer-openai {
            margin-top: 25px;
            text-align: center;
        }
        .footer-openai a img {
            width: 60px;
            height: auto;
            opacity: 0.9;
            transition: transform 0.2s ease, opacity 0.2s ease;
        }
        .footer-openai a img:hover {
            transform: scale(1.1);
            opacity: 1.0;
        }
        .footer-openai-text {
            font-size: 11pt;
            color: #666;
            margin-top: 6px;
        }
    </style>

    <div class="footer-openai">
        <a href="https://platform.openai.com/settings/organization/usage" target="_blank" title="OpenAI Usage Dashboard">
            <img src="https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg" alt="OpenAI Logo">
        </a>
        <div class="footer-openai-text">
            © 2025 Prompt Engineer Project - OpenAI Usage Dashboard <br>Powered by Romilson Lemes - Using OpenAI API
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
