# prompt-mult-ai-test-web.py
import os
import sys
import platform
from datetime import datetime
import traceback
import io
import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
# Multimodais messages (text + images) in LangChain:
try:
    from langchain_core.messages import HumanMessage
except Exception:  # fallback for old versions
    from langchain.schema import HumanMessage

import pandas as pd  # <<< ADICIONADO para a tabela de seleção

# Private Functions - Romilson Lemes
os.system("cls")
from functions.readfiles import load_llm_models_yaml as ldllm_yaml
llmmodel = ["load_llm_models_yaml"]
print(f"📦 Successfully initialized function  {llmmodel}")

# sys.path.append(os.path.abspath(os.path.dirname("utils.py")))
from functions.utils import sort_dictionary as stdic
llmutil = ["sort_dictionary"]
print(f"📦 Successfully initialized function  {llmutil}")
# sort_dictionary(data, order="asc", key=None):

load_dotenv(find_dotenv())
st.set_page_config(page_title="LLM - Prompt-mult-ai-test-web (Streamlit)", page_icon="🤖", layout="wide")

APP_TITLE = "Prompt Engineering Lab – Multi-AI Simulation"
PDF_TITLE  = APP_TITLE

# ======= Utils =======
def _read_logo_base64() -> str | None:
    """Searches for the logo in common paths and returns its base64 representation, if it exists."""
    here = Path(__file__).parent.resolve()
    candidates = [
        here / "logos" / "prompteng1.png",         # ./logos/prompteng1.png
        Path.cwd() / "logos" / "prompteng1.png",   # <cwd>/logos/prompteng1.png
        here / "prompteng1.png",                   # ./prompteng1.png
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

logo_b64 = _read_logo_base64()

# ======= CSS (global) =======
st.markdown(
    """
    <style>
      /* ===== Header (logo + title) ===== */
      .header-wrap{
        display:flex;
        align-items:center;
        gap:14px;
        margin: 6px 0 4px 0;
      }
      .header-title{
        color: #c00000;
        font-size:36pt;
        font-weight:700;
        line-height:1.0;
        margin: 0;
        padding: 0;
      }
      .header-logo{
        height: 150px;
        width: auto;
        display:block;
      }
      .page-divider{
        border: 0;
        height: 2px;
        background: linear-gradient(to right, #c9d7e8, #7aa6d9, #c9d7e8);
        margin: 8px 0 16px 0;
      }
      .blink-error {
        animation: blinker 1s linear infinite;
        color: #fff;
        background: #d32f2f;
        padding: 6px 10px;
        border-radius: 8px;
        display: inline-block;
        font-weight: 700;
        letter-spacing: .3px;
      }
      @keyframes blinker { 50% { opacity: 0; } }

      .prompt-label, .model-label {
        font-size:14pt;
        color:#c00000;
        font-weight:700;
        margin: 6px 0 4px 0;
      }
      .stSelectbox * { font-size:14pt !important; }
      .stTextInput input { font-size:14pt !important; }

      .blue-section {
        color:#0b64d8;
        margin: 12px 0 6px 0;
        font-weight: 700;
      }
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

st.title("🤖 Test multi models -  LLM (Streamlit)")
st.caption(f"System: {platform.system()} | Python: {platform.python_version()} | OS name: {os.name}")

# ======= State =======
st.session_state.setdefault("exec_log", "")
st.session_state.setdefault("current_output", "")
st.session_state.setdefault("is_running", False)
st.session_state.setdefault("last_error", "")
st.session_state.setdefault("log_pdf_bytes", b"")
st.session_state.setdefault("usage_rows", [])

# ======= Inputs =======
default_q = "Calculate the result of the following expression: ((((45 × 9) / 3) × 1898) / 2.85)"

st.markdown('<div class="prompt-label">Enter your question (prompt)</div>', unsafe_allow_html=True)
question = st.text_input("", value=default_q)

# Read List of Models (from YAML)
yaml_path = "./config/llm_models.yaml"
flat_models = ldllm_yaml(yaml_path, flatten=True)
for info in flat_models:
    print(f"Model Name......: {info['Model']:20} - {info['Platform']}")
    print(f"Platform_API_KEY: {info['Platform_API_KEY']}\n")

print("*"*50)
print(f"flat_models:\n {flat_models}")
print("*"*50)

# Ordena por Platform desc (como já fazia)
results1 = stdic(flat_models, "desc", "Platform")

# ======= NOVO: TABELA DE MULTI-SELEÇÃO (Model | Platform) =======
st.markdown('<div class="model-label">LLM Models (multi-select)</div>', unsafe_allow_html=True)

# DataFrame apenas com as duas colunas solicitadas
df_models = pd.DataFrame([{"Model": r["Model"], "Platform": r["Platform"]} for r in results1])

# Adiciona coluna de seleção booleana (checkbox)
df_models.insert(0, "Select", False)

edited_df = st.data_editor(
    df_models,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Select": st.column_config.CheckboxColumn("Select", help="Selecione os modelos para executar"),
        "Model": st.column_config.TextColumn("Model", help="Nome do modelo"),
        "Platform": st.column_config.TextColumn("Platform", help="Plataforma"),
    },
    disabled=["Model", "Platform"],  # impede edição de texto; só marca/desmarca
    key="models_editor"
)

# Extrai os modelos selecionados
selected_rows = edited_df[edited_df["Select"]]
selected_models = [{"Model": m, "Platform": p} for m, p in zip(selected_rows["Model"], selected_rows["Platform"])]

# Botões
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

# ======= Actions =======
if clear_clicked:
    st.session_state["exec_log"] = ""
    st.session_state["current_output"] = ""
    st.session_state["last_error"] = ""
    st.session_state["log_pdf_bytes"] = b""
    st.session_state["usage_rows"] = []
    st.toast("Log cleared!", icon="🧹")

def _extract_token_usage(response):
    """Extract token counts from LangChain/OpenAI response metadata."""
    input_toks = output_toks = total_toks = None
    md = getattr(response, "response_metadata", {}) or {}
    usage = md.get("token_usage") or md.get("usage") or {}

    input_toks  = usage.get("prompt_tokens")     or usage.get("input_tokens")
    output_toks = usage.get("completion_tokens") or usage.get("output_tokens")
    total_toks  = usage.get("total_tokens")

    if total_toks is None and (input_toks and output_toks):
        total_toks = input_toks + output_toks
    return input_toks, output_toks, total_toks

if run_clicked:
    if not question.strip():
        st.warning("Please enter a prompt.")
    elif len(selected_models) == 0:
        st.warning("Please select at least one model in the table.")
    else:
        st.session_state.is_running = True

        # Executa para cada modelo selecionado
        for item in selected_models:
            model_id = item["Model"]           # nome do modelo que vai para a API
            label_for_ui = f"{item['Model']} | {item['Platform']}"  # exibição no status/log

            with st.status(f"Querying the LLM ({label_for_ui})...", expanded=False):
                started = datetime.now()
                try:
                    llm = ChatOpenAI(model=model_id, max_retries=0, timeout=30)
                    response = llm.invoke(question)
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

            # Atualiza a área de saída (última resposta da rodada)
            st.session_state["current_output"] = content

            # Log
            sep = "\n" + ("-" * 80) + "\n"
            block = (
                f"Timestamp: {started.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Model: {label_for_ui}\n"
                f"Prompt:\n{question}\n\n"
                f"Response:\n{content}\n"
                f"Duration: {duration_s:.2f}s"
            )
            st.session_state["exec_log"] += (sep if st.session_state["exec_log"] else "") + block

            # Tokens
            in_tok, out_tok, tot_tok = (None, None, None)
            if response is not None:
                in_tok, out_tok, tot_tok = _extract_token_usage(response)

            seq = len(st.session_state["usage_rows"]) + 1
            st.session_state["usage_rows"].append({
                "No.": seq,
                "Model": label_for_ui,
                "Input tokens": in_tok if in_tok is not None else "-",
                "Output tokens": out_tok if out_tok is not None else "-",
                "Total tokens": (
                    tot_tok if tot_tok is not None else (
                        (in_tok or 0) + (out_tok or 0) if (in_tok is not None and out_tok is not None) else "-"
                    )
                ),
                "Time (s)": f"{duration_s:.2f}",
            })

            if ok:
                st.success(f"✅ Execution finished — result received from {label_for_ui}.")
                st.toast(f"Result received ({label_for_ui})", icon="✅")
            else:
                st.error(f"An error occurred with {label_for_ui}. See details in the footer.")
                st.toast(f"Execution error ({label_for_ui})", icon="⚠️")

        st.session_state.is_running = False

        dl_txt_placeholder.download_button(
            label="⬇️ Download Log (.txt)",
            data=st.session_state.exec_log,
            file_name=f"llm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            disabled=False,
        )

# ======= LLM Response (blue) =======
st.markdown('<h3 class="blue-section">LLM Response</h3>', unsafe_allow_html=True)
st.text_area(
    "Output",
    key="current_output",
    height=220,
    help="Only the latest run's response.",
)

# ======= Table: Token usage (blue title) =======
st.markdown('<h3 class="blue-section">Token Usage:</h3>', unsafe_allow_html=True)

def render_usage_table_component(rows):
    """Render table com 14pt, cabeçalho com gridlines, zebra rows e bordas estilo Excel."""
    rows = rows[-5:] if rows else []
    def cell(v): return "-" if (v is None or v == "") else v

    body_rows = "\n".join([
        f"""
        <tr class='row-{i % 2}'>
          <td class='no'>{cell(r.get('No.'))}</td>
          <td>{cell(r.get('Model'))}</td>
          <td class='num'>{cell(r.get('Input tokens'))}</td>
          <td class='num'>{cell(r.get('Output tokens'))}</td>
          <td class='num'>{cell(r.get('Total tokens'))}</td>
          <td class='num'>{cell(r.get('Time (s)'))}</td>
        </tr>
        """
        for i, r in enumerate(rows)
    ]) if rows else "<tr><td colspan='6' class='empty'>No records yet</td></tr>"

    html = f"""
    <html>
    <head>
      <style>
        .wrap {{
          border: 1px solid #e0e0e0; border-radius: 6px; overflow: hidden;
          width: 100%;
          max-height: 230px; display: grid; grid-template-rows: auto 1fr;
          box-sizing: border-box;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          font-size: 14pt;
          box-sizing: border-box;
        }}
        thead tr {{
          background: #ADD8E6;
          color: #000;
          font-weight: bold;
        }}
        th, td {{
          padding: 8px;
          border: 1px solid #d0d7de;
          box-sizing: border-box;
          font-size: 14pt;
        }}
        tbody {{
          display: block;
          overflow-y: auto;
          max-height: 180px;
        }}
        thead, tbody tr {{
          display: table;
          width: 100%;
          table-layout: fixed;
        }}
        thead tr th:nth-child(1), tbody tr td.no {{ width: 10%; text-align:right; }}
        thead tr th:nth-child(2), tbody tr td:nth-child(2) {{ width: 25%; }}

        tbody tr.row-0 {{ background: #ffffff; }}
        tbody tr.row-1 {{ background: #f2f8fc; }}

        td.num {{ text-align: right; }}
        td.empty {{ text-align:center; color:#666; padding:12px; }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <table>
          <thead>
            <tr>
              <th>No.</th>
              <th>Model</th>
              <th>Input tokens</th>
              <th>Output tokens</th>
              <th>Total tokens</th>
              <th>Time (s)</th>
            </tr>
          </thead>
          <tbody>
            {body_rows}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """
    return html

components.html(render_usage_table_component(st.session_state["usage_rows"]), height=280, scrolling=False)

# ======= Execution Log =======
st.subheader("Execution Log")
st.text_area(
    "History",
    key="exec_log",
    height=340,
    help="Each run is appended at the end, separated by dashed lines.",
    disabled=True,
)

# ======= Log PDF =======
st.markdown("### Export Log as PDF")

def make_pdf_from_text(text: str, logo_b64: str | None, title: str = PDF_TITLE) -> bytes:
    """
    Gera um PDF com cabeçalho (logo + título) em TODAS as páginas e o conteúdo do log abaixo.
    Inclui UMA LINHA EM BRANCO entre o cabeçalho e o conteúdo.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    left_margin   = 20 * mm
    right_margin  = 20 * mm
    top_margin    = 20 * mm
    bottom_margin = 20 * mm

    title_font_name = "Helvetica-Bold"
    title_font_size = 20
    logo_height_mm  = 16
    header_gap_mm   = 3
    header_line_gap = 2

    def draw_header():
        y_top = height - top_margin
        if logo_b64:
            try:
                img_bytes = base64.b64decode(logo_b64)
                img = ImageReader(io.BytesIO(img_bytes))
                logo_h = logo_height_mm * mm
                iw, ih = img.getSize()
                aspect = iw / ih if ih else 1.0
                logo_w = logo_h * aspect
                c.drawImage(img, left_margin, y_top - logo_h, width=logo_w, height=logo_h, mask='auto')
            except Exception:
                pass

        c.setFont(title_font_name, title_font_size)
        c.setFillColorRGB(0.752, 0.0, 0.0)
        title_y = y_top - (logo_height_mm * mm * 0.65)
        c.drawCentredString(width / 2, title_y, title)

        c.setLineWidth(1)
        c.setStrokeColorRGB(0.79, 0.84, 0.91)
        line_y = y_top - (logo_height_mm * mm) - (header_line_gap * mm)
        c.line(left_margin, line_y, width - right_margin, line_y)
        return line_y - (header_gap_mm * mm)

    text_y = draw_header()
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0, 0, 0)

    max_chars_per_line = 110
    line_height = 12

    cursor_y = text_y - line_height  # 1 linha em branco

    def flush_page_and_new():
        nonlocal cursor_y
        c.showPage()
        cursor_y = draw_header()
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0, 0, 0)
        cursor_y -= line_height

    for raw_line in text.splitlines():
        if raw_line == "":
            if cursor_y - line_height < bottom_margin:
                flush_page_and_new()
            cursor_y -= line_height
            continue
        line = raw_line
        while len(line) > max_chars_per_line:
            part = line[:max_chars_per_line]
            if cursor_y - line_height < bottom_margin:
                flush_page_and_new()
            c.drawString(left_margin, cursor_y, part)
            cursor_y -= line_height
            line = line[max_chars_per_line:]
        if cursor_y - line_height < bottom_margin:
            flush_page_and_new()
        c.drawString(left_margin, cursor_y, line)
        cursor_y -= line_height

    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

colpdf1, colpdf2, colpdf3 = st.columns([1.2, 1.2, 2])
gen_pdf_clicked = colpdf1.button("📄 Generate Log PDF", disabled=not bool(st.session_state.exec_log))
download_pdf_placeholder = colpdf2.empty()
open_new_tab_placeholder = colpdf3.empty()

if gen_pdf_clicked:
    st.session_state["log_pdf_bytes"] = make_pdf_from_text(st.session_state["exec_log"], logo_b64, PDF_TITLE)
    st.toast("PDF generated from log (with header and 1-line gap).", icon="📄")

if st.session_state["log_pdf_bytes"]:
    download_pdf_placeholder.download_button(
        label="⬇️ Download Log PDF",
        data=st.session_state["log_pdf_bytes"],
        file_name=f"llm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )
    b64_pdf = base64.b64encode(st.session_state["log_pdf_bytes"]).decode()
    open_now = open_new_tab_placeholder.button("🔗 Open Log PDF in a new tab")
    if open_now:
        components.html(f"""
            <script>
            (function() {{
                const b64 = "{b64_pdf}";
                const byteChars = atob(b64);
                const byteNumbers = new Uint8Array(byteChars.length);
                for (let i = 0; i < byteChars.length; i++) {{
                    byteNumbers[i] = byteChars.charCodeAt(i);
                }}
                const blob = new Blob([byteNumbers], {{ type: "application/pdf" }});
                const url = URL.createObjectURL(blob);
                window.open(url, "_blank");
            }})();
            </script>
        """, height=0)

# ======= Errors footer =======
st.markdown("---")
if st.session_state["last_error"]:
    st.markdown('<span class="blink-error">⚠️ EXECUTION ERROR</span>', unsafe_allow_html=True)
    st.text_area(
        "Error details",
        value=st.session_state["last_error"],
        height=140,
        key="error_area",
        help="Error returned by the LLM call.",
        disabled=True,
    )
