# prompt-mult-ai-test-web.py — Multi-model + (texto + imagens) + PDF + tabela de tokens + Rodapé OpenAI
import os
import sys
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
# Multimodal messages (text + images) in LangChain:
try:
    from langchain_core.messages import HumanMessage
except Exception:  # fallback for old versions
    from langchain.schema import HumanMessage

import pandas as pd  # para as tabelas

# ================= Private Functions - Romilson Lemes =================
os.system("cls")
from functions.readfiles import load_llm_models_yaml as ldllm_yaml
llmmodel = ["load_llm_models_yaml"]
print(f"📦 Successfully initialized function  {llmmodel}")

from functions.utils import sort_dictionary as stdic
llmutil = ["sort_dictionary"]
print(f"📦 Successfully initialized function  {llmutil}")

# ================= Setup =================
load_dotenv(find_dotenv())
st.set_page_config(page_title="LLM - Prompt-mult-ai-test-web (Streamlit)", page_icon="🤖", layout="wide")

APP_TITLE = "Prompt Engineering Lab – Multi-AI Simulation"
PDF_TITLE  = APP_TITLE

# ================= Sidebar (PDF settings) =================
st.sidebar.header("PDF Settings")
pdf_title_font_size = st.sidebar.slider("Title font size (PDF)", min_value=10, max_value=25, value=15, step=1)

# ================= Utils (logo, imagens, multimodal, tokens) =================
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

def _to_image_data_url(uploaded_file) -> Optional[str]:
    """Converte um arquivo carregado em data:image/...;base64,..."""
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
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

# ================= CSS (global) =================
DARK_BLUE = "#0b3d91"  # azul escuro para tabelas
st.markdown(
    f"""
    <style>
      .header-wrap{{ display:flex; align-items:center; gap:14px; margin:6px 0 4px 0; }}
      .header-title{{ color:#c00000; font-size:36pt; font-weight:700; line-height:1.0; margin:0; padding:0; }}
      .header-logo{{ height:150px; width:auto; display:block; }}
      .page-divider{{ border:0; height:2px; background:linear-gradient(to right,#c9d7e8,#7aa6d9,#c9d7e8); margin:8px 0 16px 0; }}
      .blink-error{{ animation:blinker 1s linear infinite; color:#fff; background:#d32f2f; padding:6px 10px; border-radius:8px; display:inline-block; font-weight:700; letter-spacing:.3px; }}
      @keyframes blinker {{ 50% {{ opacity:0; }} }}
      .prompt-label,.model-label{{ font-size:14pt; color:#c00000; font-weight:700; margin:6px 0 4px 0; }}
      .stSelectbox *{{ font-size:14pt !important; }} .stTextInput input, .stTextArea textarea{{ font-size:14pt !important; }}
      .blue-section{{ color:#0b64d8; margin:12px 0 6px 0; font-weight:700; }}

      /* Azul-escuro em todas as tabelas data_editor (modelos e token usage) */
      [data-testid="stDataEditor"] * {{ color: {DARK_BLUE} !important; }}
      [data-testid="stDataEditor"] thead * {{ font-weight: 700 !important; }}

      .thumb{{ border:1px solid #d0d7de; border-radius:8px; padding:4px; }}

      /* Cabeçalho da seção de imagens */
      .images-header {{
        margin: 10px 0 6px 0;
        padding: 8px 12px;
        background: #eef6ff;
        border: 1px solid #cfe5ff;
        border-radius: 10px;
        font-weight: 700;
        color: #0b64d8;
      }}

      /* Rodapé de numeração da miniatura na UI */
      .img-caption {{
        text-align: center;
        font-size: 12px;
        color: #444;
        margin-top: 4px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================= Header (logo + título) =================
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
        <div class="header-wrap"><div class="header-title">{APP_TITLE}</div></div>
        <hr class="page-divider">
        """,
        unsafe_allow_html=True,
    )

st.title("🤖 Test multi models -  LLM (Streamlit)")
st.caption(f"System: {platform.system()} | Python: {platform.python_version()} | OS name: {os.name}")

# ================= State =================
st.session_state.setdefault("exec_log", "")
st.session_state.setdefault("current_output", "")
st.session_state.setdefault("is_running", False)
st.session_state.setdefault("last_error", "")
st.session_state.setdefault("log_pdf_bytes", b"")
st.session_state.setdefault("usage_rows", [])

# Persistência de imagens
st.session_state.setdefault("image_data_urls", [])   # data URLs das imagens carregadas (uploads)
st.session_state.setdefault("image_http_urls", [])   # URLs HTTP coladas

# ================= Inputs (texto) =================
default_q = "Analyze the image(s) and answer: what is this and what key details stand out?"
st.markdown('<div class="prompt-label">Enter your question (prompt)</div>', unsafe_allow_html=True)
question = st.text_input("", value=default_q)

# ================= Lê lista de modelos do YAML (mantido) =================
yaml_path = "./config/llm_models.yaml"
flat_models = ldllm_yaml(yaml_path, flatten=True)
for info in flat_models:
    print(f"Model Name......: {info['Model']:20} - {info['Platform']}")
    print(f"Platform_API_KEY: {info['Platform_API_KEY']}\n")

print("*"*50); print(f"flat_models:\n {flat_models}"); print("*"*50)

# Ordena por Platform desc (mantido)
results1 = stdic(flat_models, "desc", "Platform")

# ================= Tabela multi-seleção de modelos (mantido) =================
st.markdown('<div class="model-label">LLM Models (multi-select)</div>', unsafe_allow_html=True)
col_models, col_rest = st.columns([0.2, 0.8])

df_models = pd.DataFrame([{"Model": r["Model"], "Platform": r["Platform"]} for r in results1])
df_models.insert(0, "Select", False)

VISIBLE_ROWS = 5
ROW_PX, HEADER_PX, PADDING_PX = 36, 40, 28
models_editor_height = HEADER_PX + VISIBLE_ROWS * ROW_PX + PADDING_PX  # ~248px

with col_models:
    edited_df = st.data_editor(
        df_models,
        hide_index=True,
        use_container_width=True,
        height=models_editor_height,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", help="Selecione os modelos para executar", width="small"),
            "Model": st.column_config.TextColumn("Model", help="Nome do modelo", width="medium"),
            "Platform": st.column_config.TextColumn("Platform", help="Plataforma", width="medium"),
        },
        disabled=["Model", "Platform"],
        key="models_editor"
    )

selected_rows = edited_df[edited_df["Select"]]
selected_models = [{"Model": m, "Platform": p} for m, p in zip(selected_rows["Model"], selected_rows["Platform"])]

# ================= Entrada de IMAGENS — uploads + URLs =================
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

# Processar uploads e salvar data-URLs em session_state
if up_files:
    data_urls_preview = []
    for uf in up_files[:6]:
        du = _to_image_data_url(uf)
        if du:
            data_urls_preview.append(du)
    st.session_state["image_data_urls"] = data_urls_preview
else:
    st.session_state["image_data_urls"] = []

# Processar texto de URLs HTTP e salvar
if url_text and url_text.strip():
    http_urls = [ln.strip() for ln in url_text.strip().splitlines() if ln.strip()]
    st.session_state["image_http_urls"] = http_urls
else:
    st.session_state["image_http_urls"] = []

# ======= PRÉ-VISUALIZAÇÃO (com cabeçalho + rodapé numerado) =======
if st.session_state["image_data_urls"]:
    # Cabeçalho acima das imagens (solicitado)
    st.markdown('<div class="images-header">🔎 This document presents an analysis of the images below:</div>', unsafe_allow_html=True)

    cols = st.columns(min(3, len(st.session_state["image_data_urls"])))
    for i, du in enumerate(st.session_state["image_data_urls"]):
        with cols[i % len(cols)]:
            st.image(du, use_column_width=True, caption=None, output_format="PNG")
            # Rodapé numerado
            st.markdown(f'<div class="img-caption">Image {i+1:02d}</div>', unsafe_allow_html=True)

# Conjunto de modelos com visão
vision_capable = {
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
    "gpt-5", "gpt-5-mini", "o3", "o3-mini"
}

# ================= Botões =================
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

# ================= Actions =================
if clear_clicked:
    st.session_state["exec_log"] = ""
    st.session_state["current_output"] = ""
    st.session_state["last_error"] = ""
    st.session_state["log_pdf_bytes"] = b""
    st.session_state["usage_rows"] = []
    st.session_state["image_data_urls"] = []
    st.session_state["image_http_urls"] = []
    st.toast("Log cleared!", icon="🧹")

if run_clicked:
    if not question.strip():
        st.warning("Please enter a prompt.")
    elif len(selected_models) == 0:
        st.warning("Please select at least one model in the table.")
    else:
        st.session_state.is_running = True

        data_urls_master: List[str] = st.session_state.get("image_data_urls", []) or []
        http_urls_master: List[str] = st.session_state.get("image_http_urls", []) or []

        for item in selected_models:
            model_id = item["Model"]
            label_for_ui = f"{item['Model']} | {item['Platform']}"

            send_images = (model_id in vision_capable) and (data_urls_master or http_urls_master)
            if (not (model_id in vision_capable)) and (data_urls_master or http_urls_master):
                st.info(f"ℹ️ {label_for_ui} pode não suportar visão. Somente o texto será enviado.", icon="ℹ️")

            if send_images:
                content_blocks = _build_mm_content(question, data_urls_master, http_urls_master)
                messages = [HumanMessage(content=content_blocks)]
                payload_for_log = f"{question}\n\n[Attached images: {len(data_urls_master)+len(http_urls_master)}]"
            else:
                messages = [HumanMessage(content=question)]
                payload_for_log = question

            with st.status(f"Querying the LLM ({label_for_ui})...", expanded=False):
                started = datetime.now()
                try:
                    llm = ChatOpenAI(model=model_id, max_retries=0, timeout=60)
                    response = llm.invoke(messages)
                    content = getattr(response, "content", str(response))
                    st.session_state["last_error"] = ""; ok = True
                except Exception as e:
                    response = None
                    content = f"[ERROR] {''.join(traceback.format_exception_only(type(e), e)).strip()}"
                    st.session_state["last_error"] = content; ok = False
                finished = datetime.now()
                duration_s = (finished - started).total_seconds()

            st.session_state["current_output"] = content
            sep = "\n" + ("-" * 80) + "\n"
            block = (
                f"Timestamp: {started.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Model: {label_for_ui}\n"
                f"Prompt:\n{payload_for_log}\n\n"
                f"Response:\n{content}\n"
                f"Duration: {duration_s:.2f}s"
            )
            st.session_state["exec_log"] += (sep if st.session_state["exec_log"] else "") + block

            in_tok = out_tok = tot_tok = None
            if response is not None:
                in_tok, out_tok, tot_tok = _extract_token_usage(response)

            seq = len(st.session_state["usage_rows"]) + 1
            st.session_state["usage_rows"].append({
                "No.": seq,
                "Model": label_for_ui,
                "Input tokens": in_tok if in_tok is not None else "-",
                "Output tokens": out_tok if out_tok is not None else "-",
                "Total tokens": (tot_tok if tot_tok is not None else (
                    (in_tok or 0) + (out_tok or 0) if (in_tok is not None and out_tok is not None) else "-"
                )),
                "Time (s)": f"{duration_s:.2f}",
            })

            if ok:
                st.success(f"✅ Execution finished — result received from {label_for_ui}."); st.toast(f"Result received ({label_for_ui})", icon="✅")
            else:
                st.error(f"An error occurred with {label_for_ui}. See details in the footer."); st.toast(f"Execution error ({label_for_ui})", icon="⚠️")

        st.session_state.is_running = False
        dl_txt_placeholder.download_button(
            label="⬇️ Download Log (.txt)",
            data=st.session_state.exec_log,
            file_name=f"llm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            disabled=False,
        )

# ================= LLM Response (blue) =================
st.markdown('<h3 class="blue-section">LLM Response</h3>', unsafe_allow_html=True)
st.text_area("Output", key="current_output", height=220, help="Only the latest run's response.")

# ================= Token Usage =================
st.markdown('<h3 class="blue-section">Token Usage:</h3>', unsafe_allow_html=True)
df_usage = pd.DataFrame(st.session_state["usage_rows"])

if df_usage.empty:
    st.info("No records yet")
else:
    col_usage, col_gap = st.columns([0.5, 0.5])
    VISIBLE_ROWS_USAGE = 5
    ROW_PX_USAGE, HEADER_PX_USAGE, PADDING_PX_USAGE = 36, 40, 28
    usage_editor_height = HEADER_PX_USAGE + VISIBLE_ROWS_USAGE * ROW_PX_USAGE + PADDING_PX_USAGE  # ~248px

    with col_usage:
        col_config = {
            "No.": st.column_config.NumberColumn("No.", help="Sequence number", width="small", format="%d"),
            "Model": st.column_config.TextColumn("Model", help="Model | Platform", width="large"),
            "Input tokens": st.column_config.NumberColumn("Input tokens", help="Prompt tokens", format="%d"),
            "Output tokens": st.column_config.NumberColumn("Output tokens", help="Completion tokens", format="%d"),
            "Total tokens": st.column_config.NumberColumn("Total tokens", help="Sum of input + output", format="%d"),
            "Time (s)": st.column_config.NumberColumn("Time (s)", help="Execution time in seconds", format="%.2f"),
        }
        column_order = ["No.", "Model", "Input tokens", "Output tokens", "Total tokens", "Time (s)"]
        st.data_editor(
            df_usage,
            hide_index=True,
            use_container_width=True,
            height=usage_editor_height,
            column_config=col_config,
            column_order=column_order,
            disabled=column_order,
            key="usage_editor",
        )

# ================= Execution Log =================
st.subheader("Execution Log")
st.text_area(
    "History",
    key="exec_log",
    height=340,
    help="Each run is appended at the end, separated by dashed lines.",
    disabled=True,
)

# ================= Log PDF (inclui imagens com cabeçalho + rodapé numerado) =================
st.markdown("### Export Log as PDF")

def make_pdf_from_text(
    text: str,
    logo_b64: str | None,
    image_data_urls: List[str] | None = None,
    image_http_urls: List[str] | None = None,
    title: str = PDF_TITLE,
    title_font_size: int = 20
) -> bytes:
    """
    Gera um PDF com cabeçalho (logo + título) em TODAS as páginas, inclui as imagens
    carregadas (uploads) na mesma ordem (até 3 por linha) e adiciona:
      - Cabeçalho ACIMA das imagens: "This document presents an analysis of the images below:"
      - Rodapé abaixo de cada imagem: "Image 01", "Image 02", ...
    Depois, inclui o texto do log.
    URLs HTTP (se houver) são impressas como texto (não há download).
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
        c.setFillColorRGB(0.752, 0.0, 0.0)  # cor do título no PDF
        title_y = y_top - (logo_height_mm * mm * 0.65)
        c.drawCentredString(width / 2, title_y, title)

        c.setLineWidth(1)
        c.setStrokeColorRGB(0.79, 0.84, 0.91)
        line_y = y_top - (logo_height_mm * mm) - (header_line_gap * mm)
        c.line(left_margin, line_y, width - right_margin, line_y)
        return line_y - (header_gap_mm * mm)

    # Cabeçalho do documento
    text_y = draw_header()
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0, 0, 0)

    # --- Seção de imagens ---
    cursor_y = text_y - 10  # espaço inicial

    image_data_urls = image_data_urls or []
    image_http_urls = image_http_urls or []

    # Se houver imagens, escrever o cabeçalho acima delas (solicitado)
    if image_data_urls:
        c.setFont("Helvetica-Bold", 11)
        c.setFillColorRGB(0.05, 0.39, 0.85)  # azul suave
        header_text = "🔎 This document presents an analysis of the images below:"
        c.drawString(left_margin, cursor_y, header_text)
        cursor_y -= 12
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0, 0, 0)

    # Parâmetros de layout das imagens
    max_img_height_mm = 50          # altura máxima do retângulo do slot da imagem
    gap_between_imgs_mm = 4
    caption_gap_mm = 2              # espaço entre a imagem e o texto "Image NN"
    caption_height_px = 10          # altura de uma linha de legenda (px lógicos do canvas)

    from reportlab.lib.units import inch
    avail_width = width - left_margin - right_margin
    cols = 3
    gap_px = gap_between_imgs_mm * mm
    img_w = (avail_width - (cols - 1) * gap_px) / cols
    max_h = max_img_height_mm * mm

    def draw_data_image_with_caption(data_url: str, col_idx: int, row_idx: int, img_index: int):
        """
        Desenha a imagem dentro do slot (col_idx,row_idx) e a legenda "Image NN" logo abaixo.
        Retorna a altura efetiva ocupada pela célula (max_h + alturas extras da legenda).
        """
        nonlocal cursor_y
        x = left_margin + col_idx * (img_w + gap_px)
        y_for_row_top = cursor_y - row_idx * (max_h + gap_px + caption_gap_mm * mm + caption_height_px)

        # Nova página se não couber a célula inteira
        if (y_for_row_top - max_h - caption_gap_mm * mm - caption_height_px) < bottom_margin:
            c.showPage()
            # Redesenha cabeçalho
            new_text_y = draw_header()
            c.setFont("Helvetica", 10)
            c.setFillColorRGB(0, 0, 0)
            # reposiciona o cursor e recomputa y_for_row_top para a primeira linha da nova página
            cursor_y = new_text_y - 10
            # Reimprime o cabeçalho de seção de imagens, pois ainda estamos na seção de imagens
            c.setFont("Helvetica-Bold", 11)
            c.setFillColorRGB(0.05, 0.39, 0.85)
            c.drawString(left_margin, cursor_y, "🔎 This document presents an analysis of the images below:")
            cursor_y -= 12
            c.setFont("Helvetica", 10)
            c.setFillColorRGB(0, 0, 0)
            # recomputa para a nova página
            y_for_row_top = cursor_y - row_idx * (max_h + gap_px + caption_gap_mm * mm + caption_height_px)

        # Desenha a imagem (data-url)
        try:
            b64part = data_url.split(",", 1)[1] if "," in data_url else data_url
            img_bytes = base64.b64decode(b64part)
            from reportlab.lib.utils import ImageReader
            img = ImageReader(io.BytesIO(img_bytes))
            iw, ih = img.getSize()
            aspect = iw / ih if ih else 1.0
            if (img_w / aspect) <= max_h:
                w = img_w
                h = img_w / aspect
            else:
                h = max_h
                w = max_h * aspect
            c.drawImage(img, x, y_for_row_top - h, width=w, height=h, mask='auto')
            # Centraliza a legenda sob a base da imagem
            caption_y = (y_for_row_top - h) - (caption_gap_mm * mm)
            caption_text = f"Image {img_index:02d}"
            c.setFont("Helvetica", 9)
            c.setFillColorRGB(0.26, 0.26, 0.26)
            c.drawCentredString(x + w/2, caption_y - caption_height_px/2, caption_text)
            # Volta à fonte padrão
            c.setFont("Helvetica", 10)
            c.setFillColorRGB(0, 0, 0)
        except Exception:
            # Se falhar, só imprime legenda no slot
            caption_y = (y_for_row_top - max_h) - (caption_gap_mm * mm)
            caption_text = f"Image {img_index:02d}"
            c.setFont("Helvetica", 9)
            c.setFillColorRGB(0.26, 0.26, 0.26)
            c.drawCentredString(x + img_w/2, caption_y - caption_height_px/2, caption_text)
            c.setFont("Helvetica", 10)
            c.setFillColorRGB(0, 0, 0)

    # Desenha imagens com legendas numeradas
    if image_data_urls:
        for i, du in enumerate(image_data_urls):
            col_idx = i % cols
            row_idx = i // cols
            draw_data_image_with_caption(du, col_idx, row_idx, i + 1)
        # Ajusta cursor_y ao final da última linha de imagens
        last_row = (len(image_data_urls) - 1) // cols
        cursor_y = cursor_y - (last_row + 1) * (max_h + gap_px + caption_gap_mm * mm + caption_height_px) + (gap_px / 2)

    # URLs HTTP impressas como texto
    if image_http_urls:
        if image_data_urls:
            cursor_y -= 12
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0, 0, 0)
        for url in image_http_urls:
            if cursor_y - 12 < bottom_margin:
                c.showPage()
                cursor_y = draw_header()
                c.setFont("Helvetica", 10); c.setFillColorRGB(0, 0, 0)
                # como voltamos do showPage, reabre seção imagens (texto)
                c.setFont("Helvetica-Bold", 11)
                c.setFillColorRGB(0.05, 0.39, 0.85)
                c.drawString(left_margin, cursor_y - 10, "🔎 This document presents an analysis of the images below:")
                cursor_y -= 22
                c.setFont("Helvetica-Oblique", 9); c.setFillColorRGB(0, 0, 0)
            c.drawString(left_margin, cursor_y, f"Image URL: {url}")
            cursor_y -= 12
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0, 0, 0)

    # gap entre seção de imagens e texto do log
    cursor_y -= 8

    # --- Texto do log ---
    if cursor_y - 12 < bottom_margin:
        c.showPage()
        cursor_y = draw_header()
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0, 0, 0)

    max_chars_per_line = 110
    line_height = 12

    for raw_line in text.splitlines():
        if raw_line == "":
            if cursor_y - line_height < bottom_margin:
                c.showPage()
                cursor_y = draw_header()
                c.setFont("Helvetica", 10)
                c.setFillColorRGB(0, 0, 0)
            cursor_y -= line_height
            continue
        line = raw_line
        while len(line) > max_chars_per_line:
            part = line[:max_chars_per_line]
            if cursor_y - line_height < bottom_margin:
                c.showPage()
                cursor_y = draw_header()
                c.setFont("Helvetica", 10)
                c.setFillColorRGB(0, 0, 0)
            c.drawString(left_margin, cursor_y, part)
            cursor_y -= line_height
            line = line[max_chars_per_line:]
        if cursor_y - line_height < bottom_margin:
            c.showPage()
            cursor_y = draw_header()
            c.setFont("Helvetica", 10)
            c.setFillColorRGB(0, 0, 0)
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
    st.session_state["log_pdf_bytes"] = make_pdf_from_text(
        st.session_state["exec_log"],
        logo_b64,
        image_data_urls=st.session_state.get("image_data_urls", []),
        image_http_urls=st.session_state.get("image_http_urls", []),
        title=PDF_TITLE,
        title_font_size=pdf_title_font_size
    )
    st.toast(f"PDF generated (title font size = {pdf_title_font_size}pt).", icon="📄")

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
                for (let i = 0; i < byteChars.length; i++) {{ byteNumbers[i] = byteChars.charCodeAt(i); }}
                const blob = new Blob([byteNumbers], {{ type: "application/pdf" }});
                const url = URL.createObjectURL(blob);
                window.open(url, "_blank");
            }})();
            </script>
        """, height=0)

# ================= Errors footer =================
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

# ===================== OpenAI Footer =====================
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
            © 2025 Prompt Engineer Project — OpenAI Usage Dashboard<br>
            Powered by Romilson Lemes — Using OpenAI API
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
