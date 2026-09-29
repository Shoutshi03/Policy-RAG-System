from __future__ import annotations

import uuid
from dataclasses import asdict
from html import escape

import streamlit as st
from dotenv import load_dotenv

from policyrag.demo import demo_documents
from policyrag.generation import llm_ready, model_name
from policyrag.ingestion import MAX_FILE_BYTES, extract_pdf
from policyrag.models import AssistantResult, Chunk
from policyrag.qdrant_store import get_qdrant_index, try_upsert
from policyrag.service import answer_question

load_dotenv()
st.set_page_config(
    page_title="PolicyRAG · Insight Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#1f302b; --muted:#70817a; --paper:#f5f7f4; --green:#164e41; --lime:#d5e86c; --line:#e5eae5; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
    [data-testid="stAppViewContainer"] { background:var(--paper); }
    [data-testid="stHeader"] { background:rgba(245,247,244,.85); }
    [data-testid="stSidebar"] { background:#edf1ec; border-right:1px solid #e0e7e1; }
    [data-testid="stSidebarContent"] { padding-top:1.1rem; }
    .block-container { max-width:1580px; padding-top:1.7rem; padding-bottom:3rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:#1b3028; }
    h1 { font-size:2.25rem !important; font-weight:800 !important; }
    h2 { font-size:1.35rem !important; font-weight:750 !important; }
    h3 { font-size:1.04rem !important; font-weight:700 !important; }
    .brand { display:flex; align-items:center; gap:.72rem; padding:.2rem .2rem 1.15rem; }
    .brand-mark { width:42px; height:42px; border-radius:14px; display:flex; align-items:center; justify-content:center; background:#164e41; color:#e4f078; font-size:21px; font-weight:800; box-shadow:0 5px 16px #164e4120; }
    .brand-title { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.05rem; letter-spacing:-.04em; line-height:1.1; }
    .brand-sub { color:#7c8b83; font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; margin-top:4px; }
    .eyebrow { text-transform:uppercase; letter-spacing:.14em; color:#638178; font-weight:700; font-size:.68rem; }
    .hero-title { font-size:2.4rem; font-family:'Manrope',sans-serif; font-weight:800; letter-spacing:-.06em; line-height:1.05; margin:.35rem 0 .6rem; color:#1b3028; }
    .hero-copy { color:#728078; font-size:.96rem; max-width:660px; line-height:1.55; }
    .demo-pill { display:inline-flex; align-items:center; gap:.4rem; border:1px solid #dfe8b2; background:#f2f6d7; color:#52621a; border-radius:999px; padding:.34rem .7rem; font-size:.72rem; font-weight:700; }
    .metric-card { background:#fff; border:1px solid #e6ece6; border-radius:16px; padding:16px 18px; min-height:92px; box-shadow:0 5px 18px rgba(37,63,49,.035); }
    .metric-label { color:#87948e; font-size:.68rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
    .metric-value { font-family:'Manrope',sans-serif; color:#193b30; font-size:1.6rem; line-height:1.1; font-weight:800; margin-top:.35rem; }
    .metric-note { color:#97a29c; font-size:.72rem; margin-top:.2rem; }
    .section-card { background:#fff; border:1px solid #e5eae5; border-radius:18px; padding:18px 20px; box-shadow:0 6px 24px rgba(40,64,49,.035); }
    .source-card { border:1px solid #e5ebe6; background:#fbfcfa; border-radius:13px; padding:13px; margin:.65rem 0; }
    .source-tag { display:inline-block; color:#164e41; background:#e6f2eb; border-radius:6px; padding:2px 6px; font-size:.68rem; font-weight:800; margin-right:5px; }
    .source-meta { color:#819087; font-size:.72rem; margin-top:.38rem; }
    .doc-row { background:rgba(255,255,255,.7); border:1px solid #e1e8e2; border-radius:12px; padding:10px 11px; margin:.45rem 0; }
    .doc-title { font-size:.78rem; font-weight:700; line-height:1.35; color:#354840; overflow-wrap:anywhere; }
    .doc-meta { color:#85928b; font-size:.68rem; margin-top:4px; }
    .soft-note { color:#7e8b84; font-size:.74rem; line-height:1.5; }
    div.stButton > button { border-radius:10px; min-height:2.45rem; font-weight:650; border-color:#dfe6df; transition:all .16s cubic-bezier(.23,1,.32,1); }
    div.stButton > button:active { transform:scale(.98); }
    div.stButton > button[kind="primary"] { background:#164e41; border-color:#164e41; color:#fff; }
    div.stButton > button[kind="primary"]:hover { background:#216451; border-color:#216451; }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div { border-radius:10px; }
    [data-testid="stChatMessage"] { background:#fff; border:1px solid #e7ece7; border-radius:16px; padding:.85rem 1rem; margin-bottom:.8rem; box-shadow:0 3px 14px rgba(37,63,49,.025); }
    [data-testid="stChatInput"] { border-radius:14px; border-color:#dce5de; background:#fff; }
    [data-testid="stChatInput"] textarea { font-size:.92rem; }
    .stAlert { border-radius:12px; }
    hr { border-color:#e8ece8; }
    .footer { color:#96a29b; font-size:.69rem; padding:1.2rem 0 .4rem; }
    @media(max-width:800px) { .hero-title{font-size:1.85rem}.block-container{padding-left:1rem;padding-right:1rem} }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    if "workspace_id" not in st.session_state:
        st.session_state.workspace_id = str(uuid.uuid4())
    if "documents" not in st.session_state:
        st.session_state.documents = demo_documents()
        try_upsert(st.session_state.workspace_id, [chunk for doc in st.session_state.documents for chunk in doc.chunks])
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_sources" not in st.session_state:
        st.session_state.active_sources = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    if "indexed_hashes" not in st.session_state:
        st.session_state.indexed_hashes = set()


_init_state()

MODE_LABELS = {
    "qa": "Question · Réponse",
    "summary": "Résumé",
    "explain": "Explication",
    "compare": "Comparaison",
    "extract": "Extraction",
    "translate": "Traduction",
    "insights": "Insights",
}
LANGUAGES = {"Français": "fr", "English": "en", "العربية": "ar"}


def _all_chunks() -> list[Chunk]:
    return [chunk for document in st.session_state.documents for chunk in document.chunks]


def _process_uploads(files, ocr_enabled: bool, ocr_language: str) -> None:
    progress = st.progress(0, text="Préparation des documents…")
    status = st.empty()
    total = len(files)
    added = 0
    for file_index, upload in enumerate(files):
        raw = upload.getvalue()
        if len(raw) > MAX_FILE_BYTES:
            st.error(f"{upload.name} dépasse la limite de 20 Mo.")
            continue
        digest = __import__("hashlib").sha256(raw).hexdigest()
        if digest in st.session_state.indexed_hashes or any(doc.file_hash == digest for doc in st.session_state.documents):
            st.info(f"{upload.name} est déjà indexé dans cet espace de travail.")
            continue
        try:
            def update_page(current: int, total_pages: int, message: str) -> None:
                fraction = (file_index + (current / max(total_pages, 1))) / max(total, 1)
                progress.progress(min(fraction, 0.99), text=f"{upload.name} · {message}")
                status.caption(message)

            document = extract_pdf(raw, upload.name, enable_ocr=ocr_enabled, ocr_languages=ocr_language, progress=update_page)
            st.session_state.documents.append(document)
            st.session_state.indexed_hashes.add(digest)
            index_status = try_upsert(st.session_state.workspace_id, document.chunks)
            if index_status.startswith("local-fallback"):
                st.warning("Qdrant est temporairement indisponible ; l’index local reste actif.")
            added += 1
            ocr_note = f" · {document.ocr_pages} page(s) OCR" if document.ocr_pages else ""
            st.success(f"{document.file_name} indexé · {document.pages} pages · {len(document.chunks)} passages{ocr_note}")
        except (ValueError, RuntimeError) as error:
            st.error(f"{upload.name} : {error}")
        except Exception as error:
            st.error(f"{upload.name} : erreur d’ingestion ({type(error).__name__}).")
    progress.progress(1.0, text=f"Ingestion terminée · {added} nouveau(x) document(s)")


def _send_question(question: str, mode: str, language: str) -> None:
    prior = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[-6:]
        if item.get("role") in {"user", "assistant"}
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Recherche des passages pertinents et vérification des sources…"):
        result: AssistantResult = answer_question(
            question=question,
            chunks=_all_chunks(),
            mode=mode,  # type: ignore[arg-type]
            language=language,
            history=prior,
            workspace_id=st.session_state.workspace_id,
        )
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "abstained": result.abstained,
        "confidence": result.confidence,
        "model": result.model,
        "warning": result.warning,
        "citations": [asdict(item) for item in result.citations],
    })
    st.session_state.active_sources = [asdict(item) for item in result.citations]


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-title">PolicyRAG</div><div class="brand-sub">Insight assistant</div></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("UNDP · Policy knowledge workspace")
    st.markdown("---")
    st.markdown("#### Espace documentaire")
    uploaded_files = st.file_uploader(
        "Ajouter des rapports PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDF texte ou numérisés · jusqu’à 20 Mo par fichier et 500 pages.",
    )
    ocr_language_label = st.selectbox("Langues OCR", ["Français + English", "English", "العربية + English"], index=0)
    ocr_languages = {"Français + English": "fra+eng", "English": "eng", "العربية + English": "ara+eng"}[ocr_language_label]
    ocr_enabled = st.toggle("OCR de secours pour les pages scannées", value=True)
    if st.button("Indexer les PDF sélectionnés", type="primary", use_container_width=True, disabled=not uploaded_files):
        _process_uploads(uploaded_files or [], ocr_enabled, ocr_languages)
    st.markdown("---")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Documents actifs")
    with right:
        st.caption(f"{len(st.session_state.documents)}")
    if st.button("Charger le corpus exemple", use_container_width=True):
        known = {doc.file_hash for doc in st.session_state.documents}
        additions = [doc for doc in demo_documents() if doc.file_hash not in known]
        st.session_state.documents.extend(additions)
        try_upsert(st.session_state.workspace_id, [chunk for doc in additions for chunk in doc.chunks])
        st.toast(f"{len(additions)} document(s) de démonstration ajouté(s)." if additions else "Le corpus exemple est déjà chargé.")
        st.rerun()
    if st.session_state.documents:
        for document in st.session_state.documents:
            demo_badge = " · démo fictive" if document.is_demo else ""
            st.markdown(
                f'<div class="doc-row"><div class="doc-title">{escape(document.file_name)}</div><div class="doc-meta">{document.pages} p. · {len(document.chunks)} passages{escape(demo_badge)}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("Retirer", key=f"remove-{document.document_id}", use_container_width=True):
                try:
                    index = get_qdrant_index()
                    if index:
                        index.delete_document(st.session_state.workspace_id, document.document_id)
                except Exception:
                    pass
                st.session_state.documents = [item for item in st.session_state.documents if item.document_id != document.document_id]
                st.session_state.messages = []
                st.session_state.active_sources = []
                st.rerun()
        if st.button("Vider l’espace de travail", use_container_width=True):
            try:
                index = get_qdrant_index()
                if index:
                    index.delete_workspace(st.session_state.workspace_id)
            except Exception:
                pass
            st.session_state.documents = []
            st.session_state.messages = []
            st.session_state.active_sources = []
            st.rerun()
    else:
        st.info("Aucun rapport actif. Chargez le corpus exemple ou déposez un PDF.")
    st.markdown("---")
    st.markdown(f"<div class='soft-note'>Modèle actif : <b>{escape(model_name())}</b><br/>La clé API est configurée côté serveur, jamais dans l’interface.</div>", unsafe_allow_html=True)
    mode_label = st.selectbox("Type d’analyse", list(MODE_LABELS.values()), index=0)
    mode = {label: key for key, label in MODE_LABELS.items()}[mode_label]
    response_language_label = st.selectbox("Langue de réponse", list(LANGUAGES.keys()), index=0)
    response_language = LANGUAGES[response_language_label]
    st.markdown("---")
    if st.button("Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_sources = []
        st.rerun()
    if llm_ready():
        st.markdown(f'<div class="soft-note">Génération · <b>{model_name()}</b><br/>Appels LLM via OpenRouter</div>', unsafe_allow_html=True)
    else:
        st.warning("Aucun secret OpenRouter configuré côté serveur : le retrieval, les citations et les réponses extractives restent actifs.")
    st.markdown('<div class="footer">Documents de démonstration marqués comme fictifs.<br/>Les PDF importés restent dans la session en mémoire.</div>', unsafe_allow_html=True)

# Header and status ribbon
hero_left, hero_right = st.columns([0.77, 0.23], vertical_alignment="center")
with hero_left:
    st.markdown('<div class="eyebrow">Knowledge management · Evidence first</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Policy insights,<br/>grounded in evidence.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Interrogez vos rapports de politique publique, comparez les recommandations et retrouvez chaque passage source — en français, English ou العربية.</div>', unsafe_allow_html=True)
with hero_right:
    st.markdown('<div style="text-align:right;padding-top:1.2rem"><span class="demo-pill">● &nbsp;DÉMO INTERACTIVE</span></div>', unsafe_allow_html=True)

st.write("")
num_docs = len(st.session_state.documents)
num_chunks = len(_all_chunks())
num_pages = sum(document.pages for document in st.session_state.documents)
metric_cols = st.columns(4, gap="medium")
for column, label, value, note in zip(
    metric_cols,
    ("Rapports actifs", "Passages indexés", "Pages analysées", "Langues disponibles"),
    (num_docs, num_chunks, num_pages, "FR · EN · AR"),
    ("PDF & corpus de démo", "BM25 + vecteurs n-grammes", "extraction paginée + OCR", "recherche multilingue"),
):
    with column:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

st.write("")
chat_column, sources_column = st.columns([0.68, 0.32], gap="large")
with chat_column:
    st.markdown('<div class="eyebrow">Assistant de recherche</div>', unsafe_allow_html=True)
    st.markdown("### Posez une question aux documents")
    st.caption("Les réponses sont limitées aux passages indexés ; une abstention est renvoyée lorsque les preuves sont insuffisantes.")
    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("**Commencez par une question, ou essayez un exemple :**")
            suggestions = [
                "Quelle cible de réduction est proposée pour les interruptions de services ?",
                "Compare les garanties de protection sociale et d’emploi des jeunes.",
                "Quels risques et indicateurs sont cités dans les rapports ?",
            ]
            suggestion_cols = st.columns(3)
            for column, suggestion in zip(suggestion_cols, suggestions):
                with column:
                    if st.button(suggestion, key=f"suggest-{suggestion}", use_container_width=True):
                        st.session_state.pending_prompt = suggestion
                        st.rerun()

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                if item.get("warning"):
                    st.caption(item["warning"])
                if item.get("citations"):
                    citation_text = " · ".join(f"[{cite['source_id']}] p. {cite['page']}" for cite in item["citations"][:4])
                    st.caption(f"Sources citées : {citation_text} · confiance retrieval indicative {item.get('confidence', 0)} %")
                elif item.get("abstained"):
                    st.caption("Abstention : aucun passage n’a franchi le seuil de preuve.")

    prompt_from_chat = st.chat_input("Posez une question sur les rapports actifs…")
    prompt = prompt_from_chat or st.session_state.pending_prompt
    if prompt:
        st.session_state.pending_prompt = ""
        _send_question(prompt, mode, response_language)
        st.rerun()

with sources_column:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Traçabilité</div>', unsafe_allow_html=True)
        st.markdown("### Passages sources")
        st.caption("Les références proviennent du retrieval ; document, page et section sont conservés à l’indexation.")
        sources = st.session_state.active_sources
        if sources:
            for citation in sources:
                label = " · fictif" if citation.get("is_demo") else ""
                st.markdown(
                    f'<div class="source-card"><span class="source-tag">[{escape(citation["source_id"])}]</span><b>{escape(citation["document"])}</b>'
                    f'<div class="source-meta">Page {int(citation["page"])} · {escape(citation["section"])}{escape(label)}</div>'
                    f'<div style="font-size:.78rem;line-height:1.5;margin-top:.5rem;color:#52635a">{escape(citation["excerpt"])}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Les sources de la dernière réponse apparaîtront ici.")
        st.markdown("---")
        st.markdown("**Pipeline actif**")
        st.markdown("`PDF → texte/OCR → chunks paginés → BM25 + n-grammes → reranking → LLM → citations`")
        st.caption("Les vecteurs n-grammes sont une base légère de démonstration, pas des embeddings neuronaux BGE-M3.")

st.markdown('<div class="footer">UNDP PolicyRAG & Insight Assistant · Prototype de démonstration · Les documents exemples sont synthétiques, pas des publications officielles.</div>', unsafe_allow_html=True)
