"""
Shared Streamlit theme for market notes.

Usage:
    from lib.ui.theme import apply_theme, render_header, render_takeaway, ...

    apply_theme()                          # call once at top of app
    apply_theme(show_sidebar=True)         # keep sidebar visible (for interactive notes)
    render_header("Title", "KICKER", "Subtitle text")
    render_takeaway("Key insight here.")
"""
from __future__ import annotations

import streamlit as st

# ── Fonts & colour tokens ────────────────────────────────────
_FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');"

_CSS_VARS = """
:root {
    --bg: #f4efe7;
    --card: rgba(255, 251, 245, 0.92);
    --ink: #14213d;
    --muted: #5c677d;
    --accent: #e76f51;
    --accent-2: #2a9d8f;
}
"""

_APP_BG = """
.stApp {
    background:
        radial-gradient(circle at top left, rgba(233, 196, 106, 0.28), transparent 28%),
        radial-gradient(circle at 80% 10%, rgba(42, 157, 143, 0.18), transparent 24%),
        linear-gradient(180deg, #fbf7f1 0%, #f1ece3 100%);
    color: var(--ink);
}
"""

_LAYOUT = """
.block-container,
.main .block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 980px;
    margin-left: auto;
    margin-right: auto;
    padding-top: 2rem;
    padding-bottom: 0.5rem;
    padding-left: 2.25rem;
    padding-right: 2.25rem;
}
"""

_TYPOGRAPHY = """
h1, h2, h3 {
    font-family: 'Source Serif 4', serif;
    color: var(--ink);
    letter-spacing: -0.02em;
}
body, .stMarkdown, .stMetric, .stDataFrame {
    font-family: 'IBM Plex Sans', sans-serif;
}
"""

_COMPONENTS = """
.hero-card {
    padding: 1.4rem 0 0.8rem 0;
    margin-bottom: 0.8rem;
}
.hero-kicker {
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.72rem;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.hero-dek {
    font-size: 1.15rem;
    line-height: 1.65;
    color: var(--ink);
    max-width: 52rem;
}
.standfirst {
    font-size: 1.04rem;
    line-height: 1.72;
    color: var(--ink);
    max-width: 54rem;
    margin: 0.2rem 0 1rem 0;
}
.chart-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.25rem;
    line-height: 1.35;
    color: var(--ink);
    margin: 1.1rem 0 0.55rem 0;
}
.chart-caption {
    color: var(--muted);
    font-size: 0.93rem;
    line-height: 1.55;
    margin: 0.75rem 0 0 0;
}
.takeaway-box {
    margin: 0.8rem 0 2rem 0;
    padding: 0.9rem 1rem;
    background: rgba(255, 251, 245, 0.88);
    border-left: 4px solid var(--accent);
    border-radius: 0.5rem;
}
.takeaway-label {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 0.32rem;
}
.takeaway-text {
    color: var(--ink);
    line-height: 1.55;
    font-weight: 600;
}
.small-note {
    color: var(--muted);
    font-size: 0.84rem;
    line-height: 1.45;
    margin: 0.45rem 0 0 0;
}
.annotation-box {
    margin: 0.95rem 0 2rem 0;
    padding: 0.95rem 1rem;
    background: rgba(255, 251, 245, 0.92);
    border: 1px solid rgba(20, 33, 61, 0.08);
    border-radius: 0.6rem;
}
.annotation-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.06rem;
    line-height: 1.35;
    color: var(--ink);
    margin: 0 0 0.35rem 0;
}
.annotation-text {
    color: var(--ink);
    line-height: 1.55;
}
.closing-line {
    font-family: 'Source Serif 4', serif;
    font-size: 1.25rem;
    line-height: 1.45;
    margin: 1rem 0 0.75rem 0;
}
"""

_SIDEBAR_COMPACT = """
[data-testid="stSidebar"] { min-width: 280px; max-width: 280px; }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
[data-testid="stSidebar"] hr { margin: 0.3rem 0; }
[data-testid="stSidebar"] h2 { font-size: 0.9rem; margin-bottom: 0.2rem; }
[data-testid="stSidebar"] .stSlider { margin-bottom: -0.5rem; }
[data-testid="stSidebar"] .stRadio { margin-bottom: -0.3rem; }
"""

_SIDEBAR_HIDDEN = """
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
    display: none;
}
"""


def apply_theme(*, show_sidebar: bool = False) -> None:
    """Inject shared CSS. Call once at the top of app, after st.set_page_config()."""
    sidebar_css = _SIDEBAR_COMPACT if show_sidebar else _SIDEBAR_HIDDEN
    st.markdown(
        f"<style>{_FONT_IMPORT}{_CSS_VARS}{_APP_BG}{_LAYOUT}{_TYPOGRAPHY}{_COMPONENTS}{sidebar_css}</style>",
        unsafe_allow_html=True,
    )


# ── Render helpers ───────────────────────────────────────────

def render_header(title: str, kicker: str = "", subtitle: str = "") -> None:
    parts = []
    if kicker:
        parts.append(f'<div class="hero-kicker">{kicker}</div>')
    parts.append(f"<h1>{title}</h1>")
    if subtitle:
        parts.append(f'<div class="hero-dek">{subtitle}</div>')
    st.markdown(f'<div class="hero-card">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_standfirst(text: str) -> None:
    st.markdown(f'<div class="standfirst">{text}</div>', unsafe_allow_html=True)


def render_takeaway(text: str) -> None:
    st.markdown(
        f'<div class="takeaway-box"><div class="takeaway-label">Takeaway</div>'
        f'<div class="takeaway-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_chart_title(text: str) -> None:
    st.markdown(f'<div class="chart-title">{text}</div>', unsafe_allow_html=True)


def render_chart_caption(text: str) -> None:
    st.markdown(f'<div class="chart-caption">{text}</div>', unsafe_allow_html=True)


def render_annotation(title: str, text: str) -> None:
    st.markdown(
        f'<div class="annotation-box"><div class="annotation-title">{title}</div>'
        f'<div class="annotation-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_footer_note(text: str) -> None:
    st.markdown(f'<div class="small-note">{text}</div>', unsafe_allow_html=True)


def render_closing(text: str) -> None:
    st.markdown(f'<div class="closing-line">{text}</div>', unsafe_allow_html=True)


def render_footer(author: str = "Anton Telegin", year: int = 2026) -> None:
    st.markdown(
        f'<div class="small-note" style="margin-top: 2rem; text-align: center;">'
        f'&copy; {year} {author}. Model &amp; data: open source. '
        f'This is not investment advice.</div>',
        unsafe_allow_html=True,
    )
