
import streamlit as st
import pandas as pd
import html

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="SchemeSathi AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# RGB / GRADIENT UI DESIGN
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #090414;
    background-image:
        radial-gradient(ellipse at 8% 8%, rgba(147, 51, 234, .28), transparent 32%),
        radial-gradient(ellipse at 92% 14%, rgba(34, 211, 238, .13), transparent 28%),
        radial-gradient(ellipse at 80% 82%, rgba(217, 70, 239, .12), transparent 30%),
        radial-gradient(ellipse at 12% 88%, rgba(79, 70, 229, .18), transparent 32%),
        linear-gradient(135deg, #080311 0%, #16072f 38%, #100821 68%, #070b1d 100%);
    background-attachment: fixed;
    background-size: 180% 180%;
    animation: rgbBackgroundShift 24s ease infinite;
    color: #F4F4FF;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(10, 10, 28, 0.35);
}

/* RGB glow accents and smoother dashboard styling */
[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(160deg, rgb(28, 17, 65), rgb(13, 25, 55) 55%, rgb(9, 16, 38));
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.75rem;
}

.stMarkdown, label, p, li {
    text-shadow: 0 1px 8px rgba(0, 0, 0, 0.12);
}

[data-testid="stForm"] {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.12), rgba(8, 145, 178, 0.10));
    border: 1px solid rgba(103, 232, 249, 0.30);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.16), rgba(8, 145, 178, 0.10));
    border: 1px solid rgba(167, 139, 250, 0.30);
    border-radius: 14px;
    padding: 14px;
}

[data-testid="stExpander"] {
    background: rgba(25, 23, 55, 0.70);
    border: 1px solid rgba(167, 139, 250, 0.28);
    border-radius: 12px;
}

.stAlert {
    border-radius: 12px;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSlider"] [data-baseweb="slider"] {
    accent-color: #A78BFA;
}

.stLinkButton a, a {
    transition: all 0.2s ease;
}

a:hover {
    color: #F0ABFC !important;
}

@media (max-width: 768px) {
    .hero { padding: 20px; }
    .hero h2 { font-size: 21px; }
    .scheme-card { padding: 17px; }
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgb(25, 17, 58),
        rgb(15, 22, 48),
        rgb(10, 15, 35)
    );
    border-right: 1px solid rgba(167, 139, 250, 0.30);
}

[data-testid="stSidebar"] * {
    color: #F0EDFF;
}

.main-title {
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;
    letter-spacing: -1.5px;
    background: linear-gradient(
        90deg,
        rgb(196, 181, 253),
        rgb(103, 232, 249),
        rgb(244, 114, 182)
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.subtitle {
    color: #B9B8D9;
    font-size: 15px;
    margin-bottom: 25px;
}

.hero {
    padding: 27px;
    border-radius: 22px;
    border: 1px solid rgba(167, 139, 250, 0.36);
    background: linear-gradient(
        120deg,
        rgba(109, 40, 217, 0.24),
        rgba(37, 99, 235, 0.15),
        rgba(8, 145, 178, 0.12)
    );
    box-shadow: 0 10px 38px rgba(0, 0, 0, 0.18);
    margin-bottom: 22px;
}

.hero h2 {
    color: #FFFFFF;
    font-size: 25px;
    margin: 0 0 8px 0;
}

.hero p {
    color: #D5D4F0;
    font-size: 14px;
    line-height: 1.7;
    margin-bottom: 0;
}

.metric-card {
    background: linear-gradient(
        135deg,
        rgba(124, 58, 237, 0.20),
        rgba(30, 41, 80, 0.60)
    );
    border: 1px solid rgba(167, 139, 250, 0.30);
    border-radius: 16px;
    padding: 19px;
    min-height: 105px;
    margin-bottom: 8px;
}

.metric-label {
    color: #C4C1E8;
    font-size: 12px;
    margin-bottom: 8px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 27px;
    font-weight: 800;
}

.section-title {
    color: #F5F3FF;
    font-size: 22px;
    font-weight: 750;
    margin: 23px 0 13px 0;
}

.scheme-card {
    background: linear-gradient(
        140deg,
        rgba(37, 29, 73, 0.96),
        rgba(17, 27, 54, 0.97)
    );
    border: 1px solid rgba(139, 123, 220, 0.30);
    border-radius: 19px;
    padding: 22px;
    margin-bottom: 17px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.16);
}

.scheme-card:hover {
    border-color: rgb(103, 232, 249);
    box-shadow: 0 0 20px rgba(103, 232, 249, 0.12);
}

.scheme-name {
    font-size: 20px;
    font-weight: 750;
    color: #FFFFFF;
    margin-bottom: 9px;
}

.scheme-category {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 30px;
    background: rgba(139, 92, 246, 0.20);
    border: 1px solid rgba(167, 139, 250, 0.30);
    color: #D8B4FE;
    font-size: 11px;
    margin-bottom: 12px;
}

.scheme-description {
    color: #D2D5E8;
    line-height: 1.7;
    font-size: 13px;
}

.info-label {
    color: #67E8F9;
    font-size: 12px;
    font-weight: 700;
    margin-top: 13px;
    margin-bottom: 4px;
}

.info-value {
    color: #E2E5F5;
    font-size: 13px;
    line-height: 1.65;
}

div.stButton > button {
    width: 100%;
    min-height: 45px;
    border: 1px solid rgba(196, 181, 253, 0.65);
    border-radius: 11px;
    color: white;
    font-weight: 700;
    background: linear-gradient(
        100deg,
        rgb(124, 58, 237),
        rgb(79, 70, 229),
        rgb(8, 145, 178)
    );
    transition: 0.2s ease;
}

div.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    border-color: rgb(103, 232, 249);
    box-shadow: 0 0 20px rgba(103, 232, 249, 0.34), 0 0 28px rgba(192, 132, 252, 0.18);
    transform: translateY(-1px);
    color: white;
}

[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 48px;
    border: 1px solid rgba(196, 181, 253,  0.72);
    border-radius: 12px;
    color: #FFFFFF;
    font-weight: 800;
    background: linear-gradient(100deg, rgb(124, 58, 237), rgb(79, 70, 229), rgb(8, 145, 178));
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.20);
    transition: all 0.2s ease;
}

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background: rgba(23, 25, 53, 0.96) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(167, 139, 250, 0.45) !important;
    border-radius: 10px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #AAA9C6 !important;
}

div[data-baseweb="select"] > div {
    background-color: rgb(29, 26, 61);
    border-color: rgba(167, 139, 250, 0.45);
    color: white;
}

.stCaption, small {
    color: #B7B8D3 !important;
}

hr {
    border-color: rgba(167, 139, 250, 0.20);
}

.footer {
    text-align: center;
    color: #9CA3C8;
    font-size: 12px;
    padding: 22px 0 8px 0;
}


/* ===== EXTRA RGB NEON DASHBOARD EFFECTS ===== */
:root {
  --rgb-purple: #a855f7;
  --rgb-blue: #3b82f6;
  --rgb-cyan: #22d3ee;
  --rgb-pink: #f472b6;
  --rgb-green: #34d399;
}
@keyframes rgbBorderFlow {
  0% { border-color: rgba(168,85,247,.72); box-shadow: 0 0 12px rgba(168,85,247,.13); }
  33% { border-color: rgba(34,211,238,.72); box-shadow: 0 0 16px rgba(34,211,238,.15); }
  66% { border-color: rgba(244,114,182,.72); box-shadow: 0 0 16px rgba(244,114,182,.15); }
  100% { border-color: rgba(168,85,247,.72); box-shadow: 0 0 12px rgba(168,85,247,.13); }
}
@keyframes rgbBackgroundShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.main .block-container {
  position: relative;
  z-index: 1;
}
.main .block-container::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: -1;
  background: linear-gradient(125deg, rgba(168,85,247,.055), rgba(34,211,238,.045), rgba(244,114,182,.045), rgba(52,211,153,.035));
  background-size: 300% 300%;
  animation: rgbBackgroundShift 18s ease infinite;
}
.hero {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(34,211,238,.48);
  background: linear-gradient(115deg, rgba(124,58,237,.27), rgba(37,99,235,.19), rgba(8,145,178,.19), rgba(219,39,119,.12));
  background-size: 220% 220%;
  animation: rgbBackgroundShift 12s ease infinite, rgbBorderFlow 8s linear infinite;
}
.metric-card, [data-testid="stMetric"], [data-testid="stForm"], [data-testid="stExpander"] {
  animation: rgbBorderFlow 10s linear infinite;
  transition: transform .22s ease, box-shadow .22s ease;
}
.metric-card:hover, [data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  box-shadow: 0 0 22px rgba(34,211,238,.18), 0 0 28px rgba(168,85,247,.12);
}
.scheme-card {
  position: relative;
  overflow: hidden;
  border-image: linear-gradient(120deg, rgba(168,85,247,.72), rgba(34,211,238,.65), rgba(244,114,182,.68)) 1;
  transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
}
.scheme-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 0 22px rgba(34,211,238,.16), 0 0 32px rgba(168,85,247,.13);
}
.main-title {
  background-image: linear-gradient(90deg, #c4b5fd, #67e8f9, #f9a8d4, #86efac, #c4b5fd);
  background-size: 250% auto;
  animation: rgbBackgroundShift 8s linear infinite;
}
div.stButton > button, [data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
  background: linear-gradient(100deg, #7c3aed, #2563eb, #0891b2, #db2777);
  background-size: 250% 250%;
  animation: rgbBackgroundShift 9s ease infinite;
  box-shadow: 0 5px 18px rgba(124,58,237,.23);
}
div.stButton > button:focus, [data-testid="stFormSubmitButton"] button:focus {
  outline: 2px solid #67e8f9;
  outline-offset: 3px;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
  border-color: #22d3ee !important;
  box-shadow: 0 0 0 2px rgba(34,211,238,.13), 0 0 16px rgba(168,85,247,.15) !important;
}
[data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(160deg, #1c1141 0%, #101c42 45%, #102c3e 75%, #241238 100%);
  background-size: 220% 220%;
  animation: rgbBackgroundShift 22s ease infinite;
  border-right: 1px solid rgba(34,211,238,.35);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
  color: #e9d5ff;
  text-shadow: 0 0 12px rgba(168,85,247,.2);
}
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, #a855f7, #3b82f6, #22d3ee, #f472b6) !important;
  background-size: 250% 250%;
  animation: rgbBackgroundShift 7s ease infinite;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: #67e8f9 !important;
  border-bottom-color: #a855f7 !important;
}
::selection { background: rgba(168,85,247,.48); color: #fff; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}


/* ===== SchemeSathi AI premium boxed brand ===== */
.brand-box {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 13px 22px;
    margin: 4px 0 12px 0;
    border: 1px solid transparent;
    border-radius: 17px;
    background:
        linear-gradient(135deg, rgba(35, 20, 75, .98), rgba(16, 31, 66, .96)) padding-box,
        linear-gradient(105deg, #a855f7, #22d3ee, #f472b6, #a855f7) border-box;
    background-size: 100% 100%, 260% 260%;
    animation: rgbBackgroundShift 8s ease infinite, rgbBorderFlow 8s linear infinite;
    box-shadow: 0 0 24px rgba(168, 85, 247, .20),
                inset 0 0 18px rgba(34, 211, 238, .045);
}
.brand-box .brand-icon {
    font-size: 27px;
    filter: drop-shadow(0 0 8px rgba(34, 211, 238, .55));
}
.brand-box .brand-text {
    font-size: clamp(25px, 3.2vw, 39px);
    font-weight: 850;
    letter-spacing: -.9px;
    line-height: 1.2;
    background: linear-gradient(90deg, #e9d5ff, #67e8f9, #f9a8d4, #c4b5fd);
    background-size: 240% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: rgbBackgroundShift 7s linear infinite;
}
.brand-caption {
    color: #c4b5fd;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 18px 4px;
}
@media (max-width: 600px) {
    .brand-box { padding: 11px 15px; gap: 8px; }
    .brand-box .brand-icon { font-size: 23px; }
}


/* ===== EXTRA DARK PURPLE RGB PRO UI ===== */
.stApp {
    min-height: 100vh;
}
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
[data-testid="stHeader"] {
    background: rgba(8, 3, 17, .72) !important;
    border-bottom: 1px solid rgba(168, 85, 247, .16);
    backdrop-filter: blur(14px);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #17082f 0%, #100720 48%, #090515 100%) !important;
    border-right: 1px solid rgba(168, 85, 247, .42);
    box-shadow: 8px 0 35px rgba(91, 33, 182, .12);
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(160deg, rgba(31, 10, 61, .98), rgba(12, 12, 35, .98)) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #eee7ff;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(168, 85, 247, .25);
}
.hero, .scheme-card, .metric-card, [data-testid="stMetric"], [data-testid="stForm"] {
    backdrop-filter: blur(14px);
}
.hero {
    border-radius: 22px;
    box-shadow: 0 12px 38px rgba(0, 0, 0, .22), 0 0 25px rgba(124, 58, 237, .09);
}
.metric-card {
    border: 1px solid rgba(168, 85, 247, .42);
    background: linear-gradient(145deg, rgba(48, 19, 84, .78), rgba(15, 17, 43, .86));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.035), 0 8px 28px rgba(0,0,0,.18);
}
.scheme-card {
    border: 1px solid rgba(168, 85, 247, .42);
    border-radius: 19px;
    background: linear-gradient(145deg, rgba(34, 13, 62, .96), rgba(13, 18, 43, .97));
}
.scheme-card::before {
    content: "";
    display: block;
    height: 3px;
    margin: -22px -22px 18px -22px;
    background: linear-gradient(90deg, #a855f7, #22d3ee, #f472b6, #34d399, #a855f7);
    background-size: 250% 100%;
    animation: rgbBackgroundShift 8s linear infinite;
    opacity: .9;
}
.scheme-card:hover {
    border-color: rgba(34, 211, 238, .8);
    box-shadow: 0 0 0 1px rgba(168,85,247,.13), 0 12px 34px rgba(0,0,0,.24),
                0 0 28px rgba(34,211,238,.11);
}
[data-testid="stForm"] {
    background: linear-gradient(135deg, rgba(39, 15, 70, .78), rgba(12, 18, 43, .88)) !important;
    border: 1px solid rgba(168, 85, 247, .48) !important;
    box-shadow: 0 0 28px rgba(124, 58, 237, .11);
}
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: rgba(13, 7, 30, .96) !important;
    border: 1px solid rgba(168, 85, 247, .5) !important;
    border-radius: 12px !important;
    color: #fff !important;
}
.stTextInput input:hover, .stTextArea textarea:hover {
    border-color: rgba(34, 211, 238, .7) !important;
}
div[data-baseweb="select"] > div {
    background: #160b2c !important;
    border-color: rgba(168, 85, 247, .48) !important;
    border-radius: 11px !important;
}
div.stButton > button, [data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button, .stLinkButton a {
    border: 1px solid rgba(221, 214, 254, .42) !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 22px rgba(124, 58, 237, .22);
}
div.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover, .stLinkButton a:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 18px rgba(168,85,247,.28), 0 0 26px rgba(34,211,238,.15) !important;
}
[data-testid="stExpander"] {
    border: 1px solid rgba(168, 85, 247, .34) !important;
    border-radius: 13px !important;
    background: rgba(20, 10, 41, .76) !important;
}
[data-testid="stAlert"] {
    border-radius: 13px;
    border: 1px solid rgba(168, 85, 247, .28);
    background: rgba(31, 15, 55, .7);
}
.stProgress > div > div {
    background: linear-gradient(90deg, #a855f7, #3b82f6, #22d3ee, #f472b6) !important;
}
.brand-box {
    padding: 15px 25px;
    border-radius: 19px;
    box-shadow: 0 0 28px rgba(168,85,247,.22), 0 0 44px rgba(34,211,238,.06),
                inset 0 0 20px rgba(168,85,247,.06);
}
.brand-caption {
    letter-spacing: 2.4px;
    text-shadow: 0 0 14px rgba(168,85,247,.32);
}
.footer {
    border-top: 1px solid rgba(168, 85, 247, .18);
    margin-top: 25px;
    text-shadow: 0 0 12px rgba(168, 85, 247, .12);
}
@media (max-width: 768px) {
    .main .block-container { padding: 1rem .8rem 1.5rem .8rem; }
    .brand-box { max-width: 100%; }
}


/* ===== CENTERED BRAND + EXTRA RGB ACCENTS ===== */
.brand-wrap {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin: 4px auto 18px auto;
}
.brand-box {
    justify-content: center;
    margin-left: auto;
    margin-right: auto;
    text-align: center;
}
.brand-caption {
    text-align: center;
    margin: 9px auto 18px auto;
}
.section-title {
    text-shadow: 0 0 15px rgba(168, 85, 247, .24);
}
[data-testid="stSelectbox"] label {
    color: #E9D5FF !important;
    font-weight: 750 !important;
    letter-spacing: .2px;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    min-height: 46px !important;
    background: linear-gradient(135deg, #1d0c38, #10172f) !important;
    border: 1px solid rgba(168, 85, 247, .8) !important;
    box-shadow: 0 0 0 1px rgba(34, 211, 238, .08),
                0 0 17px rgba(168, 85, 247, .14);
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    border-color: #22D3EE !important;
    box-shadow: 0 0 18px rgba(34, 211, 238, .2);
}
[data-testid="stSelectbox"] [data-baseweb="select"] input {
    color: #FFFFFF !important;
}
[data-baseweb="popover"] [role="listbox"] {
    background: #16092c !important;
    border: 1px solid rgba(168, 85, 247, .65) !important;
    border-radius: 12px !important;
}
[data-baseweb="menu"] {
    background: #16092c !important;
}
[data-baseweb="menu"] [role="option"] {
    color: #F5F3FF !important;
}
[data-baseweb="menu"] [role="option"]:hover {
    background: rgba(168, 85, 247, .24) !important;
}
[data-testid="stSelectbox"] svg {
    color: #67E8F9 !important;
}
@keyframes rgbPulse {
    0%, 100% { box-shadow: 0 0 12px rgba(168,85,247,.12), 0 0 22px rgba(34,211,238,.04); }
    50% { box-shadow: 0 0 20px rgba(168,85,247,.22), 0 0 30px rgba(244,114,182,.10); }
}
.brand-box {
    animation: rgbBackgroundShift 8s ease infinite, rgbBorderFlow 8s linear infinite, rgbPulse 5s ease-in-out infinite;
}
.metric-card, .scheme-card, [data-testid="stForm"] {
    animation: rgbPulse 7s ease-in-out infinite;
}
@media (prefers-reduced-motion: reduce) {
    .brand-box, .metric-card, .scheme-card, [data-testid="stForm"] {
        animation: none !important;
    }
}



/* ===== FINAL LAYOUT FIX: BIGGER BRAND, FULL SIDEBAR, RGB TEXT ===== */
/* Give the left navigation enough room so labels and example questions show fully. */
section[data-testid="stSidebar"] {
    width: 340px !important;
    min-width: 340px !important;
    max-width: 340px !important;
}
section[data-testid="stSidebar"] > div {
    width: 340px !important;
    min-width: 340px !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    overflow-wrap: anywhere !important;
    word-break: normal !important;
    white-space: normal !important;
    line-height: 1.55;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    min-width: 0 !important;
}

/* Make the top brand box wider, taller and more prominent. */
.brand-wrap {
    width: 100%;
    max-width: 100%;
    padding: 8px 0 5px;
}
.brand-box {
    display: flex !important;
    width: fit-content;
    max-width: 100%;
    min-height: 88px;
    padding: 22px clamp(24px, 5vw, 64px) !important;
    gap: 16px !important;
    border-radius: 24px !important;
    box-shadow: 0 0 30px rgba(168,85,247,.30),
                0 0 48px rgba(34,211,238,.12),
                inset 0 0 24px rgba(244,114,182,.07) !important;
}
.brand-box .brand-icon {
    font-size: clamp(30px, 3.4vw, 42px) !important;
    flex-shrink: 0;
}
.brand-box .brand-text {
    font-size: clamp(30px, 4vw, 48px) !important;
    font-weight: 900 !important;
    letter-spacing: -1.1px !important;
    line-height: 1.18 !important;
    background: linear-gradient(90deg, #fff7ad, #c4b5fd, #67e8f9, #f9a8d4, #86efac, #fff7ad) !important;
    background-size: 300% auto !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: rgbBackgroundShift 6s linear infinite !important;
    filter: drop-shadow(0 0 10px rgba(103,232,249,.18));
    overflow-wrap: anywhere;
}
.brand-caption {
    font-size: 12px !important;
    font-weight: 800 !important;
    letter-spacing: clamp(1px, .25vw, 3px) !important;
    background: linear-gradient(90deg, #c4b5fd, #67e8f9, #f9a8d4, #86efac) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-shadow: none !important;
}

/* Increase usable dashboard width and keep all panels inside the viewport. */
.main .block-container {
    width: 100% !important;
    max-width: 1800px !important;
    padding-left: clamp(1rem, 2.4vw, 2.5rem) !important;
    padding-right: clamp(1rem, 2.4vw, 2.5rem) !important;
}
.hero {
    padding: clamp(24px, 3vw, 38px) !important;
    border-radius: 24px !important;
}
.hero h2 {
    font-size: clamp(23px, 2.4vw, 32px) !important;
    background: linear-gradient(90deg, #ffffff, #a5f3fc, #f5d0fe) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

/* Brighter RGB accents for headings, category cards and controls. */
.section-title {
    background: linear-gradient(90deg, #f5d0fe, #a5f3fc, #c4b5fd) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 850 !important;
}
.scheme-card {
    border: 1px solid rgba(168,85,247,.56) !important;
    box-shadow: 0 10px 28px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.035);
}
.scheme-card:hover {
    border-color: #67e8f9 !important;
    box-shadow: 0 0 22px rgba(34,211,238,.20), 0 0 32px rgba(168,85,247,.17) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    box-shadow: 0 0 12px rgba(34,211,238,.35);
}

@media (max-width: 900px) {
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        width: min(320px, 86vw) !important;
        min-width: min(320px, 86vw) !important;
        max-width: min(320px, 86vw) !important;
    }
    .brand-box {
        min-height: 70px;
        padding: 17px 20px !important;
        gap: 10px !important;
    }
    .brand-box .brand-text {
        font-size: clamp(24px, 6vw, 36px) !important;
    }
}
@media (max-width: 480px) {
    .brand-box { flex-wrap: wrap; justify-content: center; }
    .brand-caption { font-size: 10px !important; letter-spacing: 1px !important; }
}



/* ===== FINAL RGB POLISH: rainbow borders, larger header, tighter spacing ===== */
@keyframes rainbowBorder {
  0% { --rainbow-angle: 0deg; }
  100% { --rainbow-angle: 360deg; }
}
@keyframes rainbowText {
  0% { background-position: 0% 50%; }
  100% { background-position: 300% 50%; }
}
.brand-wrap {
    padding: 2px 0 0 !important;
    margin: 0 auto 5px !important;
    gap: 0 !important;
}
.brand-box {
    width: min(100%, 980px) !important;
    min-height: 112px !important;
    padding: 25px clamp(26px, 5vw, 76px) !important;
    margin: 0 auto 5px !important;
    justify-content: center !important;
    border: 2px solid transparent !important;
    background:
      linear-gradient(135deg, rgba(35,20,75,.98), rgba(9,20,48,.98)) padding-box,
      linear-gradient(110deg, #ff3cac, #784ba0, #2b86c5, #22d3ee, #34d399, #facc15, #ff3cac) border-box !important;
    background-size: 100% 100%, 300% 300% !important;
    animation: rgbBackgroundShift 7s linear infinite, rgbPulse 4s ease-in-out infinite !important;
    box-shadow: 0 0 22px rgba(168,85,247,.24), 0 0 32px rgba(34,211,238,.13),
                inset 0 0 22px rgba(244,114,182,.07) !important;
}
.brand-box .brand-text {
    background: linear-gradient(90deg, #ff4ecd, #ffcf5c, #7cffcb, #55ddff, #b69cff, #ff4ecd) !important;
    background-size: 300% 100% !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: rainbowText 5s linear infinite !important;
    filter: drop-shadow(0 0 10px rgba(103,232,249,.23));
}
.brand-box .brand-icon {
    filter: drop-shadow(0 0 10px rgba(34,211,238,.65)) !important;
}
.brand-caption {
    margin: 3px auto 7px !important;
    line-height: 1.25 !important;
}
/* Remove unnecessary vertical gaps while keeping every line of text. */
.main .block-container {
    padding-top: .8rem !important;
    padding-bottom: 1.2rem !important;
}
.main .block-container [data-testid="stMarkdownContainer"] p {
    margin-bottom: .35rem;
}
.subtitle {
    margin: 2px 0 10px !important;
    padding: 0 !important;
    line-height: 1.45 !important;
}
.hero {
    margin-top: 8px !important;
    margin-bottom: 12px !important;
}
/* Rainbow animated outlines on cards, forms, metrics and expanders. */
.scheme-card, .metric-card, [data-testid="stMetric"],
[data-testid="stForm"], [data-testid="stExpander"] {
    border: 1px solid transparent !important;
    background-image:
      linear-gradient(145deg, rgba(34,13,62,.97), rgba(13,18,43,.98)),
      linear-gradient(115deg, #a855f7, #22d3ee, #f472b6, #34d399, #facc15, #a855f7) !important;
    background-origin: border-box !important;
    background-clip: padding-box, border-box !important;
}
[data-testid="stForm"] {
    background-image:
      linear-gradient(135deg, rgba(39,15,70,.88), rgba(12,18,43,.92)),
      linear-gradient(115deg, #a855f7, #22d3ee, #f472b6, #34d399, #a855f7) !important;
}
@media (max-width: 700px) {
    .brand-box {
        width: 100% !important;
        min-height: 82px !important;
        padding: 16px 12px !important;
        gap: 9px !important;
    }
    .brand-box .brand-text { font-size: clamp(23px, 6vw, 34px) !important; }
    .brand-wrap { margin-bottom: 2px !important; }
    .subtitle { margin-bottom: 8px !important; }
}



/* ===== Requested header mini-boxes + blue outer border ===== */
.brand-box {
    position: relative !important;
    border: 2px solid transparent !important;
    background:
      linear-gradient(135deg, rgba(35,20,75,.98), rgba(9,20,48,.98)) padding-box,
      linear-gradient(120deg, #168bff, #35c8ff, #245bff, #168bff) border-box !important;
    box-shadow: 0 0 18px rgba(37, 99, 235, .25),
                0 0 30px rgba(34, 211, 238, .12),
                inset 0 0 20px rgba(34, 211, 238, .06) !important;
    flex-wrap: wrap !important;
    row-gap: 10px !important;
}
.brand-mini-badges {
    flex-basis: 100% !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 1px !important;
}
.brand-mini-badge {
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
    padding: 5px 11px !important;
    border: 1px solid rgba(96, 165, 250, .8) !important;
    border-radius: 7px !important;
    color: #dbeafe !important;
    background: linear-gradient(135deg, rgba(30,64,175,.34), rgba(8,145,178,.18)) !important;
    box-shadow: inset 0 0 10px rgba(59,130,246,.08), 0 0 9px rgba(37,99,235,.12) !important;
    font-size: 10px !important;
    font-weight: 800 !important;
    letter-spacing: 1px !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
}
/* Make the selected category label red while preserving the dark dropdown. */
[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"],
[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] *,
[data-testid="stSelectbox"] [data-baseweb="select"] input {
    color: #ff4d4d !important;
    -webkit-text-fill-color: #ff4d4d !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: #60a5fa !important;
    fill: #60a5fa !important;
}
@media (max-width: 700px) {
    .brand-box { row-gap: 8px !important; }
    .brand-mini-badges { gap: 6px !important; }
    .brand-mini-badge { font-size: 9px !important; padding: 5px 8px !important; }
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# DEMO DATASET
# =====================================================
SCHEMES = [
    {
        "name": "PM-KISAN",
        "category": "Agriculture",
        "beneficiary": "Eligible farmer families",
        "description": "Financial support scheme for eligible farmer families.",
        "keywords": "farmer agriculture farming rural land financial support",
        "eligibility": "Subject to current official eligibility rules and exclusions.",
        "documents": "Check the official portal for current requirements.",
        "source": "https://www.myscheme.gov.in/"
    },
    {
        "name": "National Scholarship Portal Schemes",
        "category": "Education",
        "beneficiary": "Eligible students",
        "description": "Scholarship opportunities available under different education schemes.",
        "keywords": "student scholarship education college school fees financial aid",
        "eligibility": "Varies by scholarship, course, income and category.",
        "documents": "Check the selected scholarship guidelines.",
        "source": "https://scholarships.gov.in/"
    },
    {
        "name": "Ayushman Bharat PM-JAY",
        "category": "Healthcare",
        "beneficiary": "Eligible beneficiaries",
        "description": "Health assurance scheme providing eligible beneficiaries access to covered hospital care.",
        "keywords": "health medical hospital treatment healthcare family illness",
        "eligibility": "Verify beneficiary eligibility through official channels.",
        "documents": "Requirements depend on official beneficiary verification rules.",
        "source": "https://nha.gov.in/"
    },
    {
        "name": "Pradhan Mantri Awas Yojana",
        "category": "Housing",
        "beneficiary": "Eligible households",
        "description": "Housing programme with different components for eligible households.",
        "keywords": "house home housing rural urban shelter construction",
        "eligibility": "Depends on the applicable component and current guidelines.",
        "documents": "Check the relevant official housing portal.",
        "source": "https://www.myscheme.gov.in/"
    },
    {
        "name": "PM Ujjwala Yojana",
        "category": "Women & Family",
        "beneficiary": "Eligible adult women in qualifying households",
        "description": "LPG connection scheme for eligible women from qualifying households.",
        "keywords": "woman women cooking gas lpg cylinder clean fuel household",
        "eligibility": "Confirm the latest eligibility rules on the official portal.",
        "documents": "Check the official application requirements.",
        "source": "https://www.pmuy.gov.in/"
    },
    {
        "name": "PM SVANidhi",
        "category": "Business",
        "beneficiary": "Eligible street vendors",
        "description": "Support scheme for eligible street vendors under its applicable lending framework.",
        "keywords": "street vendor small business loan working capital shop hawker",
        "eligibility": "Subject to current scheme guidelines and vendor criteria.",
        "documents": "Check the official portal for required documents.",
        "source": "https://www.myscheme.gov.in/"
    },
    {
        "name": "Stand-Up India",
        "category": "Entrepreneurship",
        "beneficiary": "Eligible entrepreneurs",
        "description": "Credit support framework for eligible women and SC/ST entrepreneurs establishing enterprises.",
        "keywords": "business startup entrepreneur women sc st enterprise loan",
        "eligibility": "Verify current scheme status and eligibility with official sources.",
        "documents": "Requirements depend on applicable guidelines and lending institutions.",
        "source": "https://www.myscheme.gov.in/"
    },
    {
        "name": "Sukanya Samriddhi Account",
        "category": "Savings",
        "beneficiary": "Eligible girl children and guardians",
        "description": "Small savings scheme intended for the future financial needs of a girl child.",
        "keywords": "girl child daughter savings future family investment account",
        "eligibility": "Subject to the official account-opening rules.",
        "documents": "Confirm requirements with an authorized provider.",
        "source": "https://www.indiapost.gov.in/"
    }
]

df = pd.DataFrame(SCHEMES)

df["search_text"] = (
    df["name"] + " " +
    df["category"] + " " +
    df["beneficiary"] + " " +
    df["description"] + " " +
    df["keywords"] + " " +
    df["eligibility"]
)

# =====================================================
# NLP MODEL
# =====================================================
@st.cache_resource
def load_search_model(texts):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english"
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

vectorizer, scheme_vectors = load_search_model(
    tuple(df["search_text"].tolist())
)

def search_schemes(query, top_k=5, category="All Categories"):
    query = query.strip()

    if not query:
        return pd.DataFrame()

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(
        query_vector, scheme_vectors
    ).flatten()

    results = df.copy()
    results["similarity_score"] = scores

    if category != "All Categories":
        results = results[results["category"] == category]

    results = results[results["similarity_score"] > 0]
    results = results.sort_values(
        "similarity_score", ascending=False
    ).head(top_k)

    return results

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0 22px 0;">
        <div style="font-size:43px;">🏛️</div>
        <div style="font-size:23px; font-weight:800;
        background:linear-gradient(90deg,#C4B5FD,#67E8F9);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;">
            SchemeSathi AI
        </div>
        <div style="color:#B9B8D9;font-size:12px;">
            Government Scheme Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛️ Search Settings")

    categories = ["All Categories"] + sorted(
        df["category"].unique().tolist()
    )

    selected_category = st.selectbox(
        "Choose scheme category",
        categories
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=8,
        value=5
    )

    st.markdown("---")
    st.markdown("### 💡 Example Questions")

    examples = [
        "I am a student looking for a scholarship",
        "I am a farmer looking for support",
        "I need information about health schemes",
        "I want to start a small business",
        "I need a government housing scheme"
    ]

    for example in examples:
        st.caption("• " + example)

    st.markdown("---")
    st.caption("🔎 Search engine: TF-IDF + Cosine Similarity")
    st.caption("📚 Dataset: Initial demonstration dataset")

# =====================================================
# MAIN DASHBOARD HEADER
# =====================================================
st.markdown(
    '''
    <div class="brand-wrap">
        <div class="brand-box">
            <span class="brand-icon">🏛️</span>
            <span class="brand-text">SchemeSathi AI ✨</span>
            <div class="brand-mini-badges">
                <span class="brand-mini-badge">⚡ AI POWERED</span>
                <span class="brand-mini-badge">🔎 SMART SEARCH</span>
                <span class="brand-mini-badge">🏛️ SCHEME FINDER</span>
            </div>
        </div>
        <div class="brand-caption">SMART GOVERNMENT SCHEME DISCOVERY</div>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover government schemes that may match your needs.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
    <h2>🏛️ Find the support you are looking for</h2>
    <p>
        Explore scheme information using an NLP-powered search engine.
        Enter your needs in simple language, review the matching results,
        and follow the official links to verify the latest information.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# DASHBOARD METRICS
# =====================================================
m1, m2, m3 = st.columns(3)

metrics = [
    (m1, "📚", "Demo Schemes", str(len(df))),
    (m2, "🗂️", "Categories", str(df["category"].nunique())),
    (m3, "🧠", "Search Engine", "TF-IDF")
]

for col, icon, label, value in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# SEARCH FORM
# =====================================================
st.markdown(
    '<div class="section-title">🔎 Find Your Scheme</div>',
    unsafe_allow_html=True
)

with st.form("scheme_search_form"):
    query = st.text_input(
        "Describe what you need",
        placeholder="Example: I am a student looking for financial help..."
    )

    submitted = st.form_submit_button(
        "✨ Search Government Schemes",
        use_container_width=True
    )

# =====================================================
# RESULTS
# =====================================================
if submitted:
    if not query.strip():
        st.warning("Please enter your question first.")
    else:
        results = search_schemes(
            query,
            top_k=top_k,
            category=selected_category
        )

        st.markdown(
            '<div class="section-title">🎯 Matching Results</div>',
            unsafe_allow_html=True
        )

        if results.empty:
            st.info(
                "No relevant scheme found in this demo dataset. "
                "Try different keywords or choose another category."
            )
        else:
            st.caption(
                f"{len(results)} matching result(s). "
                "Similarity scores indicate text relevance, not eligibility."
            )

            for _, row in results.iterrows():
                name = html.escape(str(row["name"]))
                category = html.escape(str(row["category"]))
                beneficiary = html.escape(str(row["beneficiary"]))
                description = html.escape(str(row["description"]))
                eligibility = html.escape(str(row["eligibility"]))
                documents = html.escape(str(row["documents"]))
                source = html.escape(str(row["source"]), quote=True)
                score = float(row["similarity_score"])

                st.markdown(f"""
                <div class="scheme-card">
                    <div class="scheme-name">🏛️ {name}</div>
                    <div class="scheme-category">{category}</div>
                    <div class="scheme-description">{description}</div>
                    <div class="info-label">👥 Intended Beneficiaries</div>
                    <div class="info-value">{beneficiary}</div>
                    <div class="info-label">✅ Eligibility Guidance</div>
                    <div class="info-value">{eligibility}</div>
                    <div class="info-label">📄 Document Guidance</div>
                    <div class="info-value">{documents}</div>
                    <div class="info-label">📊 Text Similarity Score</div>
                    <div class="info-value">{score:.3f}</div>
                    <br>
                    <a href="{source}" target="_blank"
                       style="color:#67E8F9;font-weight:700;
                       text-decoration:none;">
                       🌐 Open Official Information ↗
                    </a>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"View details: {name}"):
                    st.write("**Category:**", row["category"])
                    st.write("**Beneficiaries:**", row["beneficiary"])
                    st.write("**Description:**", row["description"])
                    st.write("**Eligibility:**", row["eligibility"])
                    st.write("**Documents:**", row["documents"])
                    st.link_button(
                        "Visit Information Source",
                        row["source"]
                    )

            st.info(
                "Important: This prototype searches a small demo dataset. "
                "A match does not confirm eligibility, scheme availability, "
                "or benefit entitlement. Verify all details on official portals."
            )

else:
    st.markdown(
        '<div class="section-title">🌟 Explore Scheme Categories</div>',
        unsafe_allow_html=True
    )

    category_icons = {
        "Agriculture": "🌾",
        "Education": "🎓",
        "Healthcare": "🏥",
        "Housing": "🏠",
        "Women & Family": "👩",
        "Business": "🏪",
        "Entrepreneurship": "🚀",
        "Savings": "💰"
    }

    columns = st.columns(4)

    for i, category in enumerate(sorted(df["category"].unique())):
        with columns[i % 4]:
            count = int((df["category"] == category).sum())
            icon = category_icons.get(category, "📌")

            st.markdown(f"""
            <div class="scheme-card" style="padding:17px;min-height:120px;">
                <div style="font-size:26px;">{icon}</div>
                <div style="color:#FFFFFF;font-size:14px;
                font-weight:700;margin-top:10px;">
                    {html.escape(category)}
                </div>
                <div style="color:#B9B8D9;font-size:12px;margin-top:5px;">
                    {count} demo scheme(s)
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <h2>💡 How to get started</h2>
        <p>
            1. Type your question in the search box.<br>
            2. Select a category from the sidebar if needed.<br>
            3. Click Search Government Schemes.<br>
            4. Read the matching results and verify the official information.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    SchemeSathi AI · NLP-powered scheme discovery prototype<br>
    Information should always be verified through official government sources.
</div>
""", unsafe_allow_html=True)
