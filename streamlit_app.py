import base64
import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ──────────────────────────── Config ────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent / "image_identifier_model.h5"
IMG_SIZE = (224, 224)
REAL_COLOR = "#34D3A5"
AI_COLOR = "#FF6B9A"

st.set_page_config(
    page_title="Pixel Forensics · AI Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Styling ────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Manrope:wght@400;500;600&display=swap');

@property --i { syntax: '<integer>'; inherits: false; initial-value: 0; }
@property --d { syntax: '<integer>'; inherits: false; initial-value: 0; }

:root {
  --bg: #0E1320; --surface: #161D2E; --line: #26304A;
  --text: #E8ECF6; --muted: #8E99B5;
  --real: #34D3A5; --ai: #FF6B9A;
}

html, body, [class*="css"], .stApp { font-family: 'Manrope', sans-serif; }
.stApp {
  background:
    radial-gradient(900px 500px at 85% -10%, rgba(255,107,154,.10), transparent 60%),
    radial-gradient(800px 500px at 0% 0%, rgba(52,211,165,.08), transparent 55%),
    var(--bg);
  color: var(--text);
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.2rem; max-width: 1150px; }
section[data-testid="stSidebar"] { background: #0B101C; border-right: 1px solid var(--line); }

h1.hero {
  font-family: 'Sora', sans-serif; font-weight: 700; font-size: 2.5rem;
  letter-spacing: -0.03em; margin: 0 0 .3rem 0; color: var(--text);
}
p.sub { color: var(--muted); font-size: 1.02rem; margin: 0 0 1.6rem 0; max-width: 620px; }

.panel {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 16px; padding: 1.2rem 1.3rem;
}
.panel-title { font-family: 'Sora', sans-serif; font-weight: 600; font-size: .95rem; color: var(--muted); margin-bottom: .8rem; }

/* uploader */
[data-testid="stFileUploader"] section {
  background: var(--surface); border: 1.5px dashed #34405F; border-radius: 16px;
  transition: border-color .25s, background .25s;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--real); background: #19223A; }

/* primary button */
.stButton > button {
  width: 100%; border: 0; border-radius: 12px; padding: .75rem 1rem;
  font-family: 'Sora', sans-serif; font-weight: 600; color: #08111F;
  background: linear-gradient(90deg, var(--real), #5BE3C8);
  transition: transform .15s ease, box-shadow .25s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(52,211,165,.28); color:#08111F; }
.stButton > button:focus-visible { outline: 2px solid #fff; outline-offset: 2px; }

/* image frame + scanner */
.frame { position: relative; border-radius: 14px; overflow: hidden; border: 1px solid var(--line); background:#0A0F1B; }
.frame img { display: block; width: 100%; max-height: 460px; object-fit: contain; }
.frame.real { border-color: var(--real); box-shadow: 0 0 0 1px var(--real), 0 0 32px rgba(52,211,165,.22); }
.frame.ai { border-color: var(--ai); box-shadow: 0 0 0 1px var(--ai), 0 0 32px rgba(255,107,154,.22); }
.grid-overlay {
  position:absolute; inset:0; pointer-events:none; opacity:.0;
  background-image: linear-gradient(rgba(255,255,255,.07) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.07) 1px, transparent 1px);
  background-size: 28px 28px;
}
.scanning .grid-overlay { opacity: 1; }
.scan-line {
  position:absolute; left:0; right:0; height: 70px; top:-70px; pointer-events:none;
  background: linear-gradient(180deg, transparent, rgba(52,211,165,.45), transparent);
  border-bottom: 2px solid var(--real);
  animation: scan 1.5s ease-in-out infinite;
}
@keyframes scan { from { top:-70px; } to { top:100%; } }

/* status steps */
.steps { display:flex; flex-direction:column; gap:.55rem; margin-top:.4rem; }
.step { display:flex; align-items:center; gap:.7rem; color: var(--muted); font-size:.95rem; }
.step .dot { width:10px; height:10px; border-radius:50%; background:#33405F; }
.step.done { color: var(--text); }
.step.done .dot { background: var(--real); }
.step.active { color: var(--text); }
.step.active .dot { background: var(--real); animation: pulse 1s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(52,211,165,.6)} 100%{box-shadow:0 0 0 10px rgba(52,211,165,0)} }

/* gauge */
.gauge-wrap { display:flex; flex-direction:column; align-items:center; }
.gauge { width: 100%; max-width: 340px; overflow: visible; }
.arc-fg {
  stroke-dasharray: 282.74; stroke-dashoffset: 282.74;
  animation: fill 1.6s cubic-bezier(.2,.8,.2,1) .15s forwards;
}
@keyframes fill { to { stroke-dashoffset: var(--off); } }
.needle { transform-origin: 110px 110px; transform: rotate(-90deg);
  animation: sweep 1.6s cubic-bezier(.2,.8,.2,1) .15s forwards; }
@keyframes sweep { to { transform: rotate(var(--ang)); } }

.pct { font-family:'Sora',sans-serif; font-weight:700; font-size:3rem; letter-spacing:-0.03em;
  margin-top:-.4rem; line-height:1; animation: count 1.6s cubic-bezier(.2,.8,.2,1) .15s forwards;
  counter-reset: i var(--i) d var(--d); }
.pct::after { content: counter(i) "." counter(d, decimal-leading-zero) "%"; }
@keyframes count { from { --i:0; --d:0; } to { --i: var(--ti); --d: var(--td); } }

.verdict { font-family:'Sora',sans-serif; font-weight:600; font-size:1.25rem; margin-top:.5rem; }
.tier { display:inline-block; margin-top:.45rem; padding:.2rem .7rem; border-radius:999px; font-size:.82rem; border:1px solid currentColor; }

/* split meter */
.meter-row { margin-top:1.1rem; }
.meter-head { display:flex; justify-content:space-between; font-size:.9rem; margin-bottom:.35rem; }
.track { height:10px; background:#0F1626; border-radius:999px; overflow:hidden; border:1px solid var(--line); }
.bar { height:100%; width:0; border-radius:999px; animation: grow 1.3s cubic-bezier(.2,.8,.2,1) .3s forwards; }
@keyframes grow { to { width: var(--w); } }

/* stat chips */
.stats { display:grid; grid-template-columns: repeat(4, 1fr); gap:.6rem; margin-top:1.1rem; }
.stat { background:#0F1626; border:1px solid var(--line); border-radius:12px; padding:.6rem .75rem; }
.stat .k { color: var(--muted); font-size:.76rem; }
.stat .v { font-family:'Sora',sans-serif; font-weight:600; font-size:.98rem; margin-top:.1rem; }

.note { color: var(--muted); font-size:.82rem; margin-top:1rem; line-height:1.5; }
.empty { text-align:center; color: var(--muted); padding: 3rem 1rem; }
.fade-in { animation: fade .5s ease both; }
@keyframes fade { from { opacity:0; transform: translateY(6px); } to { opacity:1; transform:none; } }


/* homepage demo */
.home { display:grid; grid-template-columns: 1.15fr 1fr; gap: 2rem; align-items:center; margin-top:.4rem; }
.demo { position:relative; aspect-ratio: 400 / 260; border-radius:16px; overflow:hidden; border:1px solid var(--line); background:#0A0F1B; }
.demo .layer, .demo .layer svg { position:absolute; inset:0; width:100%; height:100%; display:block; }
.demo .ai-layer { animation: reveal 7s ease-in-out infinite alternate; }
.demo .divider { position:absolute; top:0; bottom:0; width:2px; background:#fff; box-shadow:0 0 14px 2px rgba(255,255,255,.55); animation: slide 7s ease-in-out infinite alternate; }
@keyframes reveal { from { clip-path: inset(0 0 0 18%); } to { clip-path: inset(0 0 0 82%); } }
@keyframes slide { from { left: 18%; } to { left: 82%; } }
.demo .tag { position:absolute; top:.7rem; padding:.2rem .65rem; border-radius:999px; font-size:.78rem; font-weight:600; background:rgba(10,15,27,.72); border:1px solid currentColor; backdrop-filter: blur(4px); }
.demo .t-real { left:.7rem; color: var(--real); }
.demo .t-ai { right:.7rem; color: var(--ai); }
.demo .br { position:absolute; width:18px; height:18px; border:2px solid rgba(232,236,246,.7); }
.demo .tl { top:8px; left:8px; border-right:0; border-bottom:0; }
.demo .tr { top:8px; right:8px; border-left:0; border-bottom:0; }
.demo .bl { bottom:8px; left:8px; border-right:0; border-top:0; }
.demo .brr { bottom:8px; right:8px; border-left:0; border-top:0; }
.demo-caption { color: var(--muted); font-size:.85rem; margin-top:.6rem; }
.how h3 { font-family:'Sora',sans-serif; font-weight:700; font-size:1.35rem; letter-spacing:-0.02em; margin:0 0 1rem 0; }
.how-step { display:flex; gap:.9rem; padding:.85rem 0; border-top:1px solid var(--line); }
.how-step:first-of-type { border-top:0; }
.how-step .n { flex:0 0 auto; width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-family:'Sora',sans-serif; font-weight:600; font-size:.9rem; color:#08111F; background:linear-gradient(135deg, var(--real), var(--ai)); }
.how-step .t { font-weight:600; }
.how-step .d { color: var(--muted); font-size:.9rem; margin-top:.1rem; line-height:1.45; }
@media (max-width: 860px) { .home { grid-template-columns: 1fr; } }
@media (max-width: 720px) { .stats { grid-template-columns: repeat(2, 1fr); } h1.hero { font-size: 1.9rem; } }
@media (prefers-reduced-motion: reduce) {
  .arc-fg, .needle, .pct, .bar, .scan-line, .fade-in { animation-duration: .01s !important; animation-delay: 0s !important; }
}
</style>
"""


def clean(markup: str) -> str:
    """Strip indentation and blank lines so Markdown won't treat the HTML as code."""
    return "\n".join(l.strip() for l in markup.splitlines() if l.strip())


def html(markup: str) -> None:
    st.markdown(clean(markup), unsafe_allow_html=True)


html(CSS)


# ──────────────────────────── Model ────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    base.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(224, 224, 3)),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.load_weights(str(MODEL_PATH))
    return model


def predict(model, image: Image.Image) -> float:
    resized = image.resize(IMG_SIZE)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return float(model.predict(arr[np.newaxis, ...], verbose=0)[0][0])


# ──────────────────────────── Helpers ────────────────────────────
def to_data_uri(image: Image.Image, max_w: int = 900) -> str:
    img = image.copy()
    img.thumbnail((max_w, max_w))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def confidence_tier(conf: float) -> str:
    if conf >= 90:
        return "Very high confidence"
    if conf >= 75:
        return "High confidence"
    if conf >= 60:
        return "Moderate confidence"
    return "Low confidence · treat as uncertain"


def frame(uri: str, state: str = "") -> str:
    scan = '<div class="scan-line"></div>' if state == "scanning" else ""
    return f"""
    <div class="frame {state}">
      <img src="{uri}" alt="Uploaded image"/>
      <div class="grid-overlay"></div>{scan}
    </div>
    """


def steps_html(active: int) -> str:
    labels = [
        "Resizing to 224 × 224",
        "Normalising pixel values",
        "Extracting MobileNetV2 features",
        "Scoring real vs. AI-generated",
    ]
    rows = ""
    for idx, text in enumerate(labels):
        cls = "done" if idx < active else "active" if idx == active else ""
        rows += f'<div class="step {cls}"><span class="dot"></span>{text}</div>'
    return f'<div class="panel"><div class="panel-title">Analysing image</div><div class="steps">{rows}</div></div>'


def gauge_html(conf: float, color: str) -> str:
    ticks = ""
    for v in (0, 25, 50, 75, 100):
        a = np.deg2rad(180 - v * 1.8)
        x1, y1 = 110 + 76 * np.cos(a), 110 - 76 * np.sin(a)
        x2, y2 = 110 + 84 * np.cos(a), 110 - 84 * np.sin(a)
        ticks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#46526F" stroke-width="2"/>'
    return f"""
    <div class="gauge-wrap">
      <svg class="gauge" viewBox="0 0 220 130" role="img" aria-label="Confidence gauge {conf:.2f} percent">
        <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="#0F1626" stroke-width="16" stroke-linecap="round"/>
        <path class="arc-fg" d="M 20 110 A 90 90 0 0 1 200 110"
              fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"/>
        {ticks}
        <g class="needle">
          <line x1="110" y1="110" x2="110" y2="38" stroke="#E8ECF6" stroke-width="3" stroke-linecap="round"/>
        </g>
        <circle cx="110" cy="110" r="7" fill="#E8ECF6"/>
        <circle cx="110" cy="110" r="3" fill="{color}"/>
      </svg>
      <div class="pct" style="color:{color}"></div>
    </div>
    """


def meter(label: str, pct: float, color: str, cls: str) -> str:
    return f"""
    <div class="meter-row">
      <div class="meter-head"><span>{label}</span><span style="color:{color};font-weight:600">{pct:.2f}%</span></div>
      <div class="track"><div class="bar {cls}" style="background:{color}"></div></div>
    </div>
    """


def result_html(pred: float, threshold: float, meta: dict) -> str:
    is_real = pred > threshold
    real_pct, ai_pct = pred * 100, (1 - pred) * 100
    conf = real_pct if is_real else ai_pct
    color = REAL_COLOR if is_real else AI_COLOR
    label = "Real photograph" if is_real else "AI-generated image"
    uid = f"r{int(time.time() * 1000)}"
    offset = 282.74 * (1 - conf / 100)
    angle = -90 + conf * 1.8
    ti, td = int(conf), int(round((conf - int(conf)) * 100)) % 100
    if td == 0 and conf - int(conf) > 0.995:
        ti, td = int(conf) + 1, 0
    dyn_css = f"""
    <style>
    @keyframes fill-{uid} {{ to {{ stroke-dashoffset: {offset:.2f}; }} }}
    @keyframes sweep-{uid} {{ to {{ transform: rotate({angle:.1f}deg); }} }}
    @keyframes count-{uid} {{ from {{ --i:0; --d:0; }} to {{ --i:{ti}; --d:{td}; }} }}
    @keyframes real-{uid} {{ to {{ width: {real_pct:.2f}%; }} }}
    @keyframes ai-{uid} {{ to {{ width: {ai_pct:.2f}%; }} }}
    #{uid} .arc-fg {{ animation-name: fill-{uid}; }}
    #{uid} .needle {{ animation-name: sweep-{uid}; }}
    #{uid} .pct {{ animation-name: count-{uid}; }}
    #{uid} .bar.real {{ animation-name: real-{uid}; }}
    #{uid} .bar.ai {{ animation-name: ai-{uid}; }}
    </style>
    """
    stats = "".join(
        f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in meta.items()
    )
    return f"""
    {dyn_css}
    <div class="panel fade-in" id="{uid}">
      <div class="panel-title">Detection result</div>
      {gauge_html(conf, color)}
      <div style="text-align:center">
        <div class="verdict" style="color:{color}">{label}</div>
        <span class="tier" style="color:{color}">{confidence_tier(conf)}</span>
      </div>
      {meter("Real photograph", real_pct, REAL_COLOR, "real")}
      {meter("AI-generated", ai_pct, AI_COLOR, "ai")}
      <div class="stats">{stats}</div>
      <div class="note">This is a statistical estimate from a MobileNetV2 classifier, not proof of origin.
      Compression, filters and screenshots can shift the result.</div>
    </div>
    """



def _mix(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


@st.cache_data(show_spinner=False)
def demo_scene() -> str:
    """Two renderings of one abstract landscape: smooth (real) and mosaic (AI)."""
    W, H, C = 400, 260, 20
    sun = (290, 82, 42)

    def hill1(x): return 178 + 16 * np.sin(x / 55)
    def hill2(x): return 208 + 11 * np.sin(x / 38 + 1)

    # smooth, photographic-style version
    xs = range(0, W + 1, 4)
    h1 = " ".join(f"{x},{hill1(x):.1f}" for x in xs)
    h2 = " ".join(f"{x},{hill2(x):.1f}" for x in xs)
    smooth = f"""
    <svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <defs><linearGradient id="skyR" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#12304A"/><stop offset="1" stop-color="#2F8F8A"/></linearGradient></defs>
      <rect width="{W}" height="{H}" fill="url(#skyR)"/>
      <circle cx="{sun[0]}" cy="{sun[1]}" r="{sun[2]}" fill="#F6E7B4"/>
      <polygon points="0,{H} {h1} {W},{H}" fill="#14504A"/>
      <polygon points="0,{H} {h2} {W},{H}" fill="#0C3535"/>
    </svg>
    """

    # mosaic version of the same scene, pink-tinted, with noisy cells
    rng = np.random.default_rng(7)
    rects = []
    for cy in range(0, H, C):
        for cx in range(0, W, C):
            mx, my = cx + C / 2, cy + C / 2
            if (mx - sun[0]) ** 2 + (my - sun[1]) ** 2 < sun[2] ** 2:
                col = (255, 214, 230)
            elif my > hill2(mx):
                col = (92, 24, 70)
            elif my > hill1(mx):
                col = (150, 40, 100)
            else:
                col = _mix((52, 36, 98), (255, 107, 154), my / H)
            k = 1 + rng.uniform(-0.13, 0.13)
            r, g, b = (min(255, int(v * k)) for v in col)
            rects.append(f'<rect x="{cx}" y="{cy}" width="{C}" height="{C}" fill="rgb({r},{g},{b})"/>')
    pixel = f'''<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice" aria-hidden="true">{"".join(rects)}</svg>'''

    return f"""
    <div class="demo">
      <div class="layer">{smooth}</div>
      <div class="layer ai-layer">{pixel}</div>
      <div class="divider"></div>
      <span class="tag t-real">Real</span><span class="tag t-ai">AI-generated</span>
      <i class="br tl"></i><i class="br tr"></i><i class="br bl"></i><i class="br brr"></i>
    </div>
    """


def home_html() -> str:
    steps = [
        ("Upload a photo", "Drop a JPG, PNG, BMP or WEBP file into the box above."),
        ("Scan it", "The image is resized to 224 × 224 and passed through the MobileNetV2 network."),
        ("Read the gauge", "You get a percentage for real and for AI-generated, plus a confidence label."),
    ]
    rows = "".join(
        f'<div class="how-step"><div class="n">{i}</div><div><div class="t">{t}</div><div class="d">{d}</div></div></div>'
        for i, (t, d) in enumerate(steps, 1)
    )
    return f"""
    <div class="home fade-in">
      <div>
        {demo_scene()}
        <div class="demo-caption">Same scene, two origins. The detector looks for the statistical traces that separate them.</div>
      </div>
      <div class="how">
        <h3>How a scan works</h3>
        {rows}
      </div>
    </div>
    """


# ──────────────────────────── Sidebar ────────────────────────────
threshold = 0.5  # sigmoid output above this is classified as real

with st.sidebar:
    st.markdown("### Pixel Forensics")
    st.caption("Real vs. AI-generated image detection")
    st.markdown("**Model**")
    st.caption("MobileNetV2 backbone · global average pooling · Dense(128) · sigmoid output")
    st.markdown("**Input**")
    st.caption("RGB, resized to 224 × 224, scaled to 0–1")
    st.divider()
    st.caption("Tip: use the original file. Heavily compressed or cropped images are harder to judge.")

# ──────────────────────────── Main ────────────────────────────
html('<h1 class="hero">AI vs real image detector</h1>')
html('<p class="sub">Upload a photo and the model scores how likely it is to be a real photograph or generated by AI.</p>')

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded is None:
    html(home_html())
else:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read this file as an image: {exc}")
        st.stop()

    uri = to_data_uri(image)
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        frame_slot = st.empty()
        frame_slot.markdown(clean(frame(uri)), unsafe_allow_html=True)
        run = st.button("Analyse image", type="primary")

    with right:
        out_slot = st.empty()

        if run:
            try:
                frame_slot.markdown(clean(frame(uri, "scanning")), unsafe_allow_html=True)

                for step in range(4):
                    out_slot.markdown(steps_html(step), unsafe_allow_html=True)
                    if step == 2:
                        t0 = time.perf_counter()
                        model = load_detector()
                        pred = predict(model, image)
                        infer_ms = (time.perf_counter() - t0) * 1000
                    time.sleep(0.4)

                is_real = pred > threshold
                frame_slot.markdown(clean(frame(uri, "real" if is_real else "ai")), unsafe_allow_html=True)

                size_kb = (uploaded.size or 0) / 1024
                meta = {
                    "Resolution": f"{image.width}×{image.height}",
                    "File size": f"{size_kb/1024:.2f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB",
                    "Format": (Path(uploaded.name).suffix.lstrip(".") or "—").upper(),
                    "Inference": f"{infer_ms:.0f} ms",
                }
                out_slot.markdown(clean(result_html(pred, threshold, meta)), unsafe_allow_html=True)
            except Exception as exc:
                frame_slot.markdown(clean(frame(uri)), unsafe_allow_html=True)
                out_slot.empty()
                st.error(f"Analysis failed: {exc}")
        else:
            out_slot.markdown(
                '<div class="panel empty">Press <b>Analyse image</b> to run the detector.</div>',
                unsafe_allow_html=True,
            )

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.caption("Powered by MobileNetV2 · AI Image Detection Project")