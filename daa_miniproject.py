# string_matching_app_fixed.py
import streamlit as st
import pandas as pd
import time
import sys
import plotly.express as px
from typing import List, Tuple, Dict

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="String Matching Visualizer", layout="wide")

# ---------------------- CUSTOM STYLES ----------------------
st.markdown("""
<style>
:root {
    --bg: #f9fbff;
    --accent: #2563eb;
    --accent-light: #e3ebff;
    --text: #000000;  /* All normal text black */
    --muted: #334155;
    --card: #ffffff;
    --border: #dbe3f0;
}

.stApp {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

h1, h2, h3, h4 {
    color: var(--accent);
    font-weight: 700;
}

.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 22px;
    font-weight: 600;
    color: var(--accent);
    border-bottom: 2px solid var(--accent-light);
    padding-bottom: 6px;
    margin-top: 25px;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    color: var(--text);
    box-shadow: 0 2px 8px rgba(37,99,235,0.08);
    margin-bottom: 20px;
}

.metric {
    background: var(--accent-light);
    border-radius: 10px;
    text-align: center;
    padding: 16px;
    color: var(--text);
    font-weight: 500;
}

.metric h2 {
    color: var(--accent);
    font-size: 26px;
    margin-bottom: 0px;
}

table.dataframe th {
    background: var(--accent);
    color: #ffffff !important;
    font-weight: 600;
    text-align: left;
}

table.dataframe td {
    color: var(--text) !important;
    border-bottom: 1px solid var(--border);
}

pre {
    background: #f3f6ff;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    padding: 10px;
    font-size: 13px;
}

/* --- Expander customization --- */
.streamlit-expanderHeader {
    color: #000000 !important;  /* black text for expanders */
    font-weight: 600 !important;
    font-size: 15px !important;
}

.streamlit-expanderContent {
    color: var(--text) !important; /* ensure inside expander is black */
}

footer {
    text-align: center;
    padding: 14px;
    color: var(--muted);
    font-size: 13px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("""
<div style="background: linear-gradient(90deg, #2563eb, #60a5fa);
            padding: 22px 26px;
            border-radius: 12px;
            color: white;
            box-shadow: 0 4px 14px rgba(37,99,235,0.3);
            margin-bottom: 30px;">
    <h1 style="margin-bottom:4px;">💎 String Matching Visualizer</h1>
    <p style="margin:0;font-size:15px;">Compare <b>Naive</b> and <b>Rabin–Karp</b> algorithms step-by-step — with runtime, memory use, and iteration logs.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- VALIDATION ----------------------
def validate_inputs(text: str, pattern: str):
    if not text or not pattern:
        raise ValueError("Text and Pattern cannot be empty.")
    if len(pattern) > len(text):
        raise ValueError("Pattern cannot be longer than text.")
    if len(text) > 20000:
        raise ValueError("Text too large for interactive demo (max 20,000 chars).")
    return text, pattern

# ---------------------- ALGORITHMS ----------------------
def naive_string_matching_instrumented(text: str, pattern: str):
    n, m = len(text), len(pattern)
    matches, iterations = [], []
    for i in range(n - m + 1):
        start_ns = time.perf_counter_ns()
        comparisons, matched, details = 0, True, []
        for j in range(m):
            comparisons += 1
            if text[i+j] != pattern[j]:
                matched = False
                details.append(f"t[{i+j}]='{text[i+j]}' != p[{j}]='{pattern[j]}' (stop)")
                break
            else:
                details.append(f"t[{i+j}]='{text[i+j]}' == p[{j}]='{pattern[j]}'")
        end_ns = time.perf_counter_ns()
        if matched:
            matches.append(i)
        iterations.append({
            "index": i,
            "matched": matched,
            "comparisons": comparisons,
            "iteration_time_s": (end_ns - start_ns)/1e9,
            "memory_bytes": sys.getsizeof(text) + sys.getsizeof(pattern),
            "details": details
        })
    return matches, iterations

def rabin_karp_instrumented(text: str, pattern: str, d: int = 256, q: int = 101):
    n, m = len(text), len(pattern)
    matches, iterations = [], []
    h = pow(d, m-1, q)
    p_hash = 0
    t_hash = 0
    t0 = time.perf_counter_ns()
    for i in range(m):
        p_hash = (d * p_hash + ord(pattern[i])) % q
        t_hash = (d * t_hash + ord(text[i])) % q
    t1 = time.perf_counter_ns()
    iterations.append({
        "index": -1, "phase": "init_hash", "p_hash": p_hash, "t_hash": t_hash,
        "matched": False, "comparisons": 0,
        "iteration_time_s": (t1 - t0)/1e9,
        "memory_bytes": sys.getsizeof(text) + sys.getsizeof(pattern),
        "details": [f"initial p_hash={p_hash}, t_hash={t_hash}, h={h}"]
    })
    for i in range(n - m + 1):
        start = time.perf_counter_ns()
        step_details, comparisons, matched_flag = [], 0, False
        if p_hash == t_hash:
            match = True
            for j in range(m):
                comparisons += 1
                if text[i+j] != pattern[j]:
                    match = False
                    step_details.append(f"t[{i+j}]='{text[i+j]}' != p[{j}]='{pattern[j]}'")
                    break
                else:
                    step_details.append(f"t[{i+j}]='{text[i+j]}' == p[{j}]='{pattern[j]}'")
            if match:
                matches.append(i)
            matched_flag = match
        else:
            step_details.append(f"Hash mismatch: {p_hash} vs {t_hash}")
        end = time.perf_counter_ns()
        iterations.append({
            "index": i, "phase": "check",
            "p_hash": p_hash, "t_hash": t_hash,
            "matched": matched_flag, "comparisons": comparisons,
            "iteration_time_s": (end - start)/1e9,
            "memory_bytes": sys.getsizeof(text) + sys.getsizeof(pattern),
            "details": step_details
        })
        if i < n - m:
            rh_start = time.perf_counter_ns()
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i+m])) % q
            rh_end = time.perf_counter_ns()
            iterations.append({
                "index": i + 0.5, "phase": "rolling_hash",
                "p_hash": p_hash, "t_hash": t_hash, "matched": False,
                "comparisons": 0, "iteration_time_s": (rh_end - rh_start)/1e9,
                "memory_bytes": sys.getsizeof(text) + sys.getsizeof(pattern),
                "details": [f"rolled t_hash -> {t_hash}"]
            })
    return matches, iterations

# ---------------------- INPUTS ----------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="section-title">📝 Input Strings</div>', unsafe_allow_html=True)
    text = st.text_area("Enter main text (T):", value="ABABDABACDABABCABAB", height=140)
    pattern = st.text_input("Enter pattern (P):", value="ABABCABAB")
with col2:
    st.markdown('<div class="section-title">⚙️ Controls</div>', unsafe_allow_html=True)
    run_btn = st.button("🚀 Run and Analyze", use_container_width=True)

# ---------------------- EXECUTION ----------------------
if run_btn:
    try:
        text, pattern = validate_inputs(text, pattern)
        naive_matches, naive_iters = naive_string_matching_instrumented(text, pattern)
        rk_matches, rk_iters = rabin_karp_instrumented(text, pattern)

        st.markdown('<div class="section-title">📊 Summary Overview</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        cols[0].metric("Text length (n)", len(text))
        cols[1].metric("Pattern length (m)", len(pattern))
        cols[2].metric("Naive matches", str(naive_matches) if naive_matches else "None")
        cols[3].metric("Rabin–Karp matches", str(rk_matches) if rk_matches else "None")

        st.markdown('<div class="section-title">🔵 Naive Algorithm</div>', unsafe_allow_html=True)
        naive_df = pd.DataFrame(naive_iters)
        st.dataframe(naive_df[["index","matched","comparisons","iteration_time_s","memory_bytes"]]
                     .style.format({"iteration_time_s":"{:.6f}","memory_bytes":"{:,}"}), height=260)
        with st.expander("🔍 Show Detailed Iterations (Naive)", expanded=False):
            for it in naive_iters:
                st.markdown(f"**Index {it['index']} — matched: {it['matched']} — time: {it['iteration_time_s']:.6f}s — mem: {it['memory_bytes']:,} bytes**")
                st.code("\n".join(it["details"]))

        st.markdown('<div class="section-title">🟢 Rabin–Karp Algorithm</div>', unsafe_allow_html=True)
        rk_df = pd.DataFrame(rk_iters)
        st.dataframe(rk_df[["index","phase","matched","comparisons","iteration_time_s","memory_bytes"]]
                     .style.format({"iteration_time_s":"{:.6f}","memory_bytes":"{:,}"}), height=260)
        with st.expander("🔍 Show Detailed Iterations (Rabin–Karp)", expanded=False):
            for it in rk_iters[:60]:
                st.markdown(f"**Index {it['index']} — phase: {it.get('phase','check')} — time: {it['iteration_time_s']:.6f}s — mem: {it['memory_bytes']:,} bytes**")
                st.code("\n".join(it["details"]))

        st.markdown('<div class="section-title">📈 Visual Performance Analysis</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(naive_df, x="index", y="iteration_time_s", title="Naive Runtime per Iteration (s)", color_discrete_sequence=["#2563eb"]), use_container_width=True)
            st.plotly_chart(px.bar(naive_df, x="index", y="comparisons", title="Naive Comparisons", color_discrete_sequence=["#3b82f6"]), use_container_width=True)
        with c2:
            st.plotly_chart(px.line(rk_df, x="index", y="iteration_time_s", title="Rabin–Karp Runtime per Iteration (s)", color_discrete_sequence=["#1d4ed8"]), use_container_width=True)
            st.plotly_chart(px.bar(rk_df, x="index", y="comparisons", title="Rabin–Karp Comparisons", color_discrete_sequence=["#2563eb"]), use_container_width=True)

        total_df = pd.DataFrame({
            "Algorithm": ["Naive", "Rabin–Karp"],
            "Total_time_s": [naive_df["iteration_time_s"].sum(), rk_df["iteration_time_s"].sum()]
        })
        st.plotly_chart(px.pie(total_df, names="Algorithm", values="Total_time_s",
                               color_discrete_sequence=["#2563eb", "#60a5fa"],
                               title="Total Runtime Share"), use_container_width=True)

        st.markdown("""
        <div class="section-title">💡 Insights</div>
        <ul style="color:#000;">
            <li><b>Naive:</b> compares characters at every shift (O(n·m)).</li>
            <li><b>Rabin–Karp:</b> uses rolling hash for O(n+m) average performance.</li>
            <li>Expand each algorithm above to see full iteration details.</li>
        </ul>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# ---------------------- FOOTER ----------------------
st.markdown("<footer>© 2025 Piyusha Supe</footer>", unsafe_allow_html=True)
