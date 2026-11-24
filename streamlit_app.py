"""HubCore: Cost sharing for energy hubs.

Run locally with:  streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from ui import state, style  # noqa: E402

st.set_page_config(page_title="HubCore · robust cost-sharing for energy hubs", page_icon="⚡",
                   layout="wide", initial_sidebar_state="auto")
state.init()
style.apply()

pages = [
    st.Page("app_pages/home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("app_pages/build.py", title="1 · Build the hub", icon=":material/hub:"),
    st.Page("app_pages/split.py", title="2 · Split the bill", icon=":material/handshake:"),
    st.Page("app_pages/stress.py", title="3 · Stress-test the deal", icon=":material/thunderstorm:"),
    st.Page("app_pages/method.py", title="Method & assumptions", icon=":material/science:"),
]
sections = {"": pages}
if state.featured_applies():
    sections["Featured case"] = [st.Page("app_pages/featured.py", title="Featured: future 26",
                                         icon=":material/flag:", url_path="future-26")]
nav = st.navigation(sections)

with st.sidebar:
    p = state.park()
    st.markdown(f"**Current park**  \n{p.name}")
    st.caption(f"{p.n} firms · " + ", ".join(p.names))
    if not state.is_preset_unchanged():
        st.caption("Edited park or assumptions: results are computed live.")
    st.divider()
    st.caption("HubCore ⚡ Built by Xinyi Mu.")

nav.run()
