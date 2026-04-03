"""
ui/app.py — Streamlit chat frontend for the Life Insurance Support Assistant.

Run with:
    streamlit run frontend/app.py
"""
import uuid
import httpx
import streamlit as st

from components.sidebar import render_sidebar

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Life Insurance Assistant",
    page_icon="🛡️",
    layout="wide",
)

BACKEND_URL = "http://localhost:8000/api/v1/chat"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }
    .stChatMessage { border-radius: 12px; margin-bottom: 0.5rem; }
    .stChatInputContainer { padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar()

# ── Main header ───────────────────────────────────────────────────────────────
st.title("🛡️ Life Insurance Support Assistant")
st.caption("Ask me anything about life insurance policies, eligibility, benefits, or claims.")
st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Handle sidebar suggestion prefill ────────────────────────────────────────
prefill = st.session_state.pop("prefill_message", None)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your question here…") or prefill

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = httpx.post(
                    BACKEND_URL,
                    json={
                        "session_id": st.session_state.session_id,
                        "message": user_input,
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                assistant_reply = data["response"]
            except httpx.ConnectError:
                assistant_reply = (
                    "⚠️ Cannot reach the backend. "
                    "Make sure the FastAPI server is running on `localhost:8000`."
                )
            except Exception as e:
                assistant_reply = f"⚠️ Error: {e}"

        st.markdown(assistant_reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
