"""
ui/components/sidebar.py — Streamlit sidebar: session controls + quick questions.
"""
import uuid
import streamlit as st


def render_sidebar() -> None:
    """Render sidebar with session management and suggested questions."""
    with st.sidebar:
        st.title("🛡️ Life Insurance Assistant")
        st.markdown("---")

        # Session management
        st.subheader("💬 Session")
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")

        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        # Suggested questions
        st.subheader("💡 Suggested Questions")
        suggestions = [
            "What types of life insurance are available?",
            "Difference between term and whole life?",
            "Am I eligible at age 55 with diabetes?",
            "How do I file a death claim?",
            "What documents are needed for a claim?",
            "What riders can I add to my policy?",
            "What is the free-look period?",
            "Are policy proceeds tax-exempt?",
        ]
        for suggestion in suggestions:
            if st.button(suggestion, use_container_width=True, key=suggestion):
                st.session_state["prefill_message"] = suggestion

        st.markdown("---")
        st.caption("Powered by Groq · LangGraph · LangChain")
