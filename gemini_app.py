import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1️⃣ Set page config
st.set_page_config(
    page_title="Data Science Assistant — Gemini (Streamlit)",
    layout="wide"
)

# 2️⃣ Load API key
load_dotenv(".env")
api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')

if not api_key:
    st.error("Gemini API key not found. Set GEMINI_API_KEY in .env")
    st.stop()

genai.configure(api_key=api_key)
MODEL_NAME = "models/gemini-2.0-flash"

# 3️⃣ Sidebar for system prompt and settings
with st.sidebar:
    st.header("System Prompt")
    default_system = (
        "You are a helpful Data Science Assistant. Give concise steps, code in Python "
        "(pandas/sklearn), and clear explanations. If the user uploads data, ask clarifying questions."
    )
    system_prompt = st.text_area("System instruction", value=default_system, height=140)
    st.markdown("---")
    st.markdown("**Model & settings**")
    max_tokens = st.slider("Max output tokens", 64, 2048, 512)
    clear_button = st.button("Reset conversation")

# 4️⃣ Initialize history (user/assistant only)
if 'history' not in st.session_state:
    st.session_state.history = []

if clear_button:
    st.session_state.history = []
    if 'chat' in st.session_state:
        del st.session_state['chat']

# 5️⃣ Initialize chat
if 'chat' not in st.session_state:
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        st.session_state.chat = model.start_chat(history=st.session_state.history)
    except Exception as e:
        st.error(f"Could not start chat: {e}")
        st.stop()
else:
    model = genai.GenerativeModel(MODEL_NAME)
    chat = st.session_state.chat

# 6️⃣ App title
st.title("Data Science Assistant — Gemini (Streamlit)")

# 7️⃣ Conversation UI
col1, col2 = st.columns([3,1])

with col1:
    st.subheader("Conversation")
    for msg in st.session_state.history:
        role = msg.get('role')
        parts = msg.get('parts', [])
        text = "\n".join([p.get('text','') for p in parts]) if parts else ""
        if role == 'user':
            st.markdown(f"**You:** {text}")
        elif role == 'assistant':
            st.markdown(f"**Assistant:** {text}")

    user_input = st.text_area("Your message", value="", key='user_input', height=120)
    send = st.button("Send")

with col2:
    st.subheader("Controls")
    st.write("Conversation length:", len(st.session_state.history))
    if st.button("Show raw history"):
        st.json(st.session_state.history)

# 8️⃣ Handle sending messages
if send and user_input.strip():
    # Prepend system prompt once
    full_user_message = f"{system_prompt}\n\nUser: {user_input.strip()}"
    user_msg = {"role": "user", "parts": [{"text": full_user_message}]}
    st.session_state.history.append(user_msg)

    try:
        chat = st.session_state.chat
        resp = chat.send_message(full_user_message)

        assistant_text = resp.text if hasattr(resp, 'text') else str(resp)
        assistant_msg = {"role": "assistant", "parts": [{"text": assistant_text}]}
        st.session_state.history.append(assistant_msg)

        # No need for experimental_rerun, Streamlit auto-rerenders
    except Exception as e:
        st.error(f"Error sending message: {e}")


# 9️⃣ Footer
st.markdown("---")
st.caption("Uses google.generativeai SDK with Gemini 2.0. Flash model.")
