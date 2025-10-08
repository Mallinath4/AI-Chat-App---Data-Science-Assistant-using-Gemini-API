import streamlit as st
import google.generativeai as genai
import os

# Read API key from file or environment
KEY_PATH = "keys/.gemini.txt"
api_key = None
if os.path.exists(KEY_PATH):
    with open(KEY_PATH, 'r') as f:
        api_key = f.read().strip()
else:
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')

if not api_key:
    st.error("Gemini API key not found. Put it in keys/.gemini.txt or set GEMINI_API_KEY env var.")
    st.stop()

# Configure the genai client
genai.configure(api_key=api_key)

# Create or reuse model
MODEL_NAME = "gemini-pro"
# You can pass a system_instruction when creating the GenerativeModel or include it in chat history.
model = genai.GenerativeModel(MODEL_NAME)

st.set_page_config(page_title="Data Science Assistant — Gemini (Streamlit)", layout="wide")
st.title("Data Science Assistant — Gemini (Streamlit)")

# Sidebar: system prompt
with st.sidebar:
    st.header("System Prompt")
    default_system = "You are a helpful Data Science Assistant. Give concise steps, code in Python (pandas/sklearn), and clear explanations. If the user uploads data, ask clarifying Qs."
    system_prompt = st.text_area("System instruction (system prompt)", value=default_system, height=140)
    st.markdown("---")
    st.markdown("**Model & settings**")
    model_name = st.text_input("Model name", value=MODEL_NAME)
    max_tokens = st.slider("Max output tokens", min_value=64, max_value=2048, value=512)
    clear_button = st.button("Reset conversation")

# Initialize session state
if 'history' not in st.session_state:
    # chat history as list of dicts compatible with generative SDK chat history format
    st.session_state.history = [
        {"role": "system", "parts": [{"type": "text", "text": system_prompt}]}]

if clear_button:
    st.session_state.history = [{"role": "system", "parts": [{"type": "text", "text": system_prompt}]}]
    if 'chat' in st.session_state:
        del st.session_state['chat']

# Ensure model variable reflects chosen model
if model_name != MODEL_NAME:
    model = genai.GenerativeModel(model_name)

# Start chat session if not present
if 'chat' not in st.session_state:
    try:
        st.session_state.chat = model.start_chat(history=st.session_state.history)
    except Exception as e:
        st.error(f"Could not start chat: {e}")
        st.stop()

# Main UI: conversation
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Conversation")
    for msg in st.session_state.history:
        role = msg.get('role', 'user')
        text = ''
        parts = msg.get('parts', [])
        if parts:
            # Join text parts
            text = '\n'.join([p.get('text','') for p in parts if p.get('type') == 'text'])
        if role == 'user':
            st.markdown(f"**You:** {text}")
        elif role == 'assistant' or role == 'model':
            st.markdown(f"**Assistant:** {text}")
        else:
            st.markdown(f"*{role}*: {text}")

    user_input = st.text_area("Your message", value="", key='user_input', height=120)
    send = st.button("Send")

with col2:
    st.subheader("Controls")
    st.write("Conversation length: ", len(st.session_state.history))
    if st.button("Show raw history"):
        st.json(st.session_state.history)

# Handle sending
if send and user_input.strip():
    # Append user message to history in the SDK format
    user_message = {"role": "user", "parts": [{"type": "text", "text": user_input.strip()}]}
    st.session_state.history.append(user_message)

    # If chat object exists, use it to send message and capture response
    try:
        chat = st.session_state.chat
        resp = chat.send_message(user_input.strip())
        # SDK typically stores the last response at chat.last
        # Some SDK responses are objects with .text
        assistant_text = None
        # try common attributes
        if hasattr(resp, 'text'):
            assistant_text = resp.text
        elif hasattr(resp, 'last') and hasattr(resp.last, 'text'):
            assistant_text = resp.last.text
        elif hasattr(chat, 'last') and hasattr(chat.last, 'text'):
            assistant_text = chat.last.text
        else:
            assistant_text = str(resp)

        assistant_message = {"role": "assistant", "parts": [{"type": "text", "text": assistant_text}]}
        st.session_state.history.append(assistant_message)

        # Rerun so UI updates
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error sending message: {e}")

# Footer / Notes
st.markdown("---")
st.caption("This demo uses google.generativeai Python SDK. Ensure you have `pip install google-genai streamlit` and a valid Gemini API key.")
