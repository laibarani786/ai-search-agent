# --- import necessary libraries ---
import streamlit as st 
from langchain_groq import ChatGroq 
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from datetime import datetime
import os

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Search Agent", page_icon="🔍")
st.title("🔍 Search Engine (GEN AI APP) using Tools and Agents")
st.sidebar.title("⚙️ Settings")

# --- Sidebar: API key input ---
api_key = st.sidebar.text_input("Please Enter your Groq API Key: ", type="password")

# --- Sidebar: Tool selection ---
st.sidebar.subheader("🧰 Select Tools")
use_search = st.sidebar.checkbox("DuckDuckGo Search", value=True)
use_arxiv = st.sidebar.checkbox("Arxiv Research Papers", value=True)
use_wiki = st.sidebar.checkbox("Wikipedia", value=True)

# --- Sidebar: Chat control buttons ---
if st.sidebar.button("🆕 New Chat"):
    st.session_state["messages"] = []

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.clear()
    st.session_state["messages"] = []

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am Chatbot who can search the web. How can I help you?"}
    ]

# --- Tools Setup ---
tools = []
if use_search:
    search = DuckDuckGoSearchRun(name="Search")
    tools.append(search)
if use_arxiv:
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    tools.append(arxiv)
if use_wiki:
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    tools.append(wiki)

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input(placeholder="Please write your question here..."):
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        st.stop()

    # ✅ Set API key as environment variable
    os.environ["GROQ_API_KEY"] = api_key

    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": f"{prompt}\n\n🕒 {datetime.now().strftime('%H:%M:%S')}"
    })

    st.chat_message("user").write(prompt)

    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True
    )

    # Initialize Agent
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # ✅ FIX: Run agent on last user message (string), not full list
    user_query = prompt

    # Get and display the response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(user_query, callbacks=[st_cb])

        st.session_state.messages.append({
            'role': 'assistant',
            'content': f"{response}\n\n🕒 {datetime.now().strftime('%H:%M:%S')}"
        })

        st.write(response)
