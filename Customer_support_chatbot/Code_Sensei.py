# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from datetime import datetime
import json
import re  # Added import for regex

# Set page configuration
st.set_page_config(
    page_title="CodeSensei - Python Programming Assistant",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - UPDATED
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4CAF50;
    }
    .retrieval-badge {
        display: inline-block;
        background-color: #ff9800;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .direct-badge {
        display: inline-block;
        background-color: #2196F3;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* ADDED: Code block styling */
    .code-block-container {
        margin: 1rem 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .code-block-header {
        background-color: #2d3748;
        color: white;
        padding: 0.5rem 1rem;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.9rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .code-block-header span {
        font-weight: bold;
    }
    
    .code-block-content {
        background-color: #1a202c;
        color: #e2e8f0;
        padding: 1rem;
        font-family: 'Consolas', 'Monaco', monospace;
        white-space: pre;
        overflow-x: auto;
        line-height: 1.5;
        margin: 0;
    }
    
    .inline-code {
        background-color: #edf2f7;
        color: #2d3748;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

def initialize_system():
    """Initialize the RAG system"""
    try:
        # Load environment variables
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            st.error("OPENAI_API_KEY not found! Please set it in your .env file.")
            return False
        
        # Import and initialize components (delayed import)
        from langgraph.graph import START, END, StateGraph, MessagesState
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import ToolNode
        from langchain_core.messages import SystemMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_chroma import Chroma
        from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from typing import Literal
        import asyncio
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            api_key=openai_api_key
        )
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key
        )
        
        # Load vector store
        vectorstore = Chroma(
            collection_name="agentic_rag_docs",
            persist_directory="./chroma_db_agentic_rag",
            embedding_function=embeddings
        )
        
        # Create retrieval tool
        @tool
        def retrieve_documents(query: str) -> str:
            """Search for relevant documents in the knowledge base."""
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10}
            )
            results = retriever.invoke(query)
            
            if not results:
                return "No relevant documents found."
            
            formatted = "\n---\n".join([
                f"Source: {r.metadata.get('source', 'Unknown')}\n{r.page_content[:200]}..." 
                for r in results
            ])
            return formatted
        
        # System prompt
        system_prompt = SystemMessage(content="""You are CodeSensei, a Python programming assistant designed to help users learn and understand Python clearly and accurately.

You have access to a knowledge base containing Python programming textbooks.

RETRIEVE FROM THE KNOWLEDGE BASE WHEN:
- The user asks about Python syntax, keywords, or language rules
- The user asks how to use Python functions, classes, or standard libraries
- The user asks "how to" or "what is" questions related to Python programming
- The user requests Python code examples or step-by-step explanations

ANSWER DIRECTLY WITHOUT RETRIEVE WHEN:
- The user greets you or engages in casual conversation
- The question is not related to Python programming
- The question involves simple general knowledge or basic arithmetic

WHEN YOU RETRIEVE:
- Use the document retrieval tool to find relevant passages
- Base your answer strictly on the retrieved content
- Cite the source book title and page or section when available
- Do not invent information not found in the documents

RESPONSE STYLE:
- Be clear, concise, and beginner-friendly
- Explain concepts step by step when needed
- Include short, correct Python code snippets where helpful
""")
        
        # Bind tool to LLM
        tools = [retrieve_documents]
        llm_with_tools = llm.bind_tools(tools)
        
        # Define agent nodes
        def assistant(state: MessagesState) -> dict:
            messages = [system_prompt] + state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return "__end__"
        
        # Build graph
        builder = StateGraph(MessagesState)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            should_continue,
            {"tools": "tools", "__end__": END}
        )
        builder.add_edge("tools", "assistant")
        
        # Add memory and compile
        memory = MemorySaver()
        agent = builder.compile(checkpointer=memory)
        
        # Store in session state
        st.session_state.agent = agent
        st.session_state.system_initialized = True
        st.session_state.vectorstore = vectorstore
        
        return True
        
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return False

def process_query(user_input):
    """Process user query and get response"""
    if not st.session_state.agent:
        return None, False
    
    try:
        result = st.session_state.agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )
        
        # Check if retrieval was used
        used_retrieval = False
        final_answer = ""
        
        for message in result["messages"]:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                used_retrieval = True
            if message.content and not (hasattr(message, 'tool_calls') and message.tool_calls):
                final_answer = message.content
        
        return final_answer, used_retrieval
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None, False

def display_message(message, is_user=False, used_retrieval=False):
    """Display a chat message - UPDATED to fix the duplicate function issue"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        badge = "🔍 Retrieved from knowledge base" if used_retrieval else "💭 Answered directly"
        badge_class = "retrieval-badge" if used_retrieval else "direct-badge"
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 CodeSensei</strong> 
            <span class="{badge_class}">{badge}</span><br>
        </div>
        """, unsafe_allow_html=True)
        
        # Process and display the message with code formatting
        formatted_message = format_message_for_display(message)
        st.markdown(formatted_message, unsafe_allow_html=True)

def format_message_for_display(message):
    """Format message with proper HTML/markdown for display - UPDATED FIXED VERSION"""
    # Clean the message
    message = message.strip()
    
    # Handle code blocks with triple backticks
    def process_code_blocks(match):
        # Extract language and code
        content = match.group(1).strip()
        
        # Check if language is specified
        lines = content.split('\n', 1)
        if len(lines) > 1 and lines[0].strip() in ['python', 'py', '']:
            language = 'python' if lines[0].strip() else ''
            code = lines[1]
        else:
            language = 'python'
            code = content
        
        # Escape HTML special characters in code
        code = (code
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))
        
        # Create code block with syntax highlighting hint
        return f'<div class="code-block-container">' \
               f'<div class="code-block-header">' \
               f'<span>{language if language else "Code"}</span>' \
               f'</div>' \
               f'<div class="code-block-content">{code}</div>' \
               f'</div>'
    
    # First pass: handle ```code``` blocks
    pattern = r'```([\s\S]*?)```'
    formatted = re.sub(pattern, process_code_blocks, message)
    
    # Second pass: handle inline `code`
    formatted = re.sub(r'`([^`\n]+?)`', r'<code class="inline-code">\1</code>', formatted)
    
    # Handle bold text
    formatted = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', formatted)
    
    # Handle lists (basic)
    lines = formatted.split('\n')
    in_list = False
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('- ') or line.startswith('* '):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            content = line[2:].strip()
            formatted_lines.append(f'<li>{content}</li>')
        elif line.startswith('1. '):
            if not in_list:
                formatted_lines.append('<ol>')
                in_list = True
            content = line[3:].strip()
            formatted_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>' if not any(line.startswith(x) for x in ['1. ', '- ', '* ']) else '')
                in_list = False
            if line:
                formatted_lines.append(f'{line}<br>')
    
    if in_list:
        formatted_lines.append('</ul>')
    
    result = ''.join(formatted_lines)
    
    # Remove trailing <br> tags
    result = result.rstrip('<br>')
    
    return result

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5968/5968350.png", width=100)
    st.title("CodeSensei")
    st.markdown("---")
    
    st.markdown("### 📚 Knowledge Base")
    st.info("""
    Contains Python textbooks:
    - Python for Dummies
    - Python for Everybody
    - A Practical Introduction to Programming
    - Introduction to Python Programming
    - Python Programming Notes
    """)
    
    st.markdown("### 🛠️ System Status")
    if st.session_state.system_initialized:
        st.success("✅ System Ready")
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.warning("⚠️ System Not Initialized")
        if st.button("🚀 Initialize System"):
            with st.spinner("Initializing CodeSensei..."):
                if initialize_system():
                    st.success("System initialized successfully!")
                    st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Session Info")
    st.caption(f"Thread ID: {st.session_state.thread_id}")
    st.caption(f"Messages: {len(st.session_state.messages)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📝 Recent Queries")
        for i, (query, _) in enumerate(st.session_state.chat_history[-5:]):
            st.caption(f"{i+1}. {query[:50]}...")

# Main content
st.markdown('<h1 class="main-header">🐍 CodeSensei</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent Python programming assistant powered by Agentic RAG</p>', unsafe_allow_html=True)

# Initialize system if not done
if not st.session_state.system_initialized:
    st.info("👋 Welcome to CodeSensei! Please initialize the system from the sidebar to begin.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📖 Learn Python")
        st.write("Get clear explanations of Python concepts")
    with col2:
        st.markdown("### 💻 Code Examples")
        st.write("Receive practical code snippets")
    with col3:  # Fixed: Changed from col2 to col3
        st.markdown("### 🔍 Smart Retrieval")
        st.write("Answers based on trusted Python textbooks")
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    if st.button("Initialize System Now", type="primary"):
        with st.spinner("Initializing CodeSensei system..."):
            if initialize_system():
                st.success("System initialized successfully!")
                st.rerun()
else:
    # Display chat messages
    for message in st.session_state.messages:
        display_message(
            message['content'],
            is_user=message['role'] == 'user',
            used_retrieval=message.get('retrieval', False)
        )
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "💭 Ask your Python question:",
            placeholder="E.g., 'What is a list comprehension?' or 'How do I use decorators?'",
            height=100,
            key="input"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            submit_button = st.form_submit_button("Send", type="primary")
        with col2:
            example_questions = st.selectbox(
                "Or try an example:",
                ["", 
                 "What is a variable in Python?", 
                 "How do I write a for loop?", 
                 "What's the difference between list and tuple?",
                 "Explain Python decorators",
                 "How to handle exceptions in Python?"]
            )
    
    # Handle form submission
    if submit_button and user_input:
        # Add user message to chat
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Process and get response
        with st.spinner("Thinking..."):
            response, used_retrieval = process_query(user_input)
        
        if response:
            # Add assistant response to chat
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response,
                'retrieval': used_retrieval,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            # Add to chat history
            st.session_state.chat_history.append((user_input, used_retrieval))
            
            # Rerun to display new messages
            st.rerun()
    
    # Handle example question selection
    if example_questions and example_questions != "":
        # Simulate form submission with example question
        st.session_state.messages.append({
            'role': 'user',
            'content': example_questions,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        with st.spinner("Thinking..."):
            response, used_retrieval = process_query(example_questions)
        
        if response:
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response,
                'retrieval': used_retrieval,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            st.session_state.chat_history.append((example_questions, used_retrieval))
            st.rerun()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("💡 Tip: Ask specific questions for better answers")
with col2:
    st.caption("📚 Powered by LangGraph & ChromaDB")
with col3:
    st.caption("🤖 Intelligent retrieval with GPT-4o")