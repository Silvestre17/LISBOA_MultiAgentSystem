# ==========================================================================
# Master Thesis - Lisbon Urban Assistant (Streamlit App)
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Main Streamlit application for the intelligent tourist assistant.
#   Provides a modern, intuitive chat interface for exploring Lisbon.
# 
#   Features:
#     - Real-time chat with LLM-powered assistant
#     - Weather and transport quick actions
#     - Multiple LLM provider support
#     - Session state management
#     - Responsive design
# 
#   Usage:
#     streamlit run app.py
#     streamlit run app.py -- --provider google
# ==========================================================================

# Required libraries:
# pip install streamlit langchain langgraph langchain-groq python-dotenv

# IMPORTANT: Load environment variables FIRST (before any LangChain imports)
# This ensures LangSmith tracing is enabled from the start
from dotenv import load_dotenv
load_dotenv()

# Suppress Torch/Streamlit file watcher warning (known compatibility issue)
import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import streamlit as st
import sys
import time
import os
from datetime import datetime
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, ".")

from agent.graph import create_assistant, LisbonAssistant
from config import Config


# ==========================================================================
# Page Configuration
# ==========================================================================

# Configure Streamlit page settings (must be first Streamlit command)
st.set_page_config(
    page_title="Lisbon Urban Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Silvestre17/Thesis2025-26_AFGS",
        "Report a bug": "https://github.com/Silvestre17/Thesis2025-26_AFGS/issues",
        "About": """
        # Lisbon Urban Assistant
        
        **Master Thesis Project**  
        André Filipe Gomes Silvestre, 2025
        
        An intelligent assistant for tourists and locals in Lisbon, 
        providing real-time information about weather, transport, 
        events, and points of interest.
        """
    }
)


# ==========================================================================
# Custom CSS Styling
# ==========================================================================

def apply_custom_css():
    """
    Applies custom CSS styling to the Streamlit app.
    
    Customizes:
        - Chat message styling
        - Button appearance
        - Sidebar formatting
        - General layout improvements
    """
    st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .app-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    
    .app-header h1 {
        margin: 0;
        font-size: 2rem;
        color: white;
    }
    
    .app-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Chat container styling */
    .chat-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #fafafa;
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* User message styling */
    .user-message {
        background-color: #e3f2fd;
        padding: 0.75rem 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        text-align: right;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #f5f5f5;
        padding: 0.75rem 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
    }
    
    /* Quick action buttons */
    .quick-action-btn {
        width: 100%;
        margin: 0.25rem 0;
    }
    
    /* Status cards */
    .status-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    
    .status-card h4 {
        margin: 0 0 0.5rem 0;
        color: #333;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        padding: 1rem;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #888;
        font-size: 0.8rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        text-align: center;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================================================
# Session State Initialization
# ==========================================================================

def initialize_session_state():
    """
    Initializes Streamlit session state variables.
    
    Session State Variables:
        - messages: List of chat messages (role, content)
        - assistant: LisbonAssistant instance
        - provider: Current LLM provider name
        - initialized: Boolean flag for initialization status
        - error: Last error message (if any)
    """
    # Initialize messages list for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize assistant instance placeholder
    if "assistant" not in st.session_state:
        st.session_state.assistant = None
    
    # Initialize current provider name
    if "provider" not in st.session_state:
        st.session_state.provider = Config.MODEL_PROVIDER
    
    # Initialize flag to track successful initialization
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    
    # Initialize error message placeholder
    if "error" not in st.session_state:
        st.session_state.error = None


def initialize_assistant(provider: str) -> bool:
    """
    Initializes or reinitializes the LisbonAssistant.
    
    Args:
        provider (str): The LLM provider to use (groq, google, openai, etc.)
    
    Returns:
        bool: True if initialization was successful, False otherwise.
    
    Side Effects:
        - Updates st.session_state.assistant with new instance
        - Updates st.session_state.provider with current provider
        - Updates st.session_state.initialized status
        - Updates st.session_state.error if initialization fails
    """
    try:
        # Create new assistant instance with specified provider
        st.session_state.assistant = create_assistant(provider)
        st.session_state.provider = provider
        st.session_state.initialized = True
        st.session_state.error = None
        return True
    except Exception as e:
        # Store error message for display
        st.session_state.error = str(e)
        st.session_state.initialized = False
        return False


# ==========================================================================
# UI Components
# ==========================================================================

def render_header():
    """
    Renders the application header with title and description.
    
    Displays:
        - App icon and title
        - Brief description
        - Current date/time
    """
    st.markdown("""
    <div class="app-header">
        <h1>🏛️ Lisbon Urban Assistant</h1>
        <p>Your intelligent guide to exploring Lisbon</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """
    Renders the sidebar with settings and quick actions.
    
    Sidebar Sections:
        1. Provider Settings: LLM provider selection
        2. Quick Actions: Pre-defined query buttons
        3. Session Info: Current session statistics
        4. About: Project information
    
    Returns:
        str: Selected LLM provider name.
    """
    with st.sidebar:
        # ===================== Provider Settings =====================
        st.header("⚙️ Settings")
        
        # Provider selection dropdown
        provider_options = ["groq", "google", "openai", "lmstudio", "ollama"]
        current_index = provider_options.index(st.session_state.provider) if st.session_state.provider in provider_options else 0
        
        selected_provider = st.selectbox(
            "LLM Provider",
            options=provider_options,
            index=current_index,
            help="Select the AI model provider. Groq is recommended for best performance."
        )
        
        # Provider-specific information
        provider_info = {
            "groq": "🚀 Fast inference with Qwen model",
            "google": "🔮 Google's Gemini models",
            "openai": "🤖 OpenAI GPT models",
            "lmstudio": "💻 Local LM Studio server",
            "ollama": "🦙 Local Ollama models"
        }
        st.caption(provider_info.get(selected_provider, ""))
        
        # Reinitialize button (only if provider changed or not initialized)
        if selected_provider != st.session_state.provider or not st.session_state.initialized:
            if st.button("🔄 Apply Changes", use_container_width=True):
                with st.spinner("Initializing assistant..."):
                    if initialize_assistant(selected_provider):
                        st.success("✅ Assistant ready!")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed: {st.session_state.error}")
        
        st.divider()
        
        # ===================== Quick Actions =====================
        st.header("⚡ Quick Actions")
        
        # Weather quick action
        if st.button("🌤️ Weather Summary", use_container_width=True, key="btn_weather"):
            return selected_provider, "What's the current weather in Lisbon? Include any active warnings."
        
        # Transport quick action
        if st.button("🚇 Transport Status", use_container_width=True, key="btn_transport"):
            return selected_provider, "What's the current status of public transport in Lisbon? Include Metro, buses, and trains."
        
        # Events quick action
        if st.button("🎭 Upcoming Events", use_container_width=True, key="btn_events"):
            return selected_provider, "What cultural events are happening in Lisbon this week?"
        
        # Points of Interest quick action
        if st.button("📍 Top Attractions", use_container_width=True, key="btn_poi"):
            return selected_provider, "What are the must-see tourist attractions in Lisbon?"
        
        # Trip planning quick action
        if st.button("🗺️ Plan My Day", use_container_width=True, key="btn_plan"):
            return selected_provider, "Help me plan a one-day trip in Lisbon. I'm interested in history and good food."
        
        st.divider()
        
        # ===================== Session Info =====================
        st.header("📊 Session Info")
        
        # Display session statistics in columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            status = "🟢" if st.session_state.initialized else "🔴"
            st.metric("Status", status)
        
        # Display model information if initialized
        if st.session_state.initialized and st.session_state.assistant:
            st.caption(f"Model: {st.session_state.assistant.model_name}")
            st.caption(f"Session: {st.session_state.assistant.state['session_id'][:8]}...")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.assistant:
                st.session_state.assistant.reset()
            st.rerun()
        
        st.divider()
        
        # ===================== About Section =====================
        st.header("ℹ️ About")
        st.markdown("""
        **Master Thesis Project**  
        NOVA IMS, 2025
        
        *LLM-Powered Urban Exploration:  
        A Framework for Adaptive Tourist  
        and Mobility Itinerary Planning*
        
        By André Filipe Gomes Silvestre
        """)
        
        # GitHub link
        st.markdown("[📂 View on GitHub](https://github.com/Silvestre17/Thesis2025-26_AFGS)")
        
        st.divider()
        
        # ===================== LangSmith Tracing =====================
        st.header("📈 Tracing")
        langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        langsmith_project = os.getenv("LANGCHAIN_PROJECT", "default")
        
        if langsmith_enabled:
            st.success("🟢 LangSmith Active")
            st.caption(f"Project: `{langsmith_project}`")
            st.markdown("[🔗 Open LangSmith](https://smith.langchain.com)")
        else:
            st.warning("🔴 LangSmith Disabled")
            st.caption("Set LANGCHAIN_TRACING_V2=true in .env")
    
    return selected_provider, None


def render_chat_messages():
    """
    Renders the chat message history.
    
    Displays all messages in st.session_state.messages using
    Streamlit's native chat_message component for consistent styling.
    
    Message Format:
        - role: "user" or "assistant"
        - content: Message text content
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_example_queries():
    """
    Renders example query suggestions for new users.
    
    Displays a grid of clickable example queries when the chat
    history is empty, helping users understand what they can ask.
    
    Returns:
        str or None: Selected example query, or None if none selected.
    """
    st.markdown("### 💡 Try asking about...")
    
    # Define example queries in a 2x3 grid
    examples = [
        ("🌤️", "Weather", "What's the weather forecast for the next 3 days in Lisbon?"),
        ("🚇", "Metro", "Is the Lisbon metro running normally today?"),
        ("🎭", "Events", "What cultural events are happening this weekend?"),
        ("🏥", "Services", "Find pharmacies and hospitals near Rossio"),
        ("🍽️", "Food", "Recommend traditional Portuguese restaurants in Alfama"),
        ("🗺️", "Planning", "Plan a 2-day itinerary for a first-time visitor to Lisbon")
    ]
    
    # Create columns for the grid layout
    cols = st.columns(3)
    
    selected_query = None
    for i, (icon, label, query) in enumerate(examples):
        with cols[i % 3]:
            if st.button(f"{icon} {label}", key=f"example_{i}", use_container_width=True):
                selected_query = query
    
    return selected_query


def render_error_panel():
    """
    Renders an error panel when assistant initialization fails.
    
    Displays:
        - Error message
        - Troubleshooting tips
        - Links to documentation
    """
    st.error("⚠️ Assistant Not Initialized")
    
    with st.expander("🔧 Troubleshooting", expanded=True):
        st.markdown("""
        **Common Issues:**
        
        1. **Missing API Key**
           - Ensure `GROQ_API_KEY` is set in your `.env` file
           - For Google: Set `GOOGLE_API_KEY`
           - For OpenAI: Set `OPENAI_API_KEY`
        
        2. **Local Models (LM Studio / Ollama)**
           - Ensure the server is running
           - LM Studio: Default port 1234
           - Ollama: Run `ollama serve`
        
        3. **Network Issues**
           - Check your internet connection
           - Verify firewall settings
        """)
        
        if st.session_state.error:
            st.code(st.session_state.error, language="text")
    
    # Retry button
    if st.button("🔄 Retry Initialization", use_container_width=True):
        with st.spinner("Initializing..."):
            if initialize_assistant(st.session_state.provider):
                st.rerun()


def process_user_input(user_input: str):
    """
    Processes user input and generates assistant response.
    
    Args:
        user_input (str): The user's message/query.
    
    Flow:
        1. Add user message to chat history
        2. Display user message
        3. Generate assistant response with streaming indicator
        4. Add assistant response to chat history
        5. Display assistant response
    """
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        # Show thinking indicator
        with st.spinner("🤔 Thinking..."):
            try:
                # Get response from assistant
                response = st.session_state.assistant.chat(user_input)
                
                # Display response with typing effect placeholder
                message_placeholder = st.empty()
                message_placeholder.markdown(response)
                
                # Add to message history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                # Handle errors gracefully with specific messages
                error_str = str(e).lower()
                
                if "401" in error_str or "unauthorized" in error_str:
                    error_msg = """❌ **API Key Error (401 Unauthorized)**
                    
Your API key is invalid, expired, or revoked. Please:
1. Go to [Groq Console](https://console.groq.com/keys) and create a new API key
2. Update your `.env` file with `GROQ_API_KEY=your_new_key`
3. Restart the app with `streamlit run app.py`

Or switch to a different provider in the sidebar."""
                elif "rate" in error_str or "limit" in error_str:
                    error_msg = """⚠️ **Rate Limit Exceeded**
                    
You've exceeded the API rate limit. Please wait a moment and try again, 
or switch to a different provider in the sidebar."""
                elif "timeout" in error_str or "connection" in error_str:
                    error_msg = """⚠️ **Connection Error**
                    
Could not connect to the API. Please check your internet connection and try again."""
                else:
                    error_msg = f"❌ **Error:** {str(e)}"
                
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please check the error message above."})


def render_footer():
    """
    Renders the application footer.
    
    Displays:
        - Copyright information
        - Version number
        - Current timestamp
    """
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("🏛️ Lisbon Urban Assistant v1.0")
    
    with col2:
        st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    with col3:
        st.caption("Made with ❤️ at NOVA IMS")


# ==========================================================================
# Main Application
# ==========================================================================

def main():
    """
    Main application entry point.
    
    Orchestrates the Streamlit app by:
        1. Applying custom CSS styles
        2. Initializing session state
        3. Rendering UI components
        4. Handling user interactions
        5. Processing chat messages
    """
    # Apply custom styling
    apply_custom_css()
    
    # Initialize session state variables
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar and get any quick action query
    selected_provider, quick_action_query = render_sidebar()
    
    # Main content area
    main_container = st.container()
    
    with main_container:
        # Initialize assistant on first run
        if not st.session_state.initialized:
            with st.spinner("🚀 Starting Lisbon Urban Assistant..."):
                if not initialize_assistant(selected_provider):
                    render_error_panel()
                    return
        
        # Check if assistant is ready
        if not st.session_state.assistant:
            render_error_panel()
            return
        
        # Render existing chat messages
        render_chat_messages()
        
        # Show example queries if chat is empty
        example_query = None
        if not st.session_state.messages:
            st.markdown("### 👋 Welcome to Lisbon!")
            st.markdown("""
            I'm your intelligent assistant for exploring Lisbon, Portugal. 
            I can help you with:
            
            - 🌤️ **Weather** - Current conditions and forecasts
            - 🚇 **Transport** - Metro, bus, and train status
            - 🎭 **Events** - Cultural events and activities
            - 📍 **Places** - Points of interest and services
            - 🗺️ **Planning** - Personalized itineraries
            
            Ask me anything about Lisbon!
            """)
            
            example_query = render_example_queries()
        
        # Handle quick action from sidebar
        if quick_action_query:
            process_user_input(quick_action_query)
            st.rerun()
        
        # Handle example query selection
        if example_query:
            process_user_input(example_query)
            st.rerun()
    
    # Chat input at the bottom
    if user_input := st.chat_input("Ask me about Lisbon...", key="chat_input"):
        process_user_input(user_input)
        st.rerun()
    
    # Render footer
    render_footer()


# ==========================================================================
# Entry Point
# ==========================================================================

if __name__ == "__main__":
    main()
