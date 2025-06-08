import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import uuid
import sqlite3
import tempfile
import os
import shutil

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Try to import enhanced manager, fallback to basic if not available
try:
    from enhanced_langchain_sql_manager import EnhancedLangChainSQLManager, QueryResult

    ENHANCED_AVAILABLE = True
except ImportError:
    print("Enhanced manager not found, using basic imports")
    ENHANCED_AVAILABLE = False
    # Fallback imports
    try:
        from langchain_sql_manager import LangChainSQLManager, QueryResult
    except ImportError:
        print("No SQL manager found!")
        LangChainSQLManager = None
        QueryResult = None

# Try to import memory system components
try:
    from langchain_memory_system import EnhancedSQLConversationMemory, ChatSession, ConversationContext

    MEMORY_AVAILABLE = True
except ImportError:
    print("Memory system not found, creating minimal replacements")
    MEMORY_AVAILABLE = False


    # Create minimal replacements
    @dataclass
    class ConversationContext:
        conversation_id: str
        user_question: str
        sql_query: str
        result_summary: str
        timestamp: datetime
        success: bool
        tables_used: List[str]
        query_type: str
        follow_up_suggestions: List[str]
        model_used: str
        execution_time: float
        session_id: str


    @dataclass
    class ChatSession:
        session_id: str
        session_name: str
        created_at: datetime
        last_active: datetime
        model_name: str
        conversation_count: int
        database_name: Optional[str] = None


    class MinimalMemorySystem:
        def __init__(self):
            self.sessions = []
            self.current_session = None
            self.current_conversations = []

        def get_available_models(self):
            return {
                "gpt-3.5-turbo": "GPT-3.5 Turbo (Fast & Cost-effective)",
                "gpt-4": "GPT-4 (Most Capable)",
                "gpt-4-turbo": "GPT-4 Turbo (Latest)",
                "gpt-4o": "GPT-4o (Omni Model)",
                "gpt-4o-mini": "GPT-4o Mini (Efficient)"
            }

        def create_new_session(self, session_name: str, model_name: str, database_name: str = None):
            session_id = str(uuid.uuid4())
            session = ChatSession(
                session_id=session_id,
                session_name=session_name,
                created_at=datetime.now(),
                last_active=datetime.now(),
                model_name=model_name,
                conversation_count=0,
                database_name=database_name
            )
            self.sessions.append(session)
            self.current_session = session
            return session_id

        def get_all_sessions(self):
            return self.sessions

        def switch_to_session(self, session_id: str):
            for session in self.sessions:
                if session.session_id == session_id:
                    self.current_session = session
                    break

        def delete_session(self, session_id: str):
            self.sessions = [s for s in self.sessions if s.session_id != session_id]
            if self.current_session and self.current_session.session_id == session_id:
                self.current_session = None

        def add_conversation(self, context: ConversationContext):
            self.current_conversations.append(context)
            if self.current_session:
                self.current_session.conversation_count += 1

        def get_session_summary(self):
            if not self.current_conversations:
                return {
                    "total_conversations": 0,
                    "successful_queries": 0,
                    "failed_queries": 0,
                    "success_rate": 0,
                    "most_used_tables": [],
                    "common_query_types": []
                }

            total = len(self.current_conversations)
            successful = sum(1 for c in self.current_conversations if c.success)

            return {
                "total_conversations": total,
                "successful_queries": successful,
                "failed_queries": total - successful,
                "success_rate": (successful / total) * 100 if total > 0 else 0,
                "most_used_tables": [],
                "common_query_types": []
            }

        def clear_session_memory(self):
            self.current_conversations = []

        def export_session_history(self, session_id: str):
            return {"conversations": []}


    EnhancedSQLConversationMemory = MinimalMemorySystem

# Try to import enhanced SQL chains
try:
    from langchain_sql_chains import LangChainSQLChains, AdvancedSQLAnalyzer, SQLMetricsCollector, EnhancedLangChainSQL

    CHAINS_AVAILABLE = True
except ImportError:
    print("SQL chains not found")
    CHAINS_AVAILABLE = False
    LangChainSQLChains = None
    AdvancedSQLAnalyzer = None
    SQLMetricsCollector = None
    EnhancedLangChainSQL = None

# Add conversation memory system
try:
    from langchain_memory_system import chat_with_memory, simple_memory, clear_memory

    CONVERSATION_MEMORY_AVAILABLE = True
    print("Conversation Memory system loaded successfully!")
except ImportError:
    print("Conversation Memory system not found!")
    CONVERSATION_MEMORY_AVAILABLE = False


    # Fallback functions
    def chat_with_memory(user_input, ai_function, system_prompt=""):
        return ai_function(user_input)


    def clear_memory():
        pass


    class simple_memory:
        messages = []

        @staticmethod
        def add_message(user, ai):
            pass

# Page config
st.set_page_config(
    page_title="SQL Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .langchain-badge {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .conversation-memory-badge {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .dataset-selector {
        background: linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .memory-status {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 0.3rem 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
        font-size: 0.9rem;
    }
    .memory-context {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .session-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .session-card:hover {
        background: #e9ecef;
    }
    .session-card.active {
        background: #d4edda;
        border-color: #28a745;
    }
    .conversation-bubble {
        background: #f1f3f4;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4285f4;
    }
    .conversation-bubble.assistant {
        background: #e8f5e8;
        border-left-color: #34a853;
    }
    .sql-query-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .agent-thought {
        background: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 0.5rem;
        margin: 0.25rem 0;
        font-style: italic;
    }
    .model-selector {
        background: linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .session-stats {
        background: linear-gradient(135deg, #20c997 0%, #17a2b8 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .conversation-history {
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
    }
    .clean-chat {
        background: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .dataset-info-box {
        background: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.memory_system = None
    st.session_state.sql_manager = None
    st.session_state.enhanced_sql = None
    st.session_state.current_session_id = None
    st.session_state.selected_model = "gpt-3.5-turbo"
    st.session_state.conversation_memory_enabled = CONVERSATION_MEMORY_AVAILABLE
    # Dataset selection state
    st.session_state.available_datasets = {}
    st.session_state.selected_dataset = None
    st.session_state.last_selected_dataset = None  # Track last selected for comparison
    st.session_state.dataset_change_pending = False


@st.cache_resource
def initialize_systems():
    """Initialize all systems with fallbacks"""

    # Initialize memory system
    if MEMORY_AVAILABLE:
        memory_system = EnhancedSQLConversationMemory()
    else:
        memory_system = MinimalMemorySystem()

    # Initialize SQL manager
    if ENHANCED_AVAILABLE:
        sql_manager = EnhancedLangChainSQLManager()
    elif LangChainSQLManager:
        sql_manager = LangChainSQLManager()
    else:
        st.error("No SQL manager available! Please check your imports.")
        return None, None, None

    # Initialize enhanced SQL if available
    enhanced_sql = None
    if CHAINS_AVAILABLE and sql_manager and sql_manager.db:
        try:
            enhanced_sql = EnhancedLangChainSQL(sql_manager.db, sql_manager.llm)
        except Exception as e:
            print(f"Enhanced SQL init failed: {e}")

    return memory_system, sql_manager, enhanced_sql


def extract_tables_from_query(sql_query: str) -> List[str]:
    """Extract table names from SQL query"""
    if not sql_query:
        return []

    import re
    tables = re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql_query, re.IGNORECASE)
    return list(set(tables))


def classify_query_type(sql_query: str) -> str:
    """Classify the type of SQL query"""
    if not sql_query:
        return "unknown"

    query_upper = sql_query.upper()

    if "GROUP BY" in query_upper or "COUNT" in query_upper or "SUM" in query_upper or "AVG" in query_upper:
        return "aggregation"
    elif "JOIN" in query_upper:
        return "join"
    elif "ORDER BY" in query_upper:
        return "sorting"
    elif "WHERE" in query_upper:
        return "filtered_select"
    else:
        return "simple_select"


def create_new_session():
    """Create a new chat session"""
    st.session_state.show_new_session_dialog = True


def delete_session_callback(session_id: str):
    """Delete a session"""
    if st.session_state.memory_system:
        st.session_state.memory_system.delete_session(session_id)
        if st.session_state.current_session_id == session_id:
            st.session_state.current_session_id = None
        st.rerun()


def switch_session_callback(session_id: str):
    """Switch to a different session"""
    st.session_state.current_session_id = session_id
    if st.session_state.memory_system:
        st.session_state.memory_system.switch_to_session(session_id)


def get_dataset_info(dataset_name: str) -> Dict:
    """Get dataset information safely"""
    if dataset_name in st.session_state.available_datasets:
        return st.session_state.available_datasets[dataset_name]
    return {
        'tables': [],
        'total_rows': 0,
        'table_info': {},
        'db_path': '',
        'created': datetime.now(),
        'source': 'Unknown'
    }


def switch_dataset(new_dataset: str):
    """Safely switch to a new dataset"""
    if new_dataset == st.session_state.selected_dataset:
        return True  # Already selected

    try:
        # Get dataset info
        dataset_info = get_dataset_info(new_dataset)
        if not dataset_info['db_path']:
            st.error(f"Dataset {new_dataset} has invalid database path")
            return False

        # Clear memory for dataset switch
        if CONVERSATION_MEMORY_AVAILABLE:
            clear_memory()

        # Disconnect current database cleanly
        if st.session_state.sql_manager and hasattr(st.session_state.sql_manager, 'db'):
            try:
                if st.session_state.sql_manager.db:
                    st.session_state.sql_manager.db.dispose()
            except Exception as e:
                print(f"Warning: Error disposing old database: {e}")

        # Update SQL manager database
        st.session_state.sql_manager.database_url = dataset_info['db_path']

        # Reconnect to new database
        success = st.session_state.sql_manager._connect()

        if success:
            # Clear enhanced SQL context if available
            if ENHANCED_AVAILABLE and hasattr(st.session_state.sql_manager, 'clear_context'):
                st.session_state.sql_manager.clear_context()

            # Update selected dataset
            st.session_state.last_selected_dataset = st.session_state.selected_dataset
            st.session_state.selected_dataset = new_dataset

            print(f"Successfully switched to dataset: {new_dataset}")
            return True
        else:
            st.error(f"Failed to connect to dataset: {new_dataset}")
            return False

    except Exception as e:
        st.error(f"Error switching dataset: {str(e)}")
        return False


def display_conversation_history(conversations: List[ConversationContext]):
    """Display conversation history"""
    st.markdown('<div class="conversation-history">', unsafe_allow_html=True)

    for i, conv in enumerate(conversations):
        # User message
        st.markdown(f"""
        <div class="conversation-bubble">
            <strong>User ({conv.timestamp.strftime('%H:%M:%S')}):</strong><br>
            {conv.user_question}
        </div>
        """, unsafe_allow_html=True)

        # Assistant response
        status_icon = "✓" if conv.success else "✗"
        st.markdown(f"""
        <div class="conversation-bubble assistant">
            <strong>Assistant ({status_icon}):</strong><br>
            {conv.result_summary[:200]}{'...' if len(conv.result_summary) > 200 else ''}
        </div>
        """, unsafe_allow_html=True)

        # Show SQL if available
        if conv.sql_query:
            with st.expander(f"SQL Query #{i + 1}"):
                st.code(conv.sql_query, language="sql")

        # Show execution details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Execution Time", f"{conv.execution_time:.2f}s")
        with col2:
            st.metric("Tables Used", len(conv.tables_used))
        with col3:
            st.metric("Query Type", conv.query_type)

    st.markdown('</div>', unsafe_allow_html=True)


def display_memory_context():
    """Display memory context"""
    if hasattr(st.session_state.sql_manager,
               'conversation_context') and st.session_state.sql_manager.conversation_context:
        st.markdown('<div class="memory-context">', unsafe_allow_html=True)
        st.subheader("Memory Context")

        if hasattr(st.session_state.sql_manager, 'get_conversation_summary'):
            context_summary = st.session_state.sql_manager.get_conversation_summary()
            st.write(context_summary)

        # Show last context in detail
        last_context = st.session_state.sql_manager.conversation_context[-1]
        if last_context.get('key_entities'):
            with st.expander("Last Query Entities"):
                st.json(last_context['key_entities'])

        st.markdown('</div>', unsafe_allow_html=True)


def display_session_sidebar():
    """Display session management in sidebar"""
    with st.sidebar:
        st.header("Chat Sessions")

        # Dataset selector
        st.markdown('<div class="dataset-selector">', unsafe_allow_html=True)
        st.subheader("Select Dataset")

        # Show available datasets
        if st.session_state.available_datasets:
            dataset_options = list(st.session_state.available_datasets.keys())

            # Ensure current selection is valid
            if st.session_state.selected_dataset not in dataset_options:
                if dataset_options:
                    st.session_state.selected_dataset = dataset_options[0]
                else:
                    st.session_state.selected_dataset = None

            if dataset_options:
                current_index = 0
                if st.session_state.selected_dataset in dataset_options:
                    current_index = dataset_options.index(st.session_state.selected_dataset)

                selected_dataset = st.selectbox(
                    "Choose Dataset:",
                    options=dataset_options,
                    index=current_index,
                    key="dataset_selector_main",
                    help="Select which dataset to query"
                )

                # Check if dataset actually changed
                if selected_dataset != st.session_state.selected_dataset:
                    with st.spinner(f"Switching to {selected_dataset}..."):
                        success = switch_dataset(selected_dataset)
                        if success:
                            st.success(f"Switched to {selected_dataset}")
                            st.rerun()
                        else:
                            # Revert selection if switch failed
                            st.session_state.selected_dataset = st.session_state.last_selected_dataset

                # Show current dataset info
                if st.session_state.selected_dataset:
                    dataset_info = get_dataset_info(st.session_state.selected_dataset)
                    st.markdown('<div class="dataset-info-box">', unsafe_allow_html=True)
                    st.caption(f"Tables: {', '.join(dataset_info['tables'])}")
                    st.caption(f"Total Rows: {dataset_info['total_rows']:,}")
                    st.caption(f"Created: {dataset_info['created'].strftime('%m/%d %H:%M')}")
                    st.caption(f"Source: {dataset_info['source']}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No datasets available")
        else:
            st.info("No datasets available. Upload data in Database Setup.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Model selector
        st.markdown('<div class="model-selector">', unsafe_allow_html=True)
        st.subheader("Select AI Model")
        available_models = st.session_state.memory_system.get_available_models()

        selected_model = st.selectbox(
            "Choose Model:",
            options=list(available_models.keys()),
            format_func=lambda x: available_models[x],
            index=list(available_models.keys()).index(st.session_state.selected_model),
            key="model_selector"
        )
        st.session_state.selected_model = selected_model
        st.markdown('</div>', unsafe_allow_html=True)

        # New session button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Chat", use_container_width=True):
                create_new_session()
        with col2:
            if st.button("Refresh", use_container_width=True):
                st.rerun()

        # New session dialog
        if st.session_state.get('show_new_session_dialog', False):
            with st.container():
                st.subheader("Create New Session")

                session_name = st.text_input(
                    "Session Name:",
                    placeholder="e.g., NBA Analysis, Employee Data Review",
                    key="new_session_name"
                )

                database_name = st.text_input(
                    "Database Name (optional):",
                    placeholder="e.g., nba_db, sales_data",
                    key="new_session_db"
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Create", type="primary"):
                        if session_name:
                            session_id = st.session_state.memory_system.create_new_session(
                                session_name=session_name,
                                model_name=selected_model,
                                database_name=database_name or None
                            )
                            st.session_state.current_session_id = session_id
                            st.session_state.show_new_session_dialog = False
                            st.success(f"Created session: {session_name}")
                            st.rerun()
                        else:
                            st.error("Please enter a session name")

                with col2:
                    if st.button("Cancel"):
                        st.session_state.show_new_session_dialog = False
                        st.rerun()

        # Session list
        st.subheader("Recent Sessions")
        sessions = st.session_state.memory_system.get_all_sessions()

        if not sessions:
            st.info("No sessions yet. Create your first session!")
        else:
            for session in sessions[:10]:  # Show last 10 sessions
                is_active = st.session_state.current_session_id == session.session_id

                # Session card
                session_class = "session-card active" if is_active else "session-card"

                with st.container():
                    st.markdown(f'<div class="{session_class}">', unsafe_allow_html=True)

                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        if st.button(
                                f"{session.session_name}",
                                key=f"session_{session.session_id}",
                                use_container_width=True
                        ):
                            switch_session_callback(session.session_id)
                            st.rerun()

                    with col2:
                        st.text(f"{session.conversation_count}")

                    with col3:
                        if st.button("Delete", key=f"delete_{session.session_id}"):
                            delete_session_callback(session.session_id)

                    # Session details
                    st.caption(f"Model: {session.model_name}")
                    st.caption(f"Active: {session.last_active.strftime('%m/%d %H:%M')}")
                    if session.database_name:
                        st.caption(f"Database: {session.database_name}")

                    st.markdown('</div>', unsafe_allow_html=True)

        # Session stats
        if st.session_state.current_session_id:
            st.markdown('<div class="session-stats">', unsafe_allow_html=True)
            st.subheader("Current Session")

            try:
                session_summary = st.session_state.memory_system.get_session_summary()
                if session_summary and "error" not in session_summary:
                    col1, col2 = st.columns(2)
                    with col1:
                        conversations_count = session_summary.get('total_conversations', 0)
                        st.metric("Conversations", conversations_count)
                    with col2:
                        success_rate = session_summary.get('success_rate', 0)
                        st.metric("Success Rate", f"{success_rate:.1f}%")

                    most_used_tables = session_summary.get('most_used_tables', [])
                    if most_used_tables:
                        st.write("Top Tables:")
                        for table, count in most_used_tables[:3]:
                            st.write(f"• {table}: {count}")
                else:
                    st.info("No session data available yet.")
            except Exception as e:
                st.warning(f"Could not load session stats: {str(e)}")

            st.markdown('</div>', unsafe_allow_html=True)


def universal_chat_interface():
    """Universal clean chat interface"""
    st.subheader("SQL Assistant")

    # Dataset selection check
    if not st.session_state.selected_dataset:
        st.error(
            "No dataset selected. Please select a dataset from the sidebar or upload data in 'Database Setup' tab.")
        return

    # Check database connection
    if not st.session_state.sql_manager or not st.session_state.sql_manager.db:
        st.error("No database connected. Please setup a database first in the 'Database Setup' tab.")
        return

    # Clean chat container
    st.markdown('<div class="clean-chat">', unsafe_allow_html=True)

    # Current dataset info + memory status in one compact line
    col1, col2 = st.columns([2, 1])

    with col1:
        if st.session_state.selected_dataset:
            dataset_info = get_dataset_info(st.session_state.selected_dataset)
            st.markdown(
                f"**Active Dataset:** {st.session_state.selected_dataset} ({', '.join(dataset_info['tables'])})")

    with col2:
        if CONVERSATION_MEMORY_AVAILABLE:
            try:
                memory_count = len(simple_memory.messages) if hasattr(simple_memory, 'messages') else 0
            except:
                memory_count = 0
            st.markdown(f"""
            <div class="memory-status">
                Memory: {memory_count} messages
            </div>
            """, unsafe_allow_html=True)

    # Memory controls (only show if there are messages)
    if CONVERSATION_MEMORY_AVAILABLE and memory_count > 0:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("Show Memory", use_container_width=True):
                with st.expander("Recent Conversation Memory", expanded=True):
                    for i, msg in enumerate(simple_memory.messages[-3:], 1):
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 8px; margin: 4px 0; border-radius: 4px; border-left: 3px solid #007bff;">
                        <strong>{i}. You:</strong> {msg['user']}<br>
                        <strong>AI:</strong> {msg['ai'][:120]}{'...' if len(msg['ai']) > 120 else ''}
                        </div>
                        """, unsafe_allow_html=True)

        with col3:
            if st.button("Clear Memory", use_container_width=True):
                clear_memory()
                st.success("Memory cleared!")
                st.rerun()

    # Main chat input - Clean and simple
    chat_query = st.text_area(
        "Ask your question:",
        height=100,
        placeholder=f"Ask questions about your {st.session_state.selected_dataset} data...\n\nWith memory enabled, you can use references like 'his', 'her', 'that person' in follow-up questions!",
        key="main_chat_input",
        help="Type your question in natural language. The AI will generate SQL and return results."
    )

    if st.button("Submit Query", type="primary", use_container_width=True):
        if chat_query:
            with st.spinner("Processing your query..."):

                # Universal memory enhanced query function
                if CONVERSATION_MEMORY_AVAILABLE:
                    def universal_memory_enhanced_query(enhanced_prompt):
                        print(f"Enhanced Prompt Received: {enhanced_prompt[:200]}...")

                        # Extract real user query from enhanced prompt
                        if "NEW USER MESSAGE:" in enhanced_prompt:
                            user_question = enhanced_prompt.split("NEW USER MESSAGE:")[-1].split("RESPONSE:")[0].strip()
                        else:
                            user_question = enhanced_prompt

                        print(f"Extracted User Question: {user_question}")

                        # Universal reference words check
                        user_lower = user_question.lower()
                        reference_words = ['his', 'her', 'he', 'she', 'this person', 'that person', 'this player',
                                           'that player', 'this employee', 'that employee', 'this company',
                                           'that company',
                                           'bu kişi', 'bu oyuncu', 'bu çalışan', 'bu şirket', 'onun', 'o', 'bunun']

                        has_reference = any(word in user_lower for word in reference_words)

                        if has_reference and simple_memory.messages:
                            # Extract entity name from last AI response
                            last_ai_response = simple_memory.messages[-1]['ai']
                            print(f"Last AI Response: {last_ai_response}")

                            # Universal entity pattern
                            import re
                            # Name patterns - universal
                            name_patterns = [
                                r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',
                                r'\b([A-Z][a-z]+)\s+is\s',
                                r'person\s+([A-Z][a-z]+ [A-Z][a-z]+)',
                                r'employee\s+([A-Z][a-z]+ [A-Z][a-z]+)',
                                r'player\s+([A-Z][a-z]+ [A-Z][a-z]+)',
                                r'company\s+([A-Z][a-z]+ [A-Z][a-z]*)',
                                r'([A-Z][a-z]+ [A-Z][a-z]+)\s+\(',
                            ]

                            found_entity = None
                            for pattern in name_patterns:
                                matches = re.findall(pattern, last_ai_response)
                                if matches:
                                    found_entity = matches[0]
                                    print(f"Found Entity Name: {found_entity}")
                                    break

                            if found_entity:
                                # Create enhanced query with entity name
                                if 'salary' in user_lower or 'maaş' in user_lower or 'wage' in user_lower:
                                    enhanced_query = f"What is {found_entity}'s salary or earnings?"
                                elif 'team' in user_lower or 'takım' in user_lower:
                                    enhanced_query = f"Which team does {found_entity} play for or belong to?"
                                elif 'company' in user_lower or 'şirket' in user_lower:
                                    enhanced_query = f"Which company does {found_entity} work for?"
                                elif 'department' in user_lower or 'departman' in user_lower:
                                    enhanced_query = f"Which department does {found_entity} work in?"
                                elif 'height' in user_lower or 'boy' in user_lower:
                                    enhanced_query = f"What is {found_entity}'s height?"
                                elif 'age' in user_lower or 'yaş' in user_lower:
                                    enhanced_query = f"What is {found_entity}'s age?"
                                else:
                                    enhanced_query = f"Show me information about {found_entity}"

                                print(f"Enhanced Query: {enhanced_query}")

                                # Send enhanced query to SQL Manager
                                result = st.session_state.sql_manager.query_natural_language(enhanced_query)
                                return result

                        # Normal query if no reference
                        print(f"Normal Query: {user_question}")
                        return st.session_state.sql_manager.query_natural_language(user_question)

                    # Run query with conversation memory
                    result = chat_with_memory(
                        chat_query,
                        universal_memory_enhanced_query,
                        f"""You are a universal database assistant with conversation memory for {st.session_state.selected_dataset} dataset.

Memory-based Reference Resolution:
1. Remember previous conversations about people, players, employees, companies
2. When user says 'his/her salary', 'he/she', 'that person/player/employee' - refer to the last mentioned entity
3. Provide specific information about ONLY that entity
4. Never show all records when a specific entity is referenced

IMPORTANT: Use exact entity names in SQL queries with WHERE clause.

Current dataset: {st.session_state.selected_dataset}
Available tables: {', '.join(get_dataset_info(st.session_state.selected_dataset)['tables']) if st.session_state.selected_dataset else 'None'}
"""
                    )

                    # If result is string, wrap it in QueryResult
                    if isinstance(result, str):
                        result_obj = type('QueryResult', (), {
                            'success': True,
                            'result_data': pd.DataFrame(),
                            'sql_query': "",
                            'explanation': result,
                            'error_message': "",
                            'execution_time': 0.0,
                            'agent_thoughts': []
                        })()
                        result = result_obj

                else:
                    # Normal query (without memory)
                    result = st.session_state.sql_manager.query_natural_language(chat_query)

                # Create conversation context
                context = ConversationContext(
                    conversation_id=str(uuid.uuid4()),
                    user_question=chat_query,
                    sql_query=result.sql_query if hasattr(result, 'sql_query') else "",
                    result_summary=result.explanation if hasattr(result, 'explanation') else str(result),
                    timestamp=datetime.now(),
                    success=result.success if hasattr(result, 'success') else True,
                    tables_used=extract_tables_from_query(result.sql_query if hasattr(result, 'sql_query') else ""),
                    query_type=classify_query_type(result.sql_query if hasattr(result, 'sql_query') else ""),
                    follow_up_suggestions=[],
                    model_used=st.session_state.memory_system.current_session.model_name if hasattr(
                        st.session_state.memory_system,
                        'current_session') and st.session_state.memory_system.current_session else "gpt-3.5-turbo",
                    execution_time=result.execution_time if hasattr(result, 'execution_time') else 0.0,
                    session_id=st.session_state.current_session_id
                )

                # Add to memory
                st.session_state.memory_system.add_conversation(context)

                # Display result
                if hasattr(result, 'success') and result.success:
                    success_msg = "Query executed successfully!"
                    if CONVERSATION_MEMORY_AVAILABLE:
                        memory_count = len(simple_memory.messages)
                        success_msg += f" (Memory: {memory_count} messages)"
                    st.success(success_msg)

                    # Results display
                    if hasattr(result,
                               'result_data') and result.result_data is not None and not result.result_data.empty:
                        st.subheader("Query Results")
                        st.dataframe(result.result_data, use_container_width=True)

                        # Download option
                        csv = result.result_data.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv,
                            file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                    # Show SQL
                    if hasattr(result, 'sql_query') and result.sql_query:
                        st.subheader("Generated SQL")
                        st.code(result.sql_query, language="sql")

                    # Show explanation
                    if hasattr(result, 'explanation') and result.explanation:
                        st.subheader("Analysis")
                        st.info(result.explanation)

                    # Agent thoughts
                    if hasattr(result, 'agent_thoughts') and result.agent_thoughts:
                        with st.expander("AI Reasoning Process"):
                            for i, thought in enumerate(result.agent_thoughts, 1):
                                st.markdown(f'<div class="agent-thought">{i}. {thought}</div>',
                                            unsafe_allow_html=True)
                else:
                    error_msg = result.error_message if hasattr(result, 'error_message') else "Unknown error"
                    st.error(f"Query failed: {error_msg}")

    st.markdown('</div>', unsafe_allow_html=True)


def create_unique_dataset_name() -> str:
    """Create a unique dataset name"""
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    return f"Dataset_{timestamp}"


def cleanup_temp_files():
    """Clean up old temporary database files"""
    try:
        temp_dir = tempfile.gettempdir()
        for file in os.listdir(temp_dir):
            if file.endswith('.db') and file.startswith('tmp'):
                file_path = os.path.join(temp_dir, file)
                # Remove files older than 1 hour
                if os.path.getmtime(file_path) < (datetime.now().timestamp() - 3600):
                    os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not clean temp files: {e}")


def handle_csv_upload(uploaded_csvs, table_names, current_session):
    """Handle CSV upload with improved dataset registration"""
    if st.button("Create Database from CSV", type="primary"):
        try:
            # Clean up old temp files first
            cleanup_temp_files()

            # Create temporary database with unique name
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db', prefix='dataset_')
            temp_db_path = temp_db.name
            temp_db.close()

            conn = sqlite3.connect(temp_db_path)

            # Process each CSV
            total_rows = 0
            table_info = {}
            processed_tables = []

            for file, table_name in zip(uploaded_csvs, table_names):
                # Validate table name
                if not table_name or not table_name.replace('_', '').isalnum():
                    st.error(f"Invalid table name: {table_name}")
                    conn.close()
                    os.unlink(temp_db_path)
                    return

                # Read and process CSV
                df = pd.read_csv(file)

                # Clean column names
                df.columns = [col.lower().replace(' ', '_').replace('-', '_').replace('.', '_')
                              for col in df.columns]

                # Write to database
                df.to_sql(table_name, conn, index=False, if_exists='replace')

                rows_count = len(df)
                total_rows += rows_count
                table_info[table_name] = rows_count
                processed_tables.append(table_name)

                st.write(f"✓ {file.name} → {table_name} ({rows_count:,} rows)")

            conn.close()

            # Create unique dataset name
            dataset_name = create_unique_dataset_name()

            # Register dataset with comprehensive info
            st.session_state.available_datasets[dataset_name] = {
                'db_path': f"sqlite:///{temp_db_path}",
                'tables': processed_tables,
                'total_rows': total_rows,
                'table_info': table_info,
                'created': datetime.now(),
                'source': 'CSV Upload',
                'file_names': [f.name for f in uploaded_csvs],
                'temp_file_path': temp_db_path  # Keep track for cleanup
            }

            # Switch to new dataset
            success = switch_dataset(dataset_name)

            if success:
                # Update current session database name
                if current_session:
                    current_session.database_name = dataset_name

                st.success(f"Created dataset '{dataset_name}' with {total_rows:,} total rows!")
                st.success("Dataset automatically selected and ready to use!")

                if CONVERSATION_MEMORY_AVAILABLE:
                    st.success("Memory cleared for new dataset!")

                # Show dataset summary
                with st.expander("Dataset Summary", expanded=True):
                    for table, rows in table_info.items():
                        st.write(f"• **{table}**: {rows:,} rows")

                st.rerun()
            else:
                # Clean up on failure
                del st.session_state.available_datasets[dataset_name]
                os.unlink(temp_db_path)
                st.error("Failed to connect to created database")

        except Exception as e:
            st.error(f"CSV conversion failed: {str(e)}")
            # Clean up on error
            try:
                if 'temp_db_path' in locals():
                    os.unlink(temp_db_path)
            except:
                pass


def delete_dataset(dataset_name: str):
    """Safely delete a dataset"""
    try:
        if dataset_name in st.session_state.available_datasets:
            dataset_info = st.session_state.available_datasets[dataset_name]

            # Clean up temp file if exists
            if 'temp_file_path' in dataset_info:
                try:
                    os.unlink(dataset_info['temp_file_path'])
                except:
                    pass

            # Remove from available datasets
            del st.session_state.available_datasets[dataset_name]

            # If this was the selected dataset, clear selection
            if st.session_state.selected_dataset == dataset_name:
                st.session_state.selected_dataset = None

                # Clear memory
                if CONVERSATION_MEMORY_AVAILABLE:
                    clear_memory()

            return True
    except Exception as e:
        st.error(f"Error deleting dataset: {str(e)}")
        return False


def main():
    # Header
    st.markdown('<h1 class="main-header">SQL Assistant</h1>', unsafe_allow_html=True)

    # Enhanced badge with conversation memory
    if CONVERSATION_MEMORY_AVAILABLE:
        badge_text = "Enhanced with Universal Memory + Multi-Dataset Support"
        st.markdown(f'<div class="conversation-memory-badge">{badge_text}</div>', unsafe_allow_html=True)
    else:
        badge_text = "SQL Assistant with Multi-Dataset Support"
        st.markdown(f'<div class="langchain-badge">{badge_text}</div>', unsafe_allow_html=True)

    # Show component status
    if not ENHANCED_AVAILABLE or not MEMORY_AVAILABLE or not CONVERSATION_MEMORY_AVAILABLE:
        with st.expander("Component Status"):
            st.write(f"Enhanced Manager: {'✓' if ENHANCED_AVAILABLE else '✗'}")
            st.write(f"Memory System: {'✓' if MEMORY_AVAILABLE else '✗'}")
            st.write(f"SQL Chains: {'✓' if CHAINS_AVAILABLE else '✗'}")
            st.write(f"**Conversation Memory: {'✓ ACTIVE' if CONVERSATION_MEMORY_AVAILABLE else '✗ NOT AVAILABLE'}**")

    # Initialize systems
    if not st.session_state.initialized:
        with st.spinner("Initializing SQL System..."):
            memory_system, sql_manager, enhanced_sql = initialize_systems()

            if memory_system is None:
                st.error("Failed to initialize systems!")
                return

            st.session_state.memory_system = memory_system
            st.session_state.sql_manager = sql_manager
            st.session_state.enhanced_sql = enhanced_sql
            st.session_state.initialized = True

    # Display session sidebar
    display_session_sidebar()

    # Main content
    if not st.session_state.current_session_id:
        # Welcome screen
        st.markdown("## Welcome to  SQL Assistant!")

        features = [
            "Multi-Dataset Support: Upload and switch between different datasets",
            "Session Management: Create multiple chat sessions",
            "Model Selection: Choose between different GPT models",
            "Enhanced Analytics: Track your SQL query performance",
            "Search History: Find previous conversations easily"
        ]

        if CONVERSATION_MEMORY_AVAILABLE:
            features.insert(1, "Universal Memory: AI remembers context across any dataset")
            features.append("Smart References: Use 'his salary', 'that person', 'this company' naturally")

        st.markdown("### Features:")
        for feature in features:
            st.markdown(f"- **{feature}**")

        # Quick start
        st.markdown("### Quick Start:")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **1. Create Session**
            - Click "New Chat"
            - Give it a name
            - Select AI model
            """)

        with col2:
            st.markdown("""
            **2. Upload Dataset**
            - Go to 'Database Setup' tab
            - Upload CSV files (NBA, employees, sales, etc.)
            - Or create sample data
            """)

        with col3:
            st.markdown("""
            **3. Start Querying**
            - Select dataset from sidebar
            - Ask questions naturally
            - Use references like "his", "that company"
            """)

        # Create first session shortcut
        st.markdown("---")
        st.markdown("### Create Your First Session:")

        col1, col2 = st.columns([2, 1])
        with col1:
            first_session_name = st.text_input(
                "Session Name:",
                placeholder="My Data Analysis Session",
                key="first_session_input"
            )
        with col2:
            if st.button("Start Analyzing", type="primary", use_container_width=True):
                if first_session_name:
                    session_id = st.session_state.memory_system.create_new_session(
                        session_name=first_session_name,
                        model_name=st.session_state.selected_model
                    )
                    st.session_state.current_session_id = session_id
                    st.rerun()
                else:
                    st.error("Please enter a session name")

        return

    # Active session interface
    current_session = None
    sessions = st.session_state.memory_system.get_all_sessions()
    for session in sessions:
        if session.session_id == st.session_state.current_session_id:
            current_session = session
            break

    if not current_session:
        st.error("Session not found")
        st.session_state.current_session_id = None
        st.rerun()
        return

    # Session header
    st.markdown(f"## {current_session.session_name}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", current_session.model_name)
    with col2:
        st.metric("Conversations", current_session.conversation_count)
    with col3:
        st.metric("Created", current_session.created_at.strftime('%m/%d/%Y'))
    with col4:
        dataset_count = len(st.session_state.available_datasets)
        st.metric("Datasets", dataset_count)

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Chat",
        "History",
        "Database Setup",
        "Analytics",
        "Settings"
    ])

    with tab1:
        universal_chat_interface()

    with tab2:
        st.subheader("Conversation History")

        # Get current session conversations
        conversations = st.session_state.memory_system.current_conversations

        if not conversations:
            st.info("No conversations yet in this session. Start chatting in the Chat tab!")
        else:
            # Search conversations
            search_query = st.text_input("Search conversations:", placeholder="Search by question, SQL, or result...")

            if search_query:
                # Filter conversations
                filtered_conversations = [
                    conv for conv in conversations
                    if search_query.lower() in conv.user_question.lower() or
                       search_query.lower() in conv.sql_query.lower() or
                       search_query.lower() in conv.result_summary.lower()
                ]
                st.write(f"Found {len(filtered_conversations)} matching conversations")
                display_conversation_history(filtered_conversations)
            else:
                # Show all conversations
                display_conversation_history(conversations)

            # Export option
            if st.button("Export Session History"):
                if hasattr(st.session_state.memory_system, 'export_session_history'):
                    session_data = st.session_state.memory_system.export_session_history(
                        st.session_state.current_session_id)
                    if "error" not in session_data:
                        json_str = json.dumps(session_data, indent=2, default=str)
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"session_history_{current_session.session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

    with tab3:
        st.subheader("Database & Dataset Setup")

        # Current datasets overview with improved management
        if st.session_state.available_datasets:
            st.subheader("Available Datasets")

            # Dataset management controls
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Total Datasets:** {len(st.session_state.available_datasets)}")
            with col2:
                if st.button("Cleanup All", help="Remove all datasets"):
                    if st.button("Confirm Delete All", type="secondary"):
                        # Clean up all temp files
                        for dataset_info in st.session_state.available_datasets.values():
                            if 'temp_file_path' in dataset_info:
                                try:
                                    os.unlink(dataset_info['temp_file_path'])
                                except:
                                    pass

                        st.session_state.available_datasets.clear()
                        st.session_state.selected_dataset = None
                        if CONVERSATION_MEMORY_AVAILABLE:
                            clear_memory()
                        st.success("All datasets removed!")
                        st.rerun()

            # Display datasets
            for name, info in st.session_state.available_datasets.items():
                is_selected = name == st.session_state.selected_dataset
                card_style = "border: 2px solid #28a745;" if is_selected else "border: 1px solid #dee2e6;"

                with st.container():
                    st.markdown(
                        f'<div style="{card_style} border-radius: 8px; padding: 1rem; margin: 0.5rem 0; background: {"#d4edda" if is_selected else "#f8f9fa"};">',
                        unsafe_allow_html=True)

                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        status_icon = "✓" if is_selected else "○"
                        st.markdown(f"### {status_icon} {name}")
                        st.write(f"**Tables:** {', '.join(info['tables'])}")
                        st.write(f"**Total Rows:** {info['total_rows']:,}")
                        st.write(f"**Source:** {info['source']}")

                    with col2:
                        st.write(f"**Created:**")
                        st.write(info['created'].strftime('%Y-%m-%d'))
                        st.write(info['created'].strftime('%H:%M:%S'))

                        if 'file_names' in info:
                            st.write(f"**Files:** {len(info['file_names'])}")

                    with col3:
                        if not is_selected:
                            if st.button(f"Select", key=f"select_{name}"):
                                success = switch_dataset(name)
                                if success:
                                    st.rerun()
                        else:
                            st.markdown("**✓ Active**")

                        if st.button(f"Delete", key=f"delete_{name}", type="secondary"):
                            if delete_dataset(name):
                                st.success(f"Deleted {name}")
                                st.rerun()

                    # Detailed table info
                    with st.expander(f"Table Details - {name}"):
                        for table, rows in info['table_info'].items():
                            st.write(f"• **{table}**: {rows:,} rows")

                        if 'file_names' in info:
                            st.write("**Source Files:**")
                            for fname in info['file_names']:
                                st.write(f"• {fname}")

                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No datasets available. Upload CSV files below to get started.")

        # Database status
        st.subheader("Current Database Connection")
        if st.session_state.sql_manager and st.session_state.sql_manager.db:
            col1, col2 = st.columns(2)
            with col1:
                st.success("Database Connected")
            with col2:
                if st.session_state.selected_dataset:
                    st.info(f"Active: {st.session_state.selected_dataset}")

            # Database info
            if hasattr(st.session_state.sql_manager, 'get_database_info'):
                try:
                    db_info = st.session_state.sql_manager.get_database_info()
                    if "tables" in db_info:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Tables", len(db_info['tables']))
                        with col2:
                            st.metric("Dialect", db_info.get('dialect', 'Unknown'))

                        # Show tables with sample data
                        with st.expander("Database Schema"):
                            for table in db_info["tables"]:
                                st.write(f"• **{table}**")
                                if table in db_info.get("samples", {}):
                                    sample_data = db_info["samples"][table]
                                    if sample_data:
                                        st.json(sample_data[0])
                except Exception as e:
                    st.warning(f"Could not load database info: {str(e)}")
        else:
            st.warning("No database connected")

        # CSV Upload section - improved
        st.markdown("---")
        st.subheader("Upload New Dataset")

        uploaded_csvs = st.file_uploader(
            "Choose CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload multiple CSV files to create a new dataset. Each file will become a table."
        )

        if uploaded_csvs:
            st.success(f"{len(uploaded_csvs)} CSV file(s) uploaded")

            # Preview uploaded files
            with st.expander("File Preview", expanded=True):
                for i, file in enumerate(uploaded_csvs):
                    try:
                        df_preview = pd.read_csv(file, nrows=3)
                        st.write(f"**{file.name}** ({file.size:,} bytes)")
                        st.write(f"Columns: {', '.join(df_preview.columns)}")
                        st.dataframe(df_preview, use_container_width=True)
                        file.seek(0)  # Reset file pointer
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {str(e)}")

            # Table name configuration
            st.subheader("Configure Table Names")
            table_names = []
            for i, file in enumerate(uploaded_csvs):
                default_name = file.name.replace('.csv', '').lower().replace(' ', '_').replace('-', '_')
                table_name = st.text_input(
                    f"Table name for {file.name}:",
                    value=default_name,
                    key=f"table_name_{i}",
                    help="Use only letters, numbers and underscores"
                )
                table_names.append(table_name)

            # Validate table names
            valid_names = True
            name_errors = []

            for name in table_names:
                if not name:
                    name_errors.append("Empty table name not allowed")
                    valid_names = False
                elif not name.replace('_', '').isalnum():
                    name_errors.append(f"'{name}' contains invalid characters")
                    valid_names = False
                elif table_names.count(name) > 1:
                    name_errors.append(f"Duplicate table name: '{name}'")
                    valid_names = False

            if name_errors:
                for error in name_errors:
                    st.error(f"{error}")

            # Upload button
            if valid_names:
                handle_csv_upload(uploaded_csvs, table_names, current_session)
            else:
                st.warning("Please fix table name errors before proceeding")

        # Sample database option
        st.markdown("---")
        st.subheader("Create Sample Dataset")
        st.write("Create a sample dataset with companies, employees, projects, and sales data for testing.")

        if st.button("Create Sample Dataset", type="secondary"):
            with st.spinner("Creating sample dataset..."):
                try:
                    # Create sample database
                    if hasattr(st.session_state.sql_manager, 'create_sample_database'):
                        success = st.session_state.sql_manager.create_sample_database()

                        if success:
                            # Register sample dataset
                            dataset_name = f"Sample_Data_{datetime.now().strftime('%m%d_%H%M%S')}"
                            st.session_state.available_datasets[dataset_name] = {
                                'db_path': st.session_state.sql_manager.database_url,
                                'tables': ['companies', 'employees', 'projects', 'sales'],
                                'total_rows': 400,  # Approximate
                                'table_info': {'companies': 10, 'employees': 100, 'projects': 50, 'sales': 240},
                                'created': datetime.now(),
                                'source': 'Sample Data'
                            }

                            # Switch to sample dataset
                            success = switch_dataset(dataset_name)

                            if success:
                                # Update session database name
                                current_session.database_name = dataset_name

                                st.success("Sample dataset created and activated!")
                                st.success("Includes: Companies, Employees, Projects, Sales data")
                                st.rerun()
                            else:
                                st.error("Created sample data but failed to switch to it")
                        else:
                            st.error("Failed to create sample dataset")
                    else:
                        st.error("Sample dataset creation not available")
                except Exception as e:
                    st.error(f"Error creating sample dataset: {str(e)}")

    with tab4:
        st.subheader("Session Analytics")

        try:
            session_summary = st.session_state.memory_system.get_session_summary()
            if session_summary and "error" not in session_summary:
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_conversations = session_summary.get('total_conversations', 0)
                    st.metric("Total Queries", total_conversations)
                with col2:
                    success_rate = session_summary.get('success_rate', 0)
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                with col3:
                    successful_queries = session_summary.get('successful_queries', 0)
                    st.metric("Successful", successful_queries)
                with col4:
                    failed_queries = session_summary.get('failed_queries', 0)
                    st.metric("Failed", failed_queries)

                # Memory stats
                if CONVERSATION_MEMORY_AVAILABLE:
                    st.markdown("### Memory Stats")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        try:
                            memory_count = len(simple_memory.messages)
                        except:
                            memory_count = 0
                        st.metric("Remembered Messages", memory_count)
                    with col2:
                        dataset_count = len(st.session_state.available_datasets)
                        st.metric("Available Datasets", dataset_count)
                    with col3:
                        active_dataset = st.session_state.selected_dataset or "None"
                        st.metric("Active Dataset", active_dataset)

                # Charts (only if we have data and plotly is available)
                if total_conversations > 0:
                    # Query types chart
                    common_query_types = session_summary.get('common_query_types', [])
                    if common_query_types:
                        st.subheader("Query Types Distribution")
                        query_types_df = pd.DataFrame(
                            [(qtype, count) for qtype, count in common_query_types],
                            columns=['Query Type', 'Count']
                        )
                        fig = px.pie(query_types_df, values='Count', names='Query Type', title="Query Types")
                        st.plotly_chart(fig, use_container_width=True)

                    # Table usage chart
                    most_used_tables = session_summary.get('most_used_tables', [])
                    if most_used_tables:
                        st.subheader("Most Used Tables")
                        tables_df = pd.DataFrame(
                            [(table, count) for table, count in most_used_tables],
                            columns=['Table', 'Usage Count']
                        )
                        fig = px.bar(tables_df, x='Table', y='Usage Count', title="Table Usage")
                        st.plotly_chart(fig, use_container_width=True)

                    # Timeline
                    conversations = st.session_state.memory_system.current_conversations
                    if conversations:
                        st.subheader("Query Timeline")
                        timeline_df = pd.DataFrame([
                            {
                                'Time': conv.timestamp,
                                'Success': conv.success,
                                'Execution Time': conv.execution_time,
                                'Query': conv.user_question[:50] + '...' if len(
                                    conv.user_question) > 50 else conv.user_question
                            }
                            for conv in conversations
                        ])

                        fig = px.scatter(
                            timeline_df,
                            x='Time',
                            y='Execution Time',
                            color='Success',
                            hover_data=['Query'],
                            title="Query Performance Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Dataset analytics
                if st.session_state.available_datasets:
                    st.subheader("Dataset Analytics")

                    # Dataset usage stats
                    dataset_stats = []
                    for name, info in st.session_state.available_datasets.items():
                        dataset_stats.append({
                            'Dataset': name,
                            'Tables': len(info['tables']),
                            'Total Rows': info['total_rows'],
                            'Created': info['created'],
                            'Active': name == st.session_state.selected_dataset
                        })

                    if dataset_stats:
                        df_stats = pd.DataFrame(dataset_stats)

                        # Dataset size comparison
                        fig = px.bar(
                            df_stats,
                            x='Dataset',
                            y='Total Rows',
                            color='Active',
                            title="Dataset Size Comparison"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Dataset table count
                        fig2 = px.pie(
                            df_stats,
                            values='Tables',
                            names='Dataset',
                            title="Tables per Dataset"
                        )
                        st.plotly_chart(fig2, use_container_width=True)

            else:
                st.info("No analytics data available yet. Start some conversations to see analytics!")
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
            st.info("This might be because the memory system is not fully initialized.")

    with tab5:
        st.subheader("Session Settings")

        # Session info
        st.markdown("### Session Information")

        # Edit session name
        new_session_name = st.text_input(
            "Session Name:",
            value=current_session.session_name,
            key="edit_session_name"
        )

        if st.button("Update Session Name"):
            current_session.session_name = new_session_name
            st.success("Session name updated!")
            st.rerun()

        # Model info
        st.info(f"**Current Model:** {current_session.model_name}")
        st.info(f"**Created:** {current_session.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        st.info(f"**Last Active:** {current_session.last_active.strftime('%Y-%m-%d %H:%M:%S')}")

        # Memory settings
        st.markdown("### Memory Settings")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clear Session Memory", use_container_width=True):
                # Clear enhanced context
                if ENHANCED_AVAILABLE and hasattr(st.session_state.sql_manager, 'clear_context'):
                    st.session_state.sql_manager.clear_context()

                # Clear session conversations
                if hasattr(st.session_state.memory_system, 'clear_session_memory'):
                    st.session_state.memory_system.clear_session_memory()

                # Clear conversation memory
                if CONVERSATION_MEMORY_AVAILABLE:
                    clear_memory()

                st.success("Session memory cleared!")
                st.rerun()

        with col2:
            if st.button("Reset Database Connection", use_container_width=True):
                if st.session_state.selected_dataset:
                    success = switch_dataset(st.session_state.selected_dataset)
                    if success:
                        st.success("Database connection reset!")
                    else:
                        st.error("Failed to reset connection")
                else:
                    st.warning("No dataset selected")

        # Dataset settings
        st.markdown("### Dataset Settings")

        dataset_count = len(st.session_state.available_datasets)
        st.info(f"**Available Datasets:** {dataset_count}")
        st.info(f"**Active Dataset:** {st.session_state.selected_dataset or 'None'}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clear All Datasets", use_container_width=True, type="secondary"):
                st.warning("This will permanently delete all datasets!")

        with col2:
            if st.button("Cleanup Temp Files", use_container_width=True):
                try:
                    cleanup_temp_files()
                    st.success("Temporary files cleaned!")
                except Exception as e:
                    st.error(f"Cleanup failed: {str(e)}")

        # Confirm dataset deletion
        if st.session_state.get('confirm_delete_all_datasets', False):
            st.error("**CONFIRM: Delete all datasets?**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete All", type="primary"):
                    # Clean up all datasets
                    for dataset_info in st.session_state.available_datasets.values():
                        if 'temp_file_path' in dataset_info:
                            try:
                                os.unlink(dataset_info['temp_file_path'])
                            except:
                                pass

                    st.session_state.available_datasets.clear()
                    st.session_state.selected_dataset = None
                    st.session_state.confirm_delete_all_datasets = False

                    if CONVERSATION_MEMORY_AVAILABLE:
                        clear_memory()

                    st.success("All datasets deleted!")
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state.confirm_delete_all_datasets = False
                    st.rerun()
        else:
            if dataset_count > 0:
                if st.button("Confirm Delete All Datasets"):
                    st.session_state.confirm_delete_all_datasets = True
                    st.rerun()

        # System info
        st.markdown("### System Information")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Enhanced Memory:** {'✓ Enabled' if ENHANCED_AVAILABLE else '✗ Disabled'}")
            st.info(f"**Memory System:** {'✓ Full' if MEMORY_AVAILABLE else '⚠ Basic'}")
        with col2:
            st.info(f"**SQL Chains:** {'✓ Available' if CHAINS_AVAILABLE else '✗ Not Available'}")
            st.info(f"**Universal Memory:** {'✓ ACTIVE' if CONVERSATION_MEMORY_AVAILABLE else '✗ NOT AVAILABLE'}")

        # Database connection info
        if st.session_state.sql_manager:
            st.info(f"**Database URL:** {st.session_state.sql_manager.database_url or 'None'}")
            st.info(f"**Connected:** {'✓ Yes' if st.session_state.sql_manager.db else '✗ No'}")

        # Danger zone
        st.markdown("### Danger Zone")

        if st.button("Delete This Session", type="secondary"):
            st.warning("This will permanently delete this session and all its conversations!")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm Delete", type="primary"):
                    delete_session_callback(st.session_state.current_session_id)
            with col2:
                if st.button("Cancel"):
                    st.rerun()


if __name__ == "__main__":
    main()