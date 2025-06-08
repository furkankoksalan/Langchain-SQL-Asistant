import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Float, Integer, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy import desc, asc

# LangChain Memory imports
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory import ConversationBufferWindowMemory, ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI

Base = declarative_base()


class ChatSessionModel(Base):
    """SQLAlchemy model for chat sessions"""
    __tablename__ = 'chat_sessions'

    session_id = Column(String, primary_key=True)
    session_name = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    last_active = Column(DateTime, nullable=False)
    model_name = Column(String, nullable=False)
    conversation_count = Column(Integer, default=0)
    database_name = Column(String, nullable=True)

    conversations = relationship("ConversationModel", back_populates="session", cascade="all, delete-orphan")


class ConversationModel(Base):
    """SQLAlchemy model for conversations"""
    __tablename__ = 'conversations'

    conversation_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('chat_sessions.session_id'), nullable=False)
    user_question = Column(Text, nullable=False)
    sql_query = Column(Text, nullable=True)
    result_summary = Column(Text, nullable=True)
    timestamp = Column(DateTime, nullable=False)
    success = Column(Boolean, nullable=False)
    tables_used = Column(JSON, nullable=True)
    query_type = Column(String, nullable=True)
    follow_up_suggestions = Column(JSON, nullable=True)
    model_used = Column(String, nullable=True)
    execution_time = Column(Float, nullable=True)

    session = relationship("ChatSessionModel", back_populates="conversations")


@dataclass
class ConversationContext:
    """Enhanced conversation context for SQL queries"""
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

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    @classmethod
    def from_model(cls, model: ConversationModel):
        """Create from SQLAlchemy model"""
        return cls(
            conversation_id=model.conversation_id,
            session_id=model.session_id,
            user_question=model.user_question,
            sql_query=model.sql_query or "",
            result_summary=model.result_summary or "",
            timestamp=model.timestamp,
            success=model.success,
            tables_used=model.tables_used or [],
            query_type=model.query_type or "",
            follow_up_suggestions=model.follow_up_suggestions or [],
            model_used=model.model_used or "",
            execution_time=model.execution_time or 0.0
        )


@dataclass
class ChatSession:
    """Chat session information"""
    session_id: str
    session_name: str
    created_at: datetime
    last_active: datetime
    model_name: str
    conversation_count: int
    database_name: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_active'] = self.last_active.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_active'] = datetime.fromisoformat(data['last_active'])
        return cls(**data)

    @classmethod
    def from_model(cls, model: ChatSessionModel):
        """Create from SQLAlchemy model"""
        return cls(
            session_id=model.session_id,
            session_name=model.session_name,
            created_at=model.created_at,
            last_active=model.last_active,
            model_name=model.model_name,
            conversation_count=model.conversation_count,
            database_name=model.database_name
        )


class ConversationDatabase:
    """SQLAlchemy database for storing conversations"""

    def __init__(self, db_url: str = "sqlite:///conversations.db"):
        self.db_url = db_url
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.init_database()

    def init_database(self):
        """Initialize the conversations database"""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def save_session(self, session: ChatSession):
        """Save or update a chat session"""
        db_session = self.get_session()
        try:
            existing = db_session.query(ChatSessionModel).filter(
                ChatSessionModel.session_id == session.session_id
            ).first()

            if existing:
                existing.session_name = session.session_name
                existing.last_active = session.last_active
                existing.model_name = session.model_name
                existing.conversation_count = session.conversation_count
                existing.database_name = session.database_name
            else:
                db_session.add(ChatSessionModel(
                    session_id=session.session_id,
                    session_name=session.session_name,
                    created_at=session.created_at,
                    last_active=session.last_active,
                    model_name=session.model_name,
                    conversation_count=session.conversation_count,
                    database_name=session.database_name
                ))

            db_session.commit()
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()

    def save_conversation(self, context: ConversationContext):
        """Save a conversation"""
        db_session = self.get_session()
        try:
            existing = db_session.query(ConversationModel).filter(
                ConversationModel.conversation_id == context.conversation_id
            ).first()

            if existing:
                existing.user_question = context.user_question
                existing.sql_query = context.sql_query
                existing.result_summary = context.result_summary
                existing.timestamp = context.timestamp
                existing.success = context.success
                existing.tables_used = context.tables_used
                existing.query_type = context.query_type
                existing.follow_up_suggestions = context.follow_up_suggestions
                existing.model_used = context.model_used
                existing.execution_time = context.execution_time
            else:
                db_session.add(ConversationModel(
                    conversation_id=context.conversation_id,
                    session_id=context.session_id,
                    user_question=context.user_question,
                    sql_query=context.sql_query,
                    result_summary=context.result_summary,
                    timestamp=context.timestamp,
                    success=context.success,
                    tables_used=context.tables_used,
                    query_type=context.query_type,
                    follow_up_suggestions=context.follow_up_suggestions,
                    model_used=context.model_used,
                    execution_time=context.execution_time
                ))

            db_session.commit()
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()

    def get_sessions(self, limit: int = 50) -> List[ChatSession]:
        """Get all chat sessions"""
        db_session = self.get_session()
        try:
            sessions = db_session.query(ChatSessionModel).order_by(
                desc(ChatSessionModel.last_active)
            ).limit(limit).all()

            return [ChatSession.from_model(session) for session in sessions]
        finally:
            db_session.close()

    def get_conversations(self, session_id: str) -> List[ConversationContext]:
        """Get conversations for a session"""
        db_session = self.get_session()
        try:
            conversations = db_session.query(ConversationModel).filter(
                ConversationModel.session_id == session_id
            ).order_by(asc(ConversationModel.timestamp)).all()

            return [ConversationContext.from_model(conv) for conv in conversations]
        finally:
            db_session.close()

    def delete_session(self, session_id: str):
        """Delete a session and all its conversations"""
        db_session = self.get_session()
        try:
            db_session.query(ConversationModel).filter(
                ConversationModel.session_id == session_id
            ).delete()

            db_session.query(ChatSessionModel).filter(
                ChatSessionModel.session_id == session_id
            ).delete()

            db_session.commit()
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()

    def update_session_activity(self, session_id: str):
        """Update session last activity time"""
        db_session = self.get_session()
        try:
            session = db_session.query(ChatSessionModel).filter(
                ChatSessionModel.session_id == session_id
            ).first()

            if session:
                session.last_active = datetime.now()
                session.conversation_count += 1
                db_session.commit()
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()

    def get_session_by_id(self, session_id: str) -> Optional[ChatSession]:
        """Get a specific session by ID"""
        db_session = self.get_session()
        try:
            session_model = db_session.query(ChatSessionModel).filter(
                ChatSessionModel.session_id == session_id
            ).first()

            if session_model:
                return ChatSession.from_model(session_model)
            return None
        finally:
            db_session.close()

    def search_conversations(self, query: str, session_id: Optional[str] = None, limit: int = 20) -> List[
        ConversationContext]:
        """Search conversations by question or SQL content"""
        db_session = self.get_session()
        try:
            query_filter = db_session.query(ConversationModel).filter(
                ConversationModel.user_question.contains(query) |
                ConversationModel.sql_query.contains(query) |
                ConversationModel.result_summary.contains(query)
            )

            if session_id:
                query_filter = query_filter.filter(ConversationModel.session_id == session_id)

            conversations = query_filter.order_by(
                desc(ConversationModel.timestamp)
            ).limit(limit).all()

            return [ConversationContext.from_model(conv) for conv in conversations]
        finally:
            db_session.close()

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get overall conversation statistics"""
        db_session = self.get_session()
        try:
            total_sessions = db_session.query(ChatSessionModel).count()
            total_conversations = db_session.query(ConversationModel).count()
            successful_conversations = db_session.query(ConversationModel).filter(
                ConversationModel.success == True
            ).count()

            return {
                "total_sessions": total_sessions,
                "total_conversations": total_conversations,
                "successful_conversations": successful_conversations,
                "success_rate": (
                            successful_conversations / total_conversations * 100) if total_conversations > 0 else 0,
                "model_usage": {}
            }
        finally:
            db_session.close()


class EnhancedSQLConversationMemory:
    """Enhanced memory system with multi-session support"""

    def __init__(self, memory_type: str = "buffer", max_tokens: int = 2000):
        self.memory_type = memory_type
        self.max_tokens = max_tokens
        self.conversation_db = ConversationDatabase()

        self.current_session: Optional[ChatSession] = None
        self.current_conversations: List[ConversationContext] = []
        self.memory: Optional[Any] = None

        self.available_models = {
            "gpt-3.5-turbo": "GPT-3.5 Turbo (Fast & Cost-effective)",
            "gpt-4": "GPT-4 (Most Capable)",
            "gpt-4-turbo": "GPT-4 Turbo (Latest)",
            "gpt-4o": "GPT-4o (Omni Model)",
            "gpt-4o-mini": "GPT-4o Mini (Efficient)"
        }

    def create_new_session(self, session_name: str, model_name: str, database_name: Optional[str] = None) -> str:
        """Create a new chat session"""
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

        self.conversation_db.save_session(session)
        self.switch_to_session(session_id)

        return session_id

    def switch_to_session(self, session_id: str):
        """Switch to a different session"""
        sessions = self.conversation_db.get_sessions()
        session = next((s for s in sessions if s.session_id == session_id), None)

        if not session:
            raise ValueError(f"Session {session_id} not found")

        self.current_session = session
        self.current_conversations = self.conversation_db.get_conversations(session_id)

        self._setup_memory()

        for conv in self.current_conversations:
            if self.memory:
                self.memory.chat_memory.add_user_message(conv.user_question)
                ai_response = f"SQL: {conv.sql_query}\nResult: {conv.result_summary}"
                self.memory.chat_memory.add_ai_message(ai_response)

    def _setup_memory(self):
        """Setup LangChain memory for current session"""
        if not self.current_session:
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found. Memory system will use basic mode.")
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="result"
            )
            return

        try:
            llm = ChatOpenAI(
                model=self.current_session.model_name,
                temperature=0,
                openai_api_key=api_key
            )

            if self.memory_type == "buffer":
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="result"
                )
            elif self.memory_type == "window":
                self.memory = ConversationBufferWindowMemory(
                    k=10,
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="result"
                )
            elif self.memory_type == "token":
                self.memory = ConversationTokenBufferMemory(
                    llm=llm,
                    max_token_limit=self.max_tokens,
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="result"
                )
            elif self.memory_type == "summary":
                self.memory = ConversationSummaryMemory(
                    llm=llm,
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="result"
                )
            else:
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="result"
                )
        except Exception as e:
            print(f"Memory setup failed: {e}. Using basic memory.")
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="result"
            )

    def add_conversation(self, context: ConversationContext):
        """Add a new conversation to current session"""
        if not self.current_session:
            raise ValueError("No active session")

        context.session_id = self.current_session.session_id
        if not context.conversation_id:
            context.conversation_id = str(uuid.uuid4())

        self.conversation_db.save_conversation(context)
        self.conversation_db.update_session_activity(self.current_session.session_id)

        self.current_conversations.append(context)

        if self.memory:
            self.memory.chat_memory.add_user_message(context.user_question)
            ai_response = f"SQL: {context.sql_query}\nResult: {context.result_summary}"
            self.memory.chat_memory.add_ai_message(ai_response)

        self.current_session.last_active = datetime.now()
        self.current_session.conversation_count += 1

    def get_all_sessions(self) -> List[ChatSession]:
        """Get all chat sessions"""
        return self.conversation_db.get_sessions()

    def get_session_conversations(self, session_id: str) -> List[ConversationContext]:
        """Get conversations for a specific session"""
        return self.conversation_db.get_conversations(session_id)

    def delete_session(self, session_id: str):
        """Delete a session"""
        self.conversation_db.delete_session(session_id)

        if self.current_session and self.current_session.session_id == session_id:
            self.current_session = None
            self.current_conversations = []
            self.memory = None

    def get_conversation_context(self) -> str:
        """Get conversation context for current session"""
        if not self.current_conversations:
            return "No previous conversation history in this session."

        recent_conversations = self.current_conversations[-5:]

        context = f"Recent conversation history (Session: {self.current_session.session_name}):\n"
        for i, conv in enumerate(recent_conversations, 1):
            context += f"\n{i}. User asked: {conv.user_question}"
            context += f"\n   SQL used: {conv.sql_query}"
            context += f"\n   Result: {conv.result_summary[:100]}..."
            context += f"\n   Tables: {', '.join(conv.tables_used)}\n"

        return context

    def get_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables for LangChain"""
        if self.memory:
            return self.memory.load_memory_variables({})
        return {}

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        if not self.current_session:
            return {"error": "No active session"}

        conversations = self.current_conversations
        if not conversations:
            return {
                "session_name": self.current_session.session_name,
                "model_name": self.current_session.model_name,
                "total_conversations": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "success_rate": 0,
                "most_used_tables": [],
                "common_query_types": []
            }

        total = len(conversations)
        successful = sum(1 for conv in conversations if conv.success)
        failed = total - successful

        all_tables = []
        for conv in conversations:
            all_tables.extend(conv.tables_used)

        from collections import Counter
        table_counts = Counter(all_tables)
        most_used_tables = table_counts.most_common(5)

        query_types = [conv.query_type for conv in conversations]
        type_counts = Counter(query_types)
        common_types = type_counts.most_common(3)

        return {
            "session_name": self.current_session.session_name,
            "model_name": self.current_session.model_name,
            "database_name": self.current_session.database_name,
            "total_conversations": total,
            "successful_queries": successful,
            "failed_queries": failed,
            "success_rate": (successful / total) * 100 if total > 0 else 0,
            "most_used_tables": most_used_tables,
            "common_query_types": common_types,
            "session_duration": (
                                            self.current_session.last_active - self.current_session.created_at).total_seconds() / 3600
        }

    def export_session_history(self, session_id: str) -> dict:
        """Export session history as JSON"""
        sessions = self.conversation_db.get_sessions()
        session = next((s for s in sessions if s.session_id == session_id), None)

        if not session:
            return {"error": "Session not found"}

        conversations = self.conversation_db.get_conversations(session_id)

        return {
            "session": session.to_dict(),
            "conversations": [conv.to_dict() for conv in conversations]
        }

    def clear_session_memory(self):
        """Clear current session memory (LangChain only, keep database)"""
        if self.memory:
            self.memory.clear()

    def get_available_models(self) -> Dict[str, str]:
        """Get available GPT models"""
        return self.available_models


class SimpleConversationMemory:
    """Simple conversation memory that sends previous messages to LLM"""

    def __init__(self, max_messages: int = 10):
        self.messages = []
        self.max_messages = max_messages

    def add_message(self, user_message: str, ai_response: str):
        """Add new message pair"""
        self.messages.append({
            'user': user_message,
            'ai': ai_response,
            'time': datetime.now().strftime("%H:%M")
        })

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        print(f"{len(self.messages)} messages stored")

    def build_context_prompt(self, new_user_message: str, base_system_prompt: str = "") -> str:
        """Build prompt with previous messages for LLM"""

        prompt = ""

        if base_system_prompt:
            prompt += f"{base_system_prompt}\n\n"

        if self.messages:
            prompt += "=== PREVIOUS CONVERSATION HISTORY ===\n"

            for i, msg in enumerate(self.messages, 1):
                prompt += f"{i}. User ({msg['time']}): {msg['user']}\n"
                prompt += f"   AI Response: {msg['ai']}\n\n"

            prompt += "=== REMEMBER THE CONVERSATIONS ABOVE ===\n"
            prompt += "Resolve references like 'this person', 'he', 'his' from previous messages.\n\n"

        prompt += f"NEW USER MESSAGE: {new_user_message}\n\n"
        prompt += "RESPONSE:"

        return prompt

    def clear(self):
        """Clear all messages"""
        self.messages.clear()
        print("Conversation memory cleared")

    def get_last_entities(self) -> str:
        """Get important information from recent messages"""
        if not self.messages:
            return ""

        recent_responses = []
        for msg in self.messages[-3:]:
            recent_responses.append(msg['ai'])

        return " ".join(recent_responses)


# Global memory instance
simple_memory = SimpleConversationMemory()


def chat_with_memory(user_input: str, your_llm_function, system_prompt: str = ""):
    """
    Chat with memory - sends previous messages to LLM

    Usage:
    response = chat_with_memory(
        user_input,
        your_langchain_function,
        "You are a SQL assistant..."
    )
    """

    full_prompt = simple_memory.build_context_prompt(user_input, system_prompt)

    print(f"Prompt sent to LLM: {len(full_prompt)} characters")

    try:
        ai_response = your_llm_function(full_prompt)
        simple_memory.add_message(user_input, str(ai_response))
        return ai_response

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        simple_memory.add_message(user_input, error_msg)
        return error_msg


def add_to_memory(user_msg: str, ai_msg: str):
    """Manually add message to memory"""
    simple_memory.add_message(user_msg, ai_msg)


def clear_memory():
    """Clear memory"""
    simple_memory.clear()


def get_conversation_history() -> str:
    """Show current conversation history"""
    if not simple_memory.messages:
        return "No conversations yet"

    history = "CONVERSATION HISTORY:\n"
    for i, msg in enumerate(simple_memory.messages, 1):
        history += f"{i}. User: {msg['user']}\n"
        history += f"   AI: {msg['ai'][:100]}...\n\n"

    return history


def integrate_simple_memory_with_enhanced():
    """Integrate existing Enhanced class with simple memory"""

    original_add_conversation = EnhancedSQLConversationMemory.add_conversation

    def enhanced_add_with_simple_memory(self, context: ConversationContext):
        original_add_conversation(self, context)

        simple_memory.add_message(
            context.user_question,
            context.result_summary or "No result found"
        )

    EnhancedSQLConversationMemory.add_conversation = enhanced_add_with_simple_memory
    print("Enhanced Memory integrated with simple conversation memory")


integrate_simple_memory_with_enhanced()


def test_conversation_memory():
    """Test conversation memory"""
    print("CONVERSATION MEMORY TEST")
    print("=" * 50)

    def demo_llm(prompt: str) -> str:
        if "tallest" in prompt.lower():
            return "Tallest player is Yao Ming (229 cm, $17M salary)"
        elif "this person" in prompt.lower() and "salary" in prompt.lower():
            if "Yao Ming" in prompt:
                return "Yao Ming salary: $17,000,000"
            else:
                return "Which person are you referring to?"
        elif "he" in prompt.lower() and "team" in prompt.lower():
            if "Yao Ming" in prompt:
                return "Yao Ming team: Retired"
            else:
                return "Which player's team are you asking about?"
        else:
            return f"'{prompt.split('NEW USER MESSAGE: ')[-1].split('RESPONSE:')[0].strip()}' query response"

    test_queries = [
        "Who is the tallest NBA player?",
        "How much does this person make?",
        "Which team does he play for?",
        "New question: Who is the best scorer?",
        "How old is this player?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. User: {query}")

        response = chat_with_memory(
            query,
            demo_llm,
            "You are an NBA database assistant. Remember previous conversations."
        )

        print(f"   AI: {response}")

        if i <= 2:
            print(f"\nPrompt sent to LLM (last part):")
            last_prompt = simple_memory.build_context_prompt(query, "NBA assistant")
            print(last_prompt[-200:] + "\n")

    print(f"\nTotal {len(simple_memory.messages)} messages stored")
    print("AI now remembers previous conversations!")


def interactive_test():
    """Interactive test"""
    print("INTERACTIVE CONVERSATION MEMORY TEST")
    print("Type 'quit' to exit")
    print("-" * 50)

    def simple_llm(prompt: str) -> str:
        user_msg = prompt.split("NEW USER MESSAGE: ")[-1].split("RESPONSE:")[0].strip()
        return f"Understood: '{user_msg}' - Responding with this context."

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                break
            elif user_input.lower() == 'clear':
                clear_memory()
                continue
            elif user_input.lower() == 'history':
                print(get_conversation_history())
                continue
            elif not user_input:
                continue

            response = chat_with_memory(user_input, simple_llm, "Demo AI")
            print(f"AI: {response}")

        except KeyboardInterrupt:
            break

    print("\nTest finished!")


def integration_guide():
    """Integration guide"""
    print("INTEGRATION GUIDE FOR YOUR EXISTING CODE")
    print("=" * 50)
    print("")
    print("1. YOUR PREVIOUS CODE:")
    print("def your_query_function(user_input):")
    print("    result = langchain_manager.query(user_input)")
    print("    return result")
    print("")
    print("2. YOUR NEW CODE:")
    print("def your_query_function(user_input):")
    print("    result = chat_with_memory(")
    print("        user_input,")
    print("        lambda prompt: langchain_manager.query(prompt),")
    print("        'You are a SQL database assistant.'")
    print("    )")
    print("    return result")
    print("")
    print("That's it! AI will now remember all conversations.")


def test_enhanced_memory_system():
    """Test the enhanced memory system"""
    print("Testing Enhanced Memory System...")

    memory = EnhancedSQLConversationMemory(memory_type="buffer")

    session1_id = memory.create_new_session("Data Analysis Session", "gpt-4", "company_db")
    print(f"Created session 1: {session1_id}")

    context = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        user_question="How many employees are there?",
        sql_query="SELECT COUNT(*) FROM employees",
        result_summary="Found 25 employees",
        timestamp=datetime.now(),
        success=True,
        tables_used=["employees"],
        query_type="aggregation",
        follow_up_suggestions=[],
        model_used="gpt-4",
        execution_time=1.5,
        session_id=session1_id
    )

    memory.add_conversation(context)
    print("Added test conversation")

    summary = memory.get_session_summary()
    print(f"Session summary: {summary['total_conversations']} conversations")

    print("Enhanced memory system test completed!")
    return memory


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "test":
            test_conversation_memory()
        elif command == "interactive":
            interactive_test()
        elif command == "enhanced":
            test_enhanced_memory_system()
        elif command == "guide":
            integration_guide()
        else:
            print("SIMPLE CONVERSATION MEMORY")
            print("Sends previous messages to LLM on each request")
            print("")
            print("Usage:")
            print("python langchain_memory_system.py test        # Demo test")
            print("python langchain_memory_system.py interactive # Interactive test")
            print("python langchain_memory_system.py enhanced   # Enhanced memory test")
            print("python langchain_memory_system.py guide      # Integration guide")
    else:
        print("CONVERSATION MEMORY READY!")
        print("=" * 40)
        print("Enhanced Memory + Simple Conversation Memory integrated")
        print("LLM will now remember all conversations")
        print("")
        print("To test:")
        print("python langchain_memory_system.py test")
        print("")
        integration_guide()