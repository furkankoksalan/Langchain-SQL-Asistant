import os
import sqlite3
import pandas as pd
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv

load_dotenv()


@dataclass
class QueryResult:
    """Structure for query execution results"""
    sql_query: str
    result_data: Optional[pd.DataFrame]
    explanation: str
    success: bool
    error_message: Optional[str]
    execution_time: float
    agent_thoughts: List[str]


class QueryCallbackHandler(BaseCallbackHandler):
    """Basic callback handler to capture agent thoughts and SQL queries"""

    def __init__(self):
        self.thoughts = []
        self.sql_queries = []
        self.current_sql = ""

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Capture agent actions"""
        self.thoughts.append(f"Action: {action.tool} - {action.tool_input}")
        if "sql" in action.tool.lower():
            self.current_sql = action.tool_input
            self.sql_queries.append(action.tool_input)

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Capture final result"""
        self.thoughts.append(f"Final Answer: {finish.return_values}")

    def reset(self):
        """Reset callback state"""
        self.thoughts = []
        self.sql_queries = []
        self.current_sql = ""


class LangChainSQLManager:
    """Basic SQL Database Manager using LangChain"""

    def __init__(self, database_url: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        self.database_url = database_url or "sqlite:///langchain_sample.db"
        self.model_name = model_name

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")

        try:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                openai_api_key=api_key
            )
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI LLM: {e}")
            self.llm = None

        self.db = None
        self.agent = None
        self.toolkit = None

        self.memory = ConversationBufferMemory(memory_key="history")
        self.callback_handler = QueryCallbackHandler()

        self._connect()

    def _connect(self) -> bool:
        """Connect to the database using LangChain SQLDatabase"""
        try:
            self.db = SQLDatabase.from_uri(self.database_url)

            if not self.llm:
                print("Cannot create agent without LLM")
                return False

            self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=self.toolkit,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                memory=self.memory,
                handle_parsing_errors=True,
                max_iterations=5,
                early_stopping_method="generate"
            )

            print(f"Connected to database: {self.database_url}")
            table_names = self.db.get_usable_table_names()
            print(f"Available tables: {table_names}")
            return True

        except Exception as e:
            print(f"Database connection failed: {str(e)}")
            return False

    def query_natural_language(self, question: str) -> QueryResult:
        """Process natural language query using LangChain SQL Agent"""
        if not self.agent:
            return QueryResult(
                sql_query="",
                result_data=None,
                explanation="Database not connected or LLM not available",
                success=False,
                error_message="No database connection or LLM",
                execution_time=0.0,
                agent_thoughts=[]
            )

        try:
            start_time = time.time()

            self.callback_handler.reset()

            result = self.agent.run(
                question,
                callbacks=[self.callback_handler]
            )

            execution_time = time.time() - start_time

            sql_query = ""
            if self.callback_handler.sql_queries:
                sql_query = self.callback_handler.sql_queries[-1]

            result_data = None
            if sql_query:
                try:
                    result_data = pd.read_sql(sql_query, self.db._engine)
                except Exception as e:
                    print(f"Warning: Could not convert result to DataFrame: {e}")

            return QueryResult(
                sql_query=sql_query,
                result_data=result_data,
                explanation=str(result),
                success=True,
                error_message=None,
                execution_time=execution_time,
                agent_thoughts=self.callback_handler.thoughts
            )

        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0.0
            return QueryResult(
                sql_query="",
                result_data=None,
                explanation=f"Query execution failed: {str(e)}",
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                agent_thoughts=self.callback_handler.thoughts
            )

    def execute_raw_sql(self, sql_query: str) -> QueryResult:
        """Execute raw SQL query"""
        if not self.db:
            return QueryResult(
                sql_query=sql_query,
                result_data=None,
                explanation="Database not connected",
                success=False,
                error_message="No database connection",
                execution_time=0.0,
                agent_thoughts=[]
            )

        try:
            start_time = time.time()

            result_data = pd.read_sql(sql_query, self.db._engine)
            execution_time = time.time() - start_time

            return QueryResult(
                sql_query=sql_query,
                result_data=result_data,
                explanation=f"Query executed successfully. {len(result_data)} rows returned.",
                success=True,
                error_message=None,
                execution_time=execution_time,
                agent_thoughts=[]
            )

        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0.0
            return QueryResult(
                sql_query=sql_query,
                result_data=None,
                explanation=f"SQL execution failed: {str(e)}",
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                agent_thoughts=[]
            )

    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information"""
        if not self.db:
            return {"error": "Database not connected"}

        try:
            table_names = self.db.get_usable_table_names()
            table_info = self.db.get_table_info()

            samples = {}
            for table in table_names[:5]:
                try:
                    sample_df = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", self.db._engine)
                    samples[table] = sample_df.to_dict('records')
                except Exception as e:
                    print(f"Warning: Could not get sample data from {table}: {e}")
                    samples[table] = []

            return {
                "tables": table_names,
                "table_info": table_info,
                "samples": samples,
                "dialect": str(self.db.dialect)
            }

        except Exception as e:
            return {"error": f"Failed to get database info: {str(e)}"}

    def create_sample_database(self) -> bool:
        """Create a comprehensive sample database"""
        try:
            db_path = "langchain_sample.db"
            if os.path.exists(db_path):
                os.remove(db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE companies (
                    company_id INTEGER PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    industry TEXT,
                    founded_year INTEGER,
                    headquarters TEXT,
                    revenue REAL,
                    employees INTEGER,
                    stock_symbol TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE employees (
                    employee_id INTEGER PRIMARY KEY,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    company_id INTEGER,
                    department TEXT,
                    position TEXT,
                    salary REAL,
                    hire_date DATE,
                    manager_id INTEGER,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (company_id) REFERENCES companies(company_id),
                    FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE projects (
                    project_id INTEGER PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    company_id INTEGER,
                    start_date DATE,
                    end_date DATE,
                    budget REAL,
                    status TEXT,
                    priority TEXT,
                    project_manager_id INTEGER,
                    FOREIGN KEY (company_id) REFERENCES companies(company_id),
                    FOREIGN KEY (project_manager_id) REFERENCES employees(employee_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE sales (
                    sale_id INTEGER PRIMARY KEY,
                    company_id INTEGER,
                    employee_id INTEGER,
                    product_name TEXT,
                    sale_date DATE,
                    amount REAL,
                    quantity INTEGER,
                    region TEXT,
                    customer_segment TEXT,
                    FOREIGN KEY (company_id) REFERENCES companies(company_id),
                    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
                )
            """)

            companies_data = [
                (1, 'TechCorp Inc', 'Technology', 2010, 'San Francisco', 50000000, 500, 'TECH'),
                (2, 'DataSoft LLC', 'Software', 2015, 'New York', 25000000, 250, 'DATA'),
                (3, 'AI Solutions', 'Artificial Intelligence', 2018, 'Boston', 15000000, 150, 'AISOL'),
                (4, 'CloudWorks', 'Cloud Computing', 2012, 'Seattle', 75000000, 750, 'CLOUD'),
                (5, 'StartupX', 'Fintech', 2020, 'Austin', 5000000, 50, None)
            ]

            cursor.executemany("""
                INSERT INTO companies (company_id, company_name, industry, founded_year, headquarters, revenue, employees, stock_symbol)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, companies_data)

            employees_data = [
                (1, 'John', 'Smith', 'john.smith@techcorp.com', 1, 'Engineering', 'Senior Developer', 95000,
                 '2019-01-15', None, 1),
                (2, 'Sarah', 'Johnson', 'sarah.j@datasoft.com', 2, 'Data Science', 'Data Scientist', 110000,
                 '2020-03-10', None, 1),
                (3, 'Mike', 'Chen', 'mike.chen@aisolutions.com', 3, 'Research', 'AI Researcher', 125000, '2021-06-01',
                 None, 1),
                (4, 'Emily', 'Davis', 'emily.d@cloudworks.com', 4, 'Engineering', 'Cloud Architect', 130000,
                 '2018-09-20', None, 1),
                (5, 'Alex', 'Wilson', 'alex.w@startupx.com', 5, 'Product', 'Product Manager', 100000, '2022-01-10', None,
                1),
            ]

            cursor.executemany("""
                INSERT INTO employees (employee_id, first_name, last_name, email, company_id, department, position, salary, hire_date, manager_id, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, employees_data)

            projects_data = [
                (1, 'AI Platform Development', 1, '2023-01-01', '2023-12-31', 2000000, 'In Progress', 'High', 1),
                (2, 'Data Analytics Dashboard', 2, '2023-03-15', '2023-09-30', 500000, 'Completed', 'Medium', 2),
                (3, 'Machine Learning Model', 3, '2023-02-01', '2023-11-30', 800000, 'In Progress', 'High', 3),
                (4, 'Cloud Migration Project', 4, '2023-01-15', '2023-08-15', 1500000, 'Completed', 'High', 4),
                (5, 'Mobile App Launch', 5, '2023-04-01', '2023-10-31', 300000, 'Planning', 'Medium', 5),
            ]

            cursor.executemany("""
                INSERT INTO projects (project_id, project_name, company_id, start_date, end_date, budget, status, priority, project_manager_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, projects_data)

            sales_data = [
                (1, 1, 1, 'AI Software License', '2023-01-15', 50000, 1, 'North America', 'Enterprise'),
                (2, 1, 1, 'Cloud Services', '2023-02-20', 25000, 5, 'North America', 'SMB'),
                (3, 2, 2, 'Data Analytics Tool', '2023-01-30', 75000, 2, 'Europe', 'Enterprise'),
                (4, 2, 2, 'Custom Dashboard', '2023-03-10', 30000, 1, 'Asia', 'SMB'),
                (5, 3, 3, 'ML Model Training', '2023-02-15', 100000, 1, 'North America', 'Enterprise'),
                (6, 4, 4, 'Cloud Infrastructure', '2023-01-25', 200000, 1, 'North America', 'Enterprise'),
                (7, 5, 5, 'Fintech Platform', '2023-04-15', 40000, 1, 'North America', 'SMB'),
            ]

            cursor.executemany("""
                INSERT INTO sales (sale_id, company_id, employee_id, product_name, sale_date, amount, quantity, region, customer_segment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, sales_data)

            conn.commit()
            conn.close()

            self.database_url = f"sqlite:///{db_path}"
            self._connect()

            print("Sample database created successfully")
            return True

        except Exception as e:
            print(f"Failed to create sample database: {str(e)}")
            return False

    def get_query_suggestions(self, context: str = "") -> List[str]:
        """Get query suggestions based on database schema"""
        suggestions = [
            "Show me all companies and their revenue",
            "What is the average salary by department?",
            "Which projects have the highest budget?",
            "Show me sales data for the last quarter",
            "Who are the top performing employees?",
            "Which company has the most employees?",
            "Show me all projects that are currently in progress",
            "What is the total revenue by industry?",
            "Which regions have the highest sales?",
            "Show me employee details with their managers"
        ]
        return suggestions

    def explain_query_result(self, query_result: QueryResult) -> str:
        """Explain the query result using basic description"""
        if not query_result.success:
            return f"Query failed: {query_result.error_message}"

        try:
            row_count = len(query_result.result_data) if query_result.result_data is not None else 0

            explanation = f"Query executed successfully in {query_result.execution_time:.2f} seconds.\n"
            explanation += f"SQL Query: {query_result.sql_query}\n"
            explanation += f"Number of rows returned: {row_count}\n"
            explanation += f"Result: {query_result.explanation}"

            return explanation

        except Exception as e:
            return f"Could not generate explanation: {str(e)}"


def test_langchain_sql_manager():
    """Test the basic LangChain SQL Manager"""
    print("Testing Basic LangChain SQL Manager...")

    manager = LangChainSQLManager()

    if manager.create_sample_database():
        print("Sample database created")
    else:
        print("Failed to create sample database")
        return

    db_info = manager.get_database_info()
    if "error" not in db_info:
        print(f"Found {len(db_info.get('tables', []))} tables")
    else:
        print(f"Database info error: {db_info['error']}")
        return

    test_queries = [
        "How many employees work at each company?",
        "What is the average salary by department?",
        "Show me all projects with high priority",
        "Which company has the highest revenue?"
    ]

    print("\nTesting Natural Language Queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Question: {query}")

        result = manager.query_natural_language(query)

        if result.success:
            print(f"   SQL: {result.sql_query}")
            print(f"   Rows: {len(result.result_data) if result.result_data is not None else 0}")
            print(f"   Time: {result.execution_time:.2f}s")
        else:
            print(f"   Failed: {result.error_message}")

    print("\nBasic LangChain SQL Manager test completed!")
    return manager


if __name__ == "__main__":
    test_langchain_sql_manager()