import os
import time
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool
)
from langchain.agents import Tool
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI


@dataclass
class ChainResult:
    """Result from LangChain SQL operations"""
    query: str
    sql_command: str
    result: Any
    intermediate_steps: List[str]
    success: bool
    error_message: Optional[str] = None


class SQLOutputParser(BaseOutputParser):
    """Custom parser for SQL chain outputs"""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the output from SQL chain"""
        return {
            "answer": text,
            "sql_command": self._extract_sql(text),
            "explanation": self._extract_explanation(text)
        }

    def _extract_sql(self, text: str) -> str:
        """Extract SQL command from text"""
        import re
        sql_match = re.search(r'SQLQuery:\s*(.*?)\n', text, re.IGNORECASE | re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        return ""

    def _extract_explanation(self, text: str) -> str:
        """Extract explanation from text"""
        import re
        exp_match = re.search(r'SQLResult:\s*(.*?)$', text, re.IGNORECASE | re.DOTALL)
        if exp_match:
            return exp_match.group(1).strip()
        return text

    @property
    def _type(self) -> str:
        """Return the type key for this parser"""
        return "sql_output_parser"


class LangChainSQLChains:
    """Advanced SQL chains and tools using LangChain"""

    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        self.output_parser = SQLOutputParser()

        self._setup_chains()
        self._setup_tools()

    def _setup_chains(self):
        """Setup various SQL chains"""

        try:
            self.basic_chain = SQLDatabaseChain.from_llm(
                llm=self.llm,
                db=self.db,
                verbose=True,
                return_intermediate_steps=True,
                use_query_checker=True
            )
        except Exception as e:
            print(f"Warning: Could not create basic chain: {e}")
            self.basic_chain = None

        try:
            from langchain_experimental.sql import SQLDatabaseSequentialChain
            self.sequential_chain = SQLDatabaseSequentialChain.from_llm(
                llm=self.llm,
                db=self.db,
                verbose=True,
                query_checker_prompt=self._get_query_checker_prompt(),
                return_intermediate_steps=True
            )
        except Exception as e:
            print(f"Warning: Could not create sequential chain: {e}")
            self.sequential_chain = None

        try:
            custom_prompt = self._get_custom_sql_prompt()
            self.custom_chain = SQLDatabaseChain.from_llm(
                llm=self.llm,
                db=self.db,
                prompt=custom_prompt,
                verbose=True,
                return_intermediate_steps=True,
                use_query_checker=True
            )
        except Exception as e:
            print(f"Warning: Could not create custom chain: {e}")
            self.custom_chain = self.basic_chain

    def _setup_tools(self):
        """Setup SQL database tools"""
        try:
            self.info_tool = InfoSQLDatabaseTool(db=self.db)
            self.list_tool = ListSQLDatabaseTool(db=self.db)
            self.query_tool = QuerySQLDataBaseTool(db=self.db)
            self.checker_tool = QuerySQLCheckerTool(db=self.db, llm=self.llm)

            self.tools = [
                Tool(
                    name="sql_db_schema",
                    description="Get information about SQL database schema and tables",
                    func=self.info_tool.run
                ),
                Tool(
                    name="sql_db_list_tables",
                    description="List all tables in the SQL database",
                    func=self.list_tool.run
                ),
                Tool(
                    name="sql_db_query",
                    description="Execute SQL query against the database",
                    func=self.query_tool.run
                ),
                Tool(
                    name="sql_db_query_checker",
                    description="Check if SQL query is correct before execution",
                    func=self.checker_tool.run
                )
            ]
        except Exception as e:
            print(f"Warning: Could not setup tools: {e}")
            self.tools = []

    def _get_custom_sql_prompt(self) -> PromptTemplate:
        """Get custom SQL prompt template"""

        template = """You are a SQL expert. Given an input question, create a syntactically correct {dialect} query to run.

Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using LIMIT clause as shown in the examples below.

You can order the results to return the most informative data in the database.

Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.

Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Pay attention to use CAST(column_name AS REAL) for arithmetic operations on numeric columns to avoid integer division.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
{table_info}

Question: {input}"""

        return PromptTemplate(
            input_variables=["input", "table_info", "dialect", "top_k"],
            template=template
        )

    def _get_query_checker_prompt(self) -> PromptTemplate:
        """Get query checker prompt"""

        template = """{query}
Double check the {dialect} query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatches in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the correct columns that exist in the tables
- Using proper JOIN syntax

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only."""

        return PromptTemplate(
            input_variables=["query", "dialect"],
            template=template
        )

    def run_basic_chain(self, question: str) -> ChainResult:
        """Run basic SQL database chain"""
        if not self.basic_chain:
            return ChainResult(
                query=question,
                sql_command="",
                result="Basic chain not available",
                intermediate_steps=[],
                success=False,
                error_message="Basic chain could not be initialized"
            )

        try:
            result = self.basic_chain.invoke({"query": question})

            sql_command = ""
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps:
                for step in intermediate_steps:
                    if isinstance(step, dict) and "sql_cmd" in step:
                        sql_command = step["sql_cmd"]
                        break

            return ChainResult(
                query=question,
                sql_command=sql_command,
                result=result.get("result", ""),
                intermediate_steps=[str(step) for step in intermediate_steps],
                success=True
            )

        except Exception as e:
            return ChainResult(
                query=question,
                sql_command="",
                result="",
                intermediate_steps=[],
                success=False,
                error_message=str(e)
            )

    def run_sequential_chain(self, question: str) -> ChainResult:
        """Run sequential SQL chain for complex queries"""
        if not self.sequential_chain:
            return ChainResult(
                query=question,
                sql_command="",
                result="Sequential chain not available",
                intermediate_steps=[],
                success=False,
                error_message="Sequential chain could not be initialized"
            )

        try:
            result = self.sequential_chain.invoke({"query": question})

            return ChainResult(
                query=question,
                sql_command=result.get("sql_command", ""),
                result=result.get("result", ""),
                intermediate_steps=[str(step) for step in result.get("intermediate_steps", [])],
                success=True
            )

        except Exception as e:
            return ChainResult(
                query=question,
                sql_command="",
                result="",
                intermediate_steps=[],
                success=False,
                error_message=str(e)
            )

    def run_custom_chain(self, question: str) -> ChainResult:
        """Run custom SQL chain with enhanced prompts"""
        if not self.custom_chain:
            return self.run_basic_chain(question)

        try:
            result = self.custom_chain.invoke({"query": question})

            sql_command = ""
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps:
                for step in intermediate_steps:
                    if isinstance(step, dict) and "sql_cmd" in step:
                        sql_command = step["sql_cmd"]
                        break

            return ChainResult(
                query=question,
                sql_command=sql_command,
                result=result.get("result", ""),
                intermediate_steps=[str(step) for step in intermediate_steps],
                success=True
            )

        except Exception as e:
            return ChainResult(
                query=question,
                sql_command="",
                result="",
                intermediate_steps=[],
                success=False,
                error_message=str(e)
            )

    def analyze_query_complexity(self, question: str) -> str:
        """Analyze query complexity and suggest best chain"""

        complexity_indicators = {
            "simple": ["count", "show", "list", "what is", "how many"],
            "medium": ["average", "sum", "group by", "order by", "join"],
            "complex": ["nested", "subquery", "multiple joins", "window functions", "cte"]
        }

        question_lower = question.lower()

        if any(indicator in question_lower for indicator in complexity_indicators["complex"]):
            return "sequential"
        elif any(indicator in question_lower for indicator in complexity_indicators["medium"]):
            return "custom"
        else:
            return "basic"

    def get_table_schema(self, table_name: Optional[str] = None) -> str:
        """Get table schema information"""
        try:
            if hasattr(self, 'info_tool') and self.info_tool:
                if table_name:
                    return self.info_tool.run(table_name)
                else:
                    return self.info_tool.run("")
            else:
                return "Info tool not available"
        except Exception as e:
            return f"Error getting schema: {str(e)}"

    def list_tables(self) -> str:
        """List all database tables"""
        try:
            if hasattr(self, 'list_tool') and self.list_tool:
                return self.list_tool.run("")
            else:
                return "List tool not available"
        except Exception as e:
            return f"Error listing tables: {str(e)}"

    def validate_query(self, sql_query: str) -> str:
        """Validate SQL query"""
        try:
            if hasattr(self, 'checker_tool') and self.checker_tool:
                return self.checker_tool.run(sql_query)
            else:
                return "Query checker tool not available"
        except Exception as e:
            return f"Error validating query: {str(e)}"

    def execute_query(self, sql_query: str) -> str:
        """Execute SQL query directly"""
        try:
            if hasattr(self, 'query_tool') and self.query_tool:
                return self.query_tool.run(sql_query)
            else:
                return "Query tool not available"
        except Exception as e:
            return f"Error executing query: {str(e)}"


class AdvancedSQLAnalyzer:
    """Advanced SQL query analyzer and optimizer"""

    def __init__(self, chains: LangChainSQLChains):
        self.chains = chains
        self.llm = chains.llm

    def suggest_optimizations(self, sql_query: str) -> List[str]:
        """Suggest SQL query optimizations"""

        optimization_prompt = f"""Analyze this SQL query and suggest optimizations:

{sql_query}

Consider:
1. Index usage opportunities
2. JOIN order optimization
3. WHERE clause improvements
4. LIMIT clause usage
5. Column selection optimization
6. Subquery vs JOIN alternatives

Provide specific, actionable recommendations."""

        try:
            response = self.llm.predict(optimization_prompt)
            return [line.strip() for line in response.split('\n') if line.strip()]
        except Exception as e:
            return [f"Error generating optimizations: {str(e)}"]

    def explain_execution_plan(self, sql_query: str) -> str:
        """Explain SQL execution plan in simple terms"""

        explanation_prompt = f"""Explain how this SQL query would be executed by the database:

{sql_query}

Provide a step-by-step explanation in simple terms that a non-technical person could understand.
Focus on:
1. What tables are accessed
2. How data is filtered
3. How results are combined
4. How results are sorted/grouped"""

        try:
            return self.llm.predict(explanation_prompt)
        except Exception as e:
            return f"Error explaining execution plan: {str(e)}"

    def generate_related_queries(self, original_query: str, context: str = "") -> List[str]:
        """Generate related query suggestions"""

        suggestion_prompt = f"""Based on this SQL query:
{original_query}

Database context:
{context}

Generate 5 related queries that might provide additional insights.
Make them progressively more complex and analytical.

Return only the natural language questions, one per line."""

        try:
            response = self.llm.predict(suggestion_prompt)
            return [q.strip() for q in response.split('\n') if q.strip()]
        except Exception as e:
            return [f"Error generating suggestions: {str(e)}"]

    def convert_to_different_sql_dialects(self, sql_query: str) -> Dict[str, str]:
        """Convert SQL query to different dialects"""

        dialects = ["PostgreSQL", "MySQL", "SQL Server", "Oracle"]
        converted_queries = {}

        for dialect in dialects:
            conversion_prompt = f"""Convert this SQLite query to {dialect} syntax:

{sql_query}

Account for dialect-specific differences in:
- Function names
- Data types
- Syntax variations
- Date/time handling

Return only the converted SQL query."""

            try:
                converted_queries[dialect] = self.llm.predict(conversion_prompt)
            except Exception as e:
                converted_queries[dialect] = f"Error converting to {dialect}: {str(e)}"

        return converted_queries


class SQLMetricsCollector:
    """Collect and analyze SQL query metrics"""

    def __init__(self):
        self.query_history = []
        self.performance_metrics = {}

    def record_query(self, chain_result: ChainResult, execution_time: float):
        """Record query execution metrics"""

        metric = {
            "timestamp": pd.Timestamp.now(),
            "question": chain_result.query,
            "sql_command": chain_result.sql_command,
            "success": chain_result.success,
            "execution_time": execution_time,
            "result_length": len(str(chain_result.result)),
            "intermediate_steps": len(chain_result.intermediate_steps)
        }

        self.query_history.append(metric)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""

        if not self.query_history:
            return {"message": "No queries recorded yet"}

        df = pd.DataFrame(self.query_history)

        return {
            "total_queries": len(self.query_history),
            "success_rate": df["success"].mean() * 100,
            "avg_execution_time": df["execution_time"].mean(),
            "fastest_query": df["execution_time"].min(),
            "slowest_query": df["execution_time"].max(),
            "most_common_patterns": self._analyze_query_patterns(df),
            "recent_queries": df.tail(5).to_dict('records')
        }

    def _analyze_query_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze common query patterns"""

        patterns = []

        for _, row in df.iterrows():
            sql = str(row["sql_command"]).upper()
            if "GROUP BY" in sql:
                patterns.append("Aggregation")
            elif "JOIN" in sql:
                patterns.append("Join")
            elif "ORDER BY" in sql:
                patterns.append("Sorting")
            else:
                patterns.append("Simple Select")

        if patterns:
            pattern_counts = pd.Series(patterns).value_counts()
            return pattern_counts.head(3).to_dict()
        else:
            return {}

    def export_metrics(self) -> pd.DataFrame:
        """Export metrics as DataFrame"""
        return pd.DataFrame(self.query_history)


class EnhancedLangChainSQL:
    """Enhanced LangChain SQL system with all components"""

    def __init__(self, db, llm):
        self.db = db
        self.llm = llm

        self.chains = LangChainSQLChains(db, llm)
        self.analyzer = AdvancedSQLAnalyzer(self.chains)
        self.metrics = SQLMetricsCollector()

    def query(self, question: str, chain_type: str = "auto") -> ChainResult:
        """Execute query with automatic chain selection"""

        start_time = time.time()

        if chain_type == "auto":
            chain_type = self.chains.analyze_query_complexity(question)

        if chain_type == "sequential":
            result = self.chains.run_sequential_chain(question)
        elif chain_type == "custom":
            result = self.chains.run_custom_chain(question)
        else:
            result = self.chains.run_basic_chain(question)

        execution_time = time.time() - start_time
        self.metrics.record_query(result, execution_time)

        return result

    def get_comprehensive_analysis(self, question: str) -> Dict[str, Any]:
        """Get comprehensive analysis of a query"""

        result = self.query(question)

        analysis = {
            "query_result": result,
            "performance_metrics": self.metrics.get_performance_summary(),
        }

        if result.success and result.sql_command:
            analysis.update({
                "optimizations": self.analyzer.suggest_optimizations(result.sql_command),
                "execution_plan": self.analyzer.explain_execution_plan(result.sql_command),
                "related_queries": self.analyzer.generate_related_queries(question),
                "dialect_conversions": self.analyzer.convert_to_different_sql_dialects(result.sql_command)
            })

        return analysis


def test_langchain_sql_chains():
    """Test LangChain SQL chains and tools"""
    print("Testing LangChain SQL Chains...")

    try:
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
            return False

        import sqlite3
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
        cursor.execute("INSERT INTO test_table VALUES (1, 'test')")
        conn.commit()

        db = SQLDatabase.from_uri("sqlite:///:memory:")
        llm = ChatOpenAI(temperature=0)

        chains = LangChainSQLChains(db, llm)

        print("LangChain SQL chains initialized successfully")

        tools = chains.tools
        print(f"Created {len(tools)} SQL tools")

        print("LangChain SQL chains test completed!")
        return True

    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("This may be normal if OpenAI API key is not configured")
        return False


if __name__ == "__main__":
    test_langchain_sql_chains()