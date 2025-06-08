# Langchain-SQL-Assistant

A comprehensive SQL assistant application that transforms natural language into SQL queries using advanced AI models. Built with LangChain, Streamlit, and OpenAI GPT models for seamless database interaction.

## Features

### Core Capabilities
- **Natural Language Processing**: Convert plain English questions into SQL queries
- **Multi-Database Support**: Work with multiple datasets simultaneously
- **Conversation Memory**: AI remembers context from previous interactions
- **Session Management**: Organize queries into separate chat sessions
- **Real-time Execution**: Get instant results with data visualization
- **CSV Integration**: Upload CSV files and convert them to queryable databases

### Advanced AI Features
- **Context Awareness**: Understands references like "his salary", "that company"
- **Query Optimization**: Suggestions for improving SQL performance
- **Multiple AI Models**: Support for GPT-3.5, GPT-4, and newer variants
- **Smart References**: Follow-up questions using pronouns and entity names
- **Error Handling**: Intelligent error recovery and query correction

### Analytics & Insights
- **Performance Tracking**: Monitor query success rates and execution times
- **Usage Analytics**: Track most used tables and query patterns
- **Session Statistics**: Comprehensive analytics for each chat session
- **Export Capabilities**: Download results and session history

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Git (for cloning the repository)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/furkankoksalan/Langchain-SQL-Asistant.git
cd Langchain-SQL-Asistant
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Launch the application**
```bash
streamlit run langchain_streamlit_app.py
```

6. **Open in browser**
Navigate to `http://localhost:8501`

## Quick Start

### Step 1: Create Your First Session
- Click "New Chat" in the sidebar
- Enter a descriptive session name (e.g., "Sales Analysis Q4")
- Select your preferred AI model
- Click "Create"

### Step 2: Set Up Your Database
- Navigate to the "Database Setup" tab
- **Option A**: Upload CSV files
  - Select one or more CSV files
  - Configure table names
  - Click "Create Database from CSV"
- **Option B**: Use sample data
  - Click "Create Sample Dataset"
  - Get pre-built tables with companies, employees, projects, and sales data

### Step 3: Start Querying
- Switch to the "Chat" tab
- Ask questions in natural language
- View SQL queries, results, and explanations

## Usage Examples

### Basic Query
```
User: "How many employees work in each department?"

AI: I'll analyze the employee distribution by department.

SQL Generated:
SELECT department, COUNT(*) as employee_count 
FROM employees 
GROUP BY department 
ORDER BY employee_count DESC

Results: Engineering (25), Sales (18), Marketing (12)...
```

### Follow-up with Context
```
User: "What's the average salary in engineering?"

AI: Based on our previous analysis of departments, here's the engineering salary data.

SQL Generated:
SELECT AVG(salary) as avg_salary 
FROM employees 
WHERE department = 'Engineering'

Results: Average Engineering Salary: $87,500
```

### Reference-based Query
```
User: "Who is the highest paid employee?"
AI: "John Smith is the highest paid with $95,000 annual salary."

User: "What's his department?"
AI: "John Smith works in the Engineering department."
```

## Project Structure

```
langchain-sql-assistant/
├── langchain_streamlit_app.py                    # Main Streamlit interface
├── langchain_sql_manager.py            # Core SQL processing engine
├── langchain_memory_system.py          # Conversation memory management
├── langchain_sql_chains.py             # SQL chain operations
├── csv_to_sqlite_converter.py          # CSV processing utilities
├── requirements.txt                    # Python dependencies
├── .env                               # Environment configuration
└── README.md                         # Documentation
```

## Configuration

### AI Model Options
- **GPT-3.5 Turbo**: Fast and economical for standard queries
- **GPT-4**: Superior performance for complex analytical tasks
- **GPT-4 Turbo**: Latest capabilities with improved efficiency
- **GPT-4o**: Optimized for conversational AI interactions
- **GPT-4o Mini**: Lightweight version for basic operations

### Memory Configuration Types
- **Buffer Memory**: Stores complete recent conversation history
- **Summary Memory**: Condenses long conversations into summaries
- **Token Buffer**: Manages memory within API token limits
- **Window Memory**: Maintains sliding window of recent interactions

### Database Support
- **SQLite**: Primary database for CSV uploads and local development
- **PostgreSQL**: Enterprise database support (configurable)
- **MySQL**: Alternative database backend (configurable)
- **Custom Connections**: Connect to existing database instances

## Advanced Features

### Analytics Dashboard
Access comprehensive analytics in the "Analytics" tab:
- **Query Performance**: Success rates, execution times, error patterns
- **Usage Patterns**: Most frequently used tables and query types
- **Session Overview**: Timeline of queries and results
- **Dataset Statistics**: Data distribution and table usage metrics

### Query Optimization Engine
- **Performance Suggestions**: AI-generated optimization recommendations
- **Execution Plan Analysis**: Detailed breakdown of query execution
- **Alternative Approaches**: Different ways to achieve the same results
- **Index Recommendations**: Suggestions for database performance improvements

### Export and Import
- **Session Export**: Download complete conversation history as JSON
- **Result Export**: Save query results as CSV files
- **Database Schema**: Export table structures and relationships
- **Query Collections**: Save and organize frequently used queries

## Core Components

### Memory System (`langchain_memory_system.py`)
- **Enhanced Conversation Memory**: Maintains context across queries
- **Session Persistence**: Separate conversation histories for different projects
- **Entity Tracking**: Remembers people, companies, and other entities mentioned
- **Reference Resolution**: Handles pronouns and contextual references

### SQL Manager (`enhanced_langchain_sql_manager.py`)
- **LangChain Integration**: Advanced SQL agent with natural language processing
- **Query Validation**: Syntax checking and optimization
- **Performance Monitoring**: Execution time tracking and analytics
- **Multi-Database Support**: Switch between different datasets seamlessly

### User Interface (`streamlit_app.py`)
- **Clean Chat Interface**: Conversation-style interaction
- **Multi-Tab Layout**: Organized sections for different functionalities
- **Real-time Updates**: Live query results and visualizations
- **Session Management**: Intuitive session creation and switching

## Requirements

### Core Dependencies
```
streamlit>=1.28.0
pandas>=2.0.0
langchain>=0.1.0
langchain-openai>=0.0.8
langchain-community>=0.0.15
langchain-experimental>=0.0.50
sqlalchemy>=2.0.0
plotly>=5.17.0
python-dotenv>=1.0.0
```

### Optional Dependencies
```
psycopg2-binary>=2.9.0  # PostgreSQL support
pymysql>=1.1.0          # MySQL support
openpyxl>=3.1.0         # Excel file support
```

## Troubleshooting

### Common Installation Issues

**ModuleNotFoundError for LangChain components:**
```bash
pip install --upgrade langchain langchain-openai langchain-community langchain-experimental
```

**OpenAI API Authentication Error:**
- Verify your API key in the `.env` file
- Check API key permissions and billing status
- Restart the Streamlit application after updating credentials

**Database Connection Problems:**
- Ensure CSV files have consistent column names
- Check for special characters in table names
- Verify file permissions for uploaded datasets

### Performance Optimization

**For Large Datasets:**
- Use specific WHERE clauses to limit result sets
- Consider creating indexes on frequently queried columns
- Break complex queries into smaller, focused questions

**Memory Management:**
- Clear session memory periodically for long conversations
- Use summary memory type for extensive chat histories
- Monitor token usage with complex queries

### Database-Specific Issues

**CSV Upload Failures:**
- Check for empty cells or inconsistent data types
- Ensure column headers don't contain special characters
- Verify file encoding (UTF-8 recommended)

**Query Execution Errors:**
- Review generated SQL in the query explanation
- Check table and column names for typos
- Use the query validation feature before execution

## API Reference

## Contributing

### Development Setup
1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature
4. Make your changes with appropriate tests
5. Submit a pull request with detailed description

### Coding Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add error handling for external API calls
- Write unit tests for new functionality
- Update documentation for new features

### Testing Guidelines
- Test with various CSV formats and data types
- Verify functionality across different AI models
- Check memory system behavior with long conversations
- Validate database operations with edge cases

## Security Considerations

### API Key Management
- Never commit API keys to version control
- Use environment variables for sensitive configuration
- Rotate API keys regularly
- Monitor API usage for unusual activity

### Data Privacy
- Local SQLite databases for sensitive data
- No data sent to external services without explicit consent
- Session data can be cleared at any time
- Export functionality for data portability

## Performance Benchmarks

### Query Processing Times
- Simple SELECT queries: < 2 seconds
- Complex JOIN operations: 3-8 seconds
- Large dataset aggregations: 5-15 seconds
- Memory retrieval: < 1 second

### Memory Usage
- Basic session: ~50MB RAM
- Large conversation history: ~200MB RAM
- Multiple datasets: ~100MB per dataset
- Streamlit interface: ~150MB base

## Roadmap

### Upcoming Features
- [ ] Support for more database types (MongoDB, ClickHouse)
- [ ] Advanced visualization templates
- [ ] Collaborative session sharing
- [ ] API endpoint for external integration
- [ ] Mobile-responsive interface improvements

### Long-term Goals
- [ ] Enterprise deployment options
- [ ] Custom model fine-tuning
- [ ] Advanced security features
- [ ] Multi-language support
- [ ] Integration with BI tools

## Support and Community

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community Q&A and feature discussions
- **Documentation**: Comprehensive guides in the project wiki
- **Examples**: Sample datasets and query collections

### Contributing
We welcome contributions from the community! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

## Acknowledgments

- **LangChain**: Powerful framework for building AI applications
- **OpenAI**: Advanced language models powering the query generation
- **Streamlit**: Elegant framework for building data applications
- **Community Contributors**: Thanks to all who have contributed to this project

---
