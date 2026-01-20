"""
SQL Service - Vanna 2.0 Agent Framework Implementation
Handles Text-to-SQL conversion using Vanna.ai 2.0 with OpenAI and PostgreSQL.
"""

from typing import Dict, Any, List, Optional
import uuid
import asyncio
import pandas as pd
import logging

logger = logging.getLogger("rag_app.sql_service")

# Vanna 2.0 Agent Framework imports
from vanna import Agent
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.postgres import PostgresRunner
from vanna.core.registry import ToolRegistry
from vanna.tools import RunSqlTool
from vanna.core.user import UserResolver, User, RequestContext

# Pinecone integration for Agent Memory
try:
    from vanna.integrations.pinecone import PineconeAgentMemory
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    # Fallback to local memory
    from vanna.integrations.local.agent_memory import DemoAgentMemory

from app.config import settings


class SimpleUserResolver(UserResolver):
    """Simple user resolver for SQL service - grants full access."""

    async def resolve_user(self, request_context: RequestContext) -> User:
        return User(
            id="sql_service_user",
            email="sql@service.local",
            group_memberships=['user', 'admin']  # Full access to SQL tools
        )


class VannaAgentWrapper:
    """
    Wrapper around Vanna 2.0 Agent for synchronous use in FastAPI.
    Handles async-to-sync conversion and component extraction.
    """

    def __init__(self, openai_api_key: str, database_url: str, pinecone_api_key: Optional[str] = None):
        """
        Initialize Vanna 2.0 Agent with all components.

        Args:
            openai_api_key: OpenAI API key for GPT-4o
            database_url: PostgreSQL connection string
            pinecone_api_key: Optional Pinecone API key for persistent memory
        """
        # Initialize OpenAI LLM with GPT-4o
        self.llm = OpenAILlmService(
            api_key=openai_api_key,
            model=settings.VANNA_MODEL  # "gpt-4o"
        )

        # Monkey-patch LLM to inject determinism parameters
        # This ensures consistent SQL generation across multiple runs
        logger.info(
            f"Configuring SQL LLM with deterministic settings: "
            f"temperature={settings.VANNA_TEMPERATURE}, "
            f"top_p={settings.VANNA_TOP_P}, "
            f"seed={settings.VANNA_SEED}"
        )

        # Store original _build_payload method
        original_build_payload = self.llm._build_payload

        # Create wrapper that injects determinism parameters
        def deterministic_build_payload(request):
            """Wraps Vanna's _build_payload to add temperature, top_p, and seed."""
            payload = original_build_payload(request)

            # Inject determinism parameters
            payload['temperature'] = settings.VANNA_TEMPERATURE
            payload['top_p'] = settings.VANNA_TOP_P
            payload['seed'] = settings.VANNA_SEED

            # Override max_tokens if configured
            if settings.VANNA_MAX_TOKENS:
                payload['max_tokens'] = settings.VANNA_MAX_TOKENS

            logger.debug(f"SQL LLM payload: {payload}")
            return payload

        # Replace the method with our wrapper
        self.llm._build_payload = deterministic_build_payload

        # Initialize PostgreSQL Runner
        self.postgres_runner = PostgresRunner(
            connection_string=database_url
        )

        # Create tool registry with RunSqlTool
        self.tools = ToolRegistry()
        self.tools.register_local_tool(
            RunSqlTool(sql_runner=self.postgres_runner),
            access_groups=['user', 'admin']
        )

        # Create user resolver
        self.user_resolver = SimpleUserResolver()

        # Initialize Agent Memory (Pinecone or local)
        if PINECONE_AVAILABLE and pinecone_api_key:
            logger.info(f"Using Pinecone for SQL Agent memory (index: {settings.VANNA_PINECONE_INDEX})")
            self.memory = PineconeAgentMemory(
                api_key=pinecone_api_key,
                index_name=settings.VANNA_PINECONE_INDEX,
                environment="us-east-1",  # Match PINECONE_ENVIRONMENT
                dimension=1536,  # OpenAI text-embedding-3-small dimension
                metric="cosine"
            )
        else:
            logger.warning("Using in-memory storage for SQL Agent (data will not persist)")
            self.memory = DemoAgentMemory()

        # Create Agent
        self.agent = Agent(
            llm_service=self.llm,
            tool_registry=self.tools,
            user_resolver=self.user_resolver,
            agent_memory=self.memory
        )

        logger.info("✓ Vanna 2.0 Agent initialized successfully")

    async def generate_sql_async(self, question: str, schema_context: str = "") -> str:
        """
        Generate SQL from natural language question (async).

        Args:
            question: Natural language question
            schema_context: Database schema documentation

        Returns:
            Generated SQL query string

        Raises:
            ValueError: If Agent fails to generate SQL
        """
        # Prepare full message with schema context
        if schema_context:
            full_message = f"{schema_context}\n\nQUESTION: {question}"
        else:
            full_message = question

        # Extract SQL from Agent
        return await self._extract_sql_from_agent(full_message)

    async def _extract_sql_from_agent(self, message: str) -> str:
        """
        Extract SQL from Agent's UI components.

        Args:
            message: Full message including schema context and question

        Returns:
            Extracted SQL query

        Raises:
            ValueError: If no SQL found in Agent response
        """
        request_context = RequestContext()
        sql = None

        # Iterate through Agent's streaming UI components
        async for component in self.agent.send_message(
            request_context=request_context,
            message=message
        ):
            rich_comp = component.rich_component

            # Extract SQL from StatusCard metadata (primary source)
            if hasattr(rich_comp, 'metadata') and rich_comp.metadata:
                if 'sql' in rich_comp.metadata:
                    sql = rich_comp.metadata['sql']

            # Fallback: Extract from SQL code blocks
            if hasattr(rich_comp, 'content') and rich_comp.content:
                content = str(rich_comp.content)
                # Look for SQL in markdown code blocks
                if '```sql' in content.lower():
                    # Extract SQL from code block
                    parts = content.split('```')
                    for part in parts:
                        if part.strip().lower().startswith('sql'):
                            sql = part[3:].strip()  # Remove 'sql' prefix

        if not sql:
            raise ValueError("Agent did not generate SQL. Please try rephrasing your question.")

        return sql

    async def execute_sql_async(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute SQL and return results (async).

        Args:
            sql: SQL query to execute

        Returns:
            List of row dictionaries

        Raises:
            Exception: If SQL execution fails
        """
        return await self._execute_and_extract_results(sql)

    async def _execute_and_extract_results(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute SQL directly using psycopg2 and return results.

        PostgresRunner.run_sql() is designed to be called by the Agent as a Tool,
        not directly. For manual SQL execution, we use psycopg2 directly.

        Args:
            sql: SQL query to execute

        Returns:
            List of row dictionaries
        """
        logger.info(f"Executing SQL directly: {sql[:100]}...")

        try:
            import psycopg2
            import psycopg2.extras
            import socket
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

            # Parse connection string and force IPv4 for Lambda compatibility
            # AWS Lambda doesn't support IPv6 outbound connections
            conn_str = self.postgres_runner.connection_string

            # Parse the connection URL
            parsed = urlparse(conn_str)
            hostname = parsed.hostname

            # Force IPv4 resolution by resolving hostname to IPv4 address
            # This prevents psycopg2 from trying to use IPv6
            try:
                logger.debug(f"Resolving hostname {hostname} to IPv4...")
                # Get only IPv4 addresses (AF_INET)
                addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET)
                ipv4_address = addr_info[0][4][0]
                logger.info(f"Resolved {hostname} to IPv4: {ipv4_address}")

                # Replace hostname with IPv4 address in connection string
                conn_str = conn_str.replace(hostname, ipv4_address)
            except socket.gaierror as e:
                logger.warning(f"Failed to resolve hostname to IPv4: {e}, using original hostname")

            # Connect to database using the modified connection string
            conn = psycopg2.connect(conn_str)

            try:
                # Execute query
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute(sql)

                # Fetch results
                rows = cursor.fetchall()

                # Convert RealDictRow objects to regular dicts
                results = [dict(row) for row in rows]

                cursor.close()
                conn.close()

                logger.info(f"✓ SQL executed successfully: {len(results)} rows returned")
                return results

            except Exception as e:
                conn.close()
                raise e

        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise ValueError(f"Failed to execute SQL: {str(e)}")


class TextToSQLService:
    """
    Service for converting natural language to SQL using Vanna.ai 2.0 Agent Framework.
    Maintains compatibility with existing FastAPI endpoints.
    """

    def __init__(self, database_url: str | None = None, openai_api_key: str | None = None, query_cache_service=None):
        """
        Initialize the Text-to-SQL service with Vanna 2.0 Agent.

        Args:
            database_url: PostgreSQL connection string
            openai_api_key: OpenAI API key
            query_cache_service: Optional QueryCacheService for SQL caching

        Raises:
            ValueError: If required credentials are missing
        """
        self.database_url = database_url or settings.DATABASE_URL
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.query_cache_service = query_cache_service  # Optional cache service

        # Validation
        if not self.database_url:
            raise ValueError("DATABASE_URL is required for Text-to-SQL features")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for Text-to-SQL features")

        # Initialize Vanna 2.0 Agent wrapper
        pinecone_key = settings.PINECONE_API_KEY if PINECONE_AVAILABLE else None
        self.vanna = VannaAgentWrapper(
            openai_api_key=self.openai_api_key,
            database_url=self.database_url,
            pinecone_api_key=pinecone_key
        )

        # Approval workflow state (in-memory)
        self.pending_queries: Dict[str, Dict[str, Any]] = {}

        # Training flag
        self.is_trained = False

        # Schema context (replaces traditional Vanna training)
        self.schema_context = ""

    def complete_training(self):
        """
        Prepare schema context for Vanna 2.0.
        Note: Vanna 2.0 Agent doesn't use the same training approach as legacy Vanna.
        Instead, we provide schema context with each query.
        """
        logger.info("Preparing schema context for Vanna 2.0...")

        # Build comprehensive schema context
        self.schema_context = self._build_schema_context()

        self.is_trained = True
        logger.info("✓ Schema context prepared for Vanna 2.0 Agent!")

    def _build_schema_context(self) -> str:
        """
        Build comprehensive schema context by querying database.
        Provides same information as legacy Vanna training.

        Returns:
            Formatted schema documentation string
        """
        schema_parts = []

        # Header
        schema_parts.append("DATABASE SCHEMA DOCUMENTATION")
        schema_parts.append("=" * 60)

        # Database overview
        documentation = """
This is an e-commerce database with three main tables:
- customers: Contains customer information including name, email, segment (SMB, Enterprise, Individual), and country
- products: Product catalog with name, category, price, stock quantity, and description
- orders: Customer orders with order date, total amount, status (Pending, Delivered, Cancelled, Processing), and shipping address

The customers table has a one-to-many relationship with orders (one customer can have many orders).

IMPORTANT NOTES:
- For order revenue/pricing, use orders.total_amount (NOT 'price')
- Customer segments: 'SMB', 'Enterprise', 'Individual' (case-sensitive)
- Order statuses: 'Pending', 'Delivered', 'Cancelled', 'Processing' (case-sensitive)
- To join customers and orders: JOIN orders ON customers.id = orders.customer_id
"""
        schema_parts.append(documentation)

        # Table schemas
        schema_parts.append("\nTABLE SCHEMAS:")
        schema_parts.append("-" * 60)

        schema_parts.append("""
Table: customers
Columns:
  - id (SERIAL PRIMARY KEY)
  - name (VARCHAR) - Customer full name
  - email (VARCHAR) - Customer email address
  - segment (VARCHAR) - One of: 'SMB', 'Enterprise', 'Individual'
  - country (VARCHAR) - Customer country
  - created_at (TIMESTAMP)
  - updated_at (TIMESTAMP)
""")

        schema_parts.append("""
Table: products
Columns:
  - id (SERIAL PRIMARY KEY)
  - name (VARCHAR) - Product name
  - category (VARCHAR) - Product category (Electronics, Software, Hardware, etc.)
  - price (DECIMAL) - Product unit price
  - stock_quantity (INT) - Current inventory count
  - description (TEXT)
  - created_at (TIMESTAMP)
  - updated_at (TIMESTAMP)
""")

        schema_parts.append("""
Table: orders
Columns:
  - id (SERIAL PRIMARY KEY)
  - customer_id (INT) - Foreign key to customers.id
  - order_date (DATE) - Date of order
  - total_amount (DECIMAL) - TOTAL ORDER PRICE (use this for revenue, NOT 'price'!)
  - status (VARCHAR) - One of: 'Pending', 'Delivered', 'Cancelled', 'Processing'
  - shipping_address (TEXT)
  - created_at (TIMESTAMP)
  - updated_at (TIMESTAMP)
""")

        # Golden examples
        schema_parts.append("\nEXAMPLE QUERIES:")
        schema_parts.append("-" * 60)

        examples = [
            ("How many customers do we have?", "SELECT COUNT(*) as customer_count FROM customers;"),
            ("What is the total revenue from all orders?", "SELECT SUM(total_amount) as total_revenue FROM orders;"),
            ("List all delivered orders", "SELECT * FROM orders WHERE status = 'Delivered' ORDER BY order_date DESC;"),
            ("How many orders per customer segment?", "SELECT c.segment, COUNT(o.id) as order_count FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.segment;"),
            ("Top 10 customers by total spending", "SELECT c.name, c.email, SUM(o.total_amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name, c.email ORDER BY total_spent DESC LIMIT 10;"),
        ]

        for i, (question, sql) in enumerate(examples, 1):
            schema_parts.append(f"\nExample {i}:")
            schema_parts.append(f"Question: {question}")
            schema_parts.append(f"SQL: {sql}")

        return "\n".join(schema_parts)

    async def generate_sql_for_approval(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL from a natural language question using Vanna 2.0 Agent.
        Returns SQL for user approval before execution.

        NEW: Implements SQL generation caching to save ~$0.08 per cache hit.
        - Cache key: hash(question)
        - Cache TTL: 24 hours (schema relatively stable)
        - Falls back to uncached if Redis unavailable

        Args:
            question: Natural language question

        Returns:
            Dictionary with query_id, question, SQL, status, and cache_hit indicator

        Raises:
            Exception: If schema context not prepared or SQL generation fails
        """
        if not self.is_trained:
            raise Exception("Schema context not prepared. Call complete_training() first.")

        # Check cache first (if cache service is available)
        if self.query_cache_service and self.query_cache_service.enabled:
            cache_key = self.query_cache_service.get_sql_gen_key(question)
            cached_result = self.query_cache_service.get(cache_key, cache_type="sql_gen")

            if cached_result and "sql" in cached_result:
                logger.info(f"SQL generation cache HIT for question: '{question[:50]}...'")

                # Create new query ID for approval workflow (even for cached SQL)
                query_id = str(uuid.uuid4())

                # Store in pending queries
                self.pending_queries[query_id] = {
                    'question': question,
                    'sql': cached_result["sql"],
                    'status': 'pending_approval',
                    'generated_at': pd.Timestamp.now().isoformat(),
                    'cache_hit': True
                }

                return {
                    'query_id': query_id,
                    'question': question,
                    'sql': cached_result["sql"],
                    'explanation': cached_result.get("explanation", "This SQL will retrieve data from your database. Please review before approving."),
                    'status': 'pending_approval',
                    'cache_hit': True,
                    'cost_saved': "$0.08"  # Approximate GPT-4o cost per SQL generation
                }

        try:
            # Generate SQL using Vanna 2.0 Agent
            sql = await self.vanna.generate_sql_async(
                question=question,
                schema_context=self.schema_context
            )

            explanation = "This SQL will retrieve data from your database. Please review before approving."

            # Cache the SQL generation result (if cache service is available)
            if self.query_cache_service and self.query_cache_service.enabled:
                cache_key = self.query_cache_service.get_sql_gen_key(question)
                cache_value = {
                    "sql": sql,
                    "explanation": explanation,
                    "question": question
                }
                ttl = settings.CACHE_TTL_SQL_GEN  # Default: 24 hours
                self.query_cache_service.set(cache_key, cache_value, ttl=ttl, cache_type="sql_gen")
                logger.info(f"SQL generation cache MISS - cached for '{question[:50]}...' (TTL: {ttl}s)")

            # Create unique query ID for approval workflow
            query_id = str(uuid.uuid4())

            # Store pending query
            self.pending_queries[query_id] = {
                'question': question,
                'sql': sql,
                'status': 'pending_approval',
                'generated_at': pd.Timestamp.now().isoformat(),
                'cache_hit': False
            }

            return {
                'query_id': query_id,
                'question': question,
                'sql': sql,
                'explanation': explanation,
                'status': 'pending_approval',
                'cache_hit': False,
                'cost_saved': "$0.00"
            }

        except Exception as e:
            raise Exception(f"Failed to generate SQL: {str(e)}")

    async def execute_approved_query(self, query_id: str, approved: bool) -> Dict[str, Any]:
        """
        Execute a SQL query after user approval using Vanna 2.0 Agent.

        NEW: Implements SQL result caching for SELECT queries.
        - Cache key: hash(normalized_sql)
        - Cache TTL: 15 minutes (data changes frequently)
        - Only caches read-only SELECT queries

        Args:
            query_id: ID of the pending query
            approved: Whether the user approved execution

        Returns:
            Dictionary with results or rejection message, plus cache_hit indicator
        """
        if query_id not in self.pending_queries:
            return {
                'error': 'Query ID not found',
                'status': 'error'
            }

        query_info = self.pending_queries[query_id]

        if not approved:
            # User rejected the query
            del self.pending_queries[query_id]
            return {
                'query_id': query_id,
                'status': 'rejected',
                'message': 'Query execution cancelled by user'
            }

        sql = query_info['sql']

        # Check if this is a SELECT query (safe to cache)
        is_select_query = sql.strip().upper().startswith("SELECT")

        # Check cache for SELECT queries only
        if is_select_query and self.query_cache_service and self.query_cache_service.enabled:
            cache_key = self.query_cache_service.get_sql_result_key(sql)
            cached_result = self.query_cache_service.get(cache_key, cache_type="sql_result")

            if cached_result and "results" in cached_result:
                logger.info(f"SQL result cache HIT for query: '{sql[:50]}...'")

                # Clean up pending query
                del self.pending_queries[query_id]

                return {
                    'query_id': query_id,
                    'question': query_info['question'],
                    'sql': sql,
                    'results': cached_result["results"],
                    'result_count': cached_result["result_count"],
                    'status': 'executed',
                    'cache_hit': True,
                    'cached_at': cached_result.get("executed_at")
                }

        # Execute the SQL using Vanna 2.0 Agent
        try:
            results = await self.vanna.execute_sql_async(sql)

            # Cache SELECT query results (if cache service is available)
            if is_select_query and self.query_cache_service and self.query_cache_service.enabled:
                cache_key = self.query_cache_service.get_sql_result_key(sql)
                cache_value = {
                    "results": results,
                    "result_count": len(results),
                    "sql": sql,
                    "executed_at": pd.Timestamp.now().isoformat()
                }
                ttl = settings.CACHE_TTL_SQL_RESULT  # Default: 15 minutes
                self.query_cache_service.set(cache_key, cache_value, ttl=ttl, cache_type="sql_result")
                logger.info(f"SQL result cache MISS - cached for '{sql[:50]}...' (TTL: {ttl}s)")

            # Clean up pending query
            del self.pending_queries[query_id]

            return {
                'query_id': query_id,
                'question': query_info['question'],
                'sql': sql,
                'results': results,
                'result_count': len(results),
                'status': 'executed',
                'cache_hit': False
            }

        except Exception as e:
            return {
                'query_id': query_id,
                'error': str(e),
                'status': 'error'
            }

    def get_pending_queries(self) -> List[Dict[str, Any]]:
        """
        Get list of all pending queries awaiting approval.

        Returns:
            List of pending query information
        """
        return [
            {
                'query_id': qid,
                **info
            }
            for qid, info in self.pending_queries.items()
        ]
