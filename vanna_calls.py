import streamlit as st
import psycopg2
import pandas as pd
from vanna.remote import VannaDefault
import psutil
import time
from time import sleep
from datetime import datetime, timedelta
import queue
import threading

# Add these global variables after imports
REQUEST_QUEUE = queue.Queue()
LAST_REQUEST_TIME = datetime.now()
MIN_REQUEST_INTERVAL = 2  # seconds between requests
MAX_RETRIES = 3
CPU_THRESHOLD = 80  # Lower threshold for CPU usage

# Add after imports
PREDEFINED_QUERIES = {
    "Compare energy usage between September and October": """
        WITH monthly_stats AS (
            SELECT 
                tr.tag_id,
                t.tag_name,
                EXTRACT(MONTH FROM tr.date_time) as month,
                AVG(tr.tag_value) as avg_consumption,
                COUNT(*) as reading_count
            FROM tag_readings tr
            JOIN tags t ON tr.tag_id = t.tag_id
            JOIN tag_masters tm ON t.tag_id = tm.tag_id
            WHERE EXTRACT(MONTH FROM tr.date_time) IN (9, 10)
            AND EXTRACT(YEAR FROM tr.date_time) = EXTRACT(YEAR FROM CURRENT_DATE)
            GROUP BY tr.tag_id, t.tag_name, EXTRACT(MONTH FROM tr.date_time)
        )
        SELECT 
            tag_name,
            month,
            avg_consumption,
            reading_count
        FROM monthly_stats
        ORDER BY tag_name, month;
    """,
    
    "Show consumption from August 1st to October 31st": """
        SELECT 
            DATE_TRUNC('day', tr.date_time) as day,
            t.tag_name,
            AVG(tr.tag_value) as daily_consumption,
            COUNT(*) as readings_per_day
        FROM tag_readings tr
        JOIN tags t ON tr.tag_id = t.tag_id
        JOIN tag_masters tm ON t.tag_id = tm.tag_id
        WHERE tr.date_time BETWEEN 
            DATE_TRUNC('month', CURRENT_DATE - INTERVAL '2 months')
            AND DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day'
        GROUP BY DATE_TRUNC('day', tr.date_time), t.tag_name
        ORDER BY day, tag_name;
    """,
    
    "What's the usage trend for the past 30 days?": """
        WITH daily_usage AS (
            SELECT 
                DATE_TRUNC('day', tr.date_time) as day,
                t.tag_name,
                AVG(tr.tag_value) as avg_consumption,
                COUNT(*) as reading_count
            FROM tag_readings tr
            JOIN tags t ON tr.tag_id = t.tag_id
            WHERE tr.date_time >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE_TRUNC('day', tr.date_time), t.tag_name
        )
        SELECT 
            day,
            tag_name,
            avg_consumption,
            reading_count,
            AVG(avg_consumption) OVER (
                PARTITION BY tag_name 
                ORDER BY day 
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) as moving_avg_7_days
        FROM daily_usage
        ORDER BY tag_name, day;
    """,
    
    "Compare this week with last week": """
        WITH weekly_comparison AS (
            SELECT 
                tr.tag_id,
                t.tag_name,
                CASE 
                    WHEN tr.date_time >= DATE_TRUNC('week', CURRENT_DATE) 
                    THEN 'This Week'
                    ELSE 'Last Week'
                END as week_period,
                AVG(tr.tag_value) as avg_consumption,
                COUNT(*) as reading_count
            FROM tag_readings tr
            JOIN tags t ON tr.tag_id = t.tag_id
            JOIN tag_masters tm ON t.tag_id = tm.tag_id
            WHERE tr.date_time >= DATE_TRUNC('week', CURRENT_DATE - INTERVAL '1 week')
            GROUP BY tr.tag_id, t.tag_name, week_period
        )
        SELECT 
            tag_name,
            week_period,
            avg_consumption,
            reading_count
        FROM weekly_comparison
        ORDER BY tag_name, week_period DESC;
    """,
    
    "Show energy patterns between 9 AM and 5 PM": """
        WITH hourly_patterns AS (
            SELECT 
                tr.tag_id,
                t.tag_name,
                EXTRACT(HOUR FROM tr.date_time) as hour,
                AVG(tr.tag_value) as avg_consumption,
                COUNT(*) as reading_count,
                STDDEV(tr.tag_value) as consumption_variation
            FROM tag_readings tr
            JOIN tags t ON tr.tag_id = t.tag_id
            WHERE EXTRACT(HOUR FROM tr.date_time) BETWEEN 9 AND 17
            AND tr.date_time >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY tr.tag_id, t.tag_name, EXTRACT(HOUR FROM tr.date_time)
        )
        SELECT 
            tag_name,
            hour,
            avg_consumption,
            reading_count,
            consumption_variation
        FROM hourly_patterns
        ORDER BY tag_name, hour;
    """
}

QUERY_TEMPLATES = {
    "time_comparison": """
        WITH time_stats AS (
            SELECT 
                tr.tag_id,
                t.tag_name,
                {time_extract} as time_period,
                AVG(tr.tag_value) as avg_consumption,
                COUNT(*) as reading_count
            FROM tag_readings tr
            JOIN tags t ON tr.tag_id = t.tag_id
            WHERE {time_condition}
            GROUP BY tr.tag_id, t.tag_name, {time_extract}
        )
        SELECT 
            tag_name,
            time_period,
            avg_consumption,
            reading_count
        FROM time_stats
        ORDER BY tag_name, time_period;
    """,
    
    "daily_pattern": """
        SELECT 
            {time_grouping},
            t.tag_name,
            AVG(tr.tag_value) as avg_consumption,
            COUNT(*) as readings,
            STDDEV(tr.tag_value) as consumption_variation
        FROM tag_readings tr
        JOIN tags t ON tr.tag_id = t.tag_id
        WHERE {time_filter}
        GROUP BY {time_grouping}, t.tag_name
        ORDER BY {time_grouping}, tag_name;
    """
}

def rate_limit_check():
    """Enforce minimum time between requests"""
    global LAST_REQUEST_TIME
    current_time = datetime.now()
    time_since_last = (current_time - LAST_REQUEST_TIME).total_seconds()
    
    if time_since_last < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - time_since_last
        st.info(f"Rate limiting: waiting {sleep_time:.1f} seconds...")
        sleep(sleep_time)
    
    LAST_REQUEST_TIME = datetime.now()

def get_cpu_usage():
    """Get average CPU usage over 1 second"""
    return psutil.cpu_percent(interval=1)

def check_cpu_usage(threshold=CPU_THRESHOLD):
    """
    Enhanced CPU usage check with progressive backoff
    """
    cpu_usage = get_cpu_usage()
    
    if cpu_usage > threshold:
        wait_time = (cpu_usage - threshold) / 10  # Progressive wait based on CPU usage
        st.warning(f"High CPU usage detected ({cpu_usage:.1f}%). Cooling down...")
        sleep(wait_time)
        return True
    return False

def wait_for_cpu_cooldown():
    """Wait until CPU usage is below threshold"""
    retries = 0
    while retries < MAX_RETRIES:
        cpu_usage = get_cpu_usage()
        if cpu_usage < CPU_THRESHOLD:
            return True
        
        wait_time = 1 + (retries * 0.5)  # Progressive backoff
        st.warning(f"CPU at {cpu_usage:.1f}%. Waiting {wait_time:.1f}s...")
        sleep(wait_time)
        retries += 1
    
    return False

def check_memory_usage(threshold=85):
    """Check if memory usage is too high"""
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > threshold:
        st.warning(f"High memory usage ({memory_percent}%). Please wait...")
        sleep(2)
        return True
    return False

class EnergyQueryEngine(VannaDefault):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema_info = {}
        self._table_relationships = {}
        self._training_data = []
        self._questions = []

    def connect_to_database(self, db_config):
        """Connect to database and initialize schema"""
        try:
            # Test connection
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()

            # Define custom run_sql function
            def custom_run_sql(sql: str):
                try:
                    conn = psycopg2.connect(**db_config)
                    cur = conn.cursor()
                    cur.execute(sql)
                    columns = [desc[0] for desc in cur.description] if cur.description else []
                    rows = cur.fetchall()
                    cur.close()
                    conn.close()
                    return pd.DataFrame(rows, columns=columns)
                except Exception as e:
                    st.error(f"SQL execution error: {str(e)}")
                    return None

            # Set the custom run_sql function
            self.run_sql = custom_run_sql

            # Initialize schema after setting run_sql
            self._initialize_schema()
            return True

        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return False

    def _initialize_schema(self):
        """Initialize schema understanding"""
        try:
            # Get table relationships
            relationship_query = """
            SELECT
                tc.table_name as table_name,
                kcu.column_name as column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY';
            """
            
            # Get table columns
            columns_query = """
            SELECT 
                table_name,
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
            """
            
            # Execute queries
            relationships_df = self.run_sql(relationship_query)
            columns_df = self.run_sql(columns_query)
            
            # Build schema information
            for _, row in columns_df.iterrows():
                table = row['table_name']
                if table not in self._schema_info:
                    self._schema_info[table] = []
                self._schema_info[table].append({
                    'column': row['column_name'],
                    'type': row['data_type'],
                    'nullable': row['is_nullable']
                })
            
            # Build relationships
            for _, row in relationships_df.iterrows():
                if row['table_name'] not in self._table_relationships:
                    self._table_relationships[row['table_name']] = []
                self._table_relationships[row['table_name']].append({
                    'from_column': row['column_name'],
                    'to_table': row['foreign_table_name'],
                    'to_column': row['foreign_column_name']
                })
            
            # Create schema context for LLM
            schema_context = []
            for table, columns in self._schema_info.items():
                schema_context.append(f"Table: {table}")
                for col in columns:
                    schema_context.append(f"  - {col['column']} ({col['type']})")
                if table in self._table_relationships:
                    for rel in self._table_relationships[table]:
                        schema_context.append(f"  - Relates to {rel['to_table']} via {rel['from_column']} -> {rel['to_column']}")
            
            self._schema_context = "\n".join(schema_context)
            
            # Update example questions to match time-based analysis focus
            example_questions = [
                "Compare energy usage between September and October",
                "Show consumption from August 1st to October 31st",
                "What's the usage trend for the past 30 days?",
                "Compare this week with last week",
                "Show energy patterns between 9 AM and 5 PM"
            ]
            self._questions.extend(example_questions)
            
        except Exception as e:
            st.error(f"Error initializing schema: {str(e)}")

    def generate_sql(self, question, **kwargs):
        """Generate SQL using template-based approach"""
        try:
            # Check if question has a predefined query
            if question in PREDEFINED_QUERIES:
                return PREDEFINED_QUERIES[question]
            
            # For AI-generated queries, use rate limiting
            rate_limit_check()
            
            if not wait_for_cpu_cooldown():
                st.error("System is too busy. Please try again later.")
                return None
            
            # Create prompt with templates
            prompt = f"""
            Given this database schema:
            {self._schema_context}
            
            And these query templates:
            1. Time Comparison Template:
            {QUERY_TEMPLATES['time_comparison']}
            
            2. Daily Pattern Template:
            {QUERY_TEMPLATES['daily_pattern']}
            
            Generate a SQL query for: "{question}"
            
            Rules:
            1. Use the appropriate template based on the question type
            2. For time comparisons, use:
               - EXTRACT(HOUR FROM date_time) for hourly
               - DATE_TRUNC('day', date_time) for daily
               - EXTRACT(MONTH FROM date_time) for monthly
            3. Always join with tags table to get tag names
            4. Include reading counts for data quality
            5. Add STDDEV for variation analysis where appropriate
            6. Use WHERE conditions to limit time ranges
            7. Order results logically
            
            The query should be executable PostgreSQL.
            Return only the SQL query, no explanations.
            """
            
            # Get response from Ollama with retries
            for attempt in range(MAX_RETRIES):
                response = self._get_ollama_response(prompt)
                if response:
                    break
                sleep(1 * (attempt + 1))
            
            if not response:
                st.error("Failed to get response from Ollama after retries")
                return None
            
            # Extract and validate SQL
            sql = self._extract_sql(response)
            if sql and self._validate_sql(sql):
                return sql
            return None
            
        except Exception as e:
            st.error(f"Error generating SQL: {str(e)}")
            return None

    def _get_ollama_response(self, prompt):
        """Get response from Ollama"""
        import requests
        import json
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "phi",
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json()['response']
        except Exception as e:
            st.error(f"Error calling Ollama: {str(e)}")
            return None

    def _extract_sql(self, response):
        """Extract SQL query from Ollama response"""
        if not response:
            return None
            
        # Clean up the response and extract SQL
        sql = response.strip()
        if sql.lower().startswith('select'):
            return sql
        return None

    def generate_questions(self):
        """Return predefined questions"""
        return self._questions

    def _validate_sql(self, sql):
        """Basic SQL validation"""
        required_elements = [
            'SELECT',
            'FROM tag_readings',
            'JOIN tags',
            'GROUP BY',
            'ORDER BY'
        ]
        
        sql_upper = sql.upper()
        return all(element.upper() in sql_upper for element in required_elements)

def get_database_schema():
    """Get the complete database schema including relationships"""
    schema_query = """
    WITH fk_info AS (
        SELECT
            tc.table_schema, 
            tc.constraint_name, 
            tc.table_name, 
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM 
            information_schema.table_constraints AS tc 
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
    )
    SELECT 
        t.table_name,
        array_agg(
            c.column_name || ' ' || 
            c.data_type || 
            CASE 
                WHEN c.is_nullable = 'NO' THEN ' NOT NULL'
                ELSE ''
            END ||
            CASE 
                WHEN pk.column_name IS NOT NULL THEN ' PRIMARY KEY'
                ELSE ''
            END ||
            CASE 
                WHEN fk.foreign_table_name IS NOT NULL 
                THEN ' REFERENCES ' || fk.foreign_table_name || '(' || fk.foreign_column_name || ')'
                ELSE ''
            END
        ) as columns
    FROM 
        information_schema.tables t
        JOIN information_schema.columns c ON t.table_name = c.table_name
        LEFT JOIN (
            SELECT kcu.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu 
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
        ) pk ON t.table_name = pk.table_name AND c.column_name = pk.column_name
        LEFT JOIN fk_info fk 
            ON t.table_name = fk.table_name 
            AND c.column_name = fk.column_name
    WHERE 
        t.table_schema = 'public'
    GROUP BY 
        t.table_name
    ORDER BY 
        t.table_name;
    """
    
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="rds",
            user="postgres",
            password="mysecretpassword",
            port=5432
        )
        cur = conn.cursor()
        cur.execute(schema_query)
        schema = cur.fetchall()
        cur.close()
        conn.close()
        
        # Print schema for analysis
        print("\nDatabase Schema:")
        for table, columns in schema:
            print(f"\nTable: {table}")
            for col in columns:
                print(f"  {col}")
        
        return schema
    except Exception as e:
        print(f"Error getting schema: {str(e)}")
        return None

@st.cache_resource
def setup_vanna():
    """Initialize Vanna with resource checks"""
    # Print schema first
    print("\n=== DATABASE SCHEMA ===")
    schema_df = get_schema_info()
    if schema_df is not None:
        for _, row in schema_df.iterrows():
            print(f"\n{row['table_name']}.{row['column_name']}")
            print(f"  Type: {row['data_type']}")
            print(f"  Nullable: {row['is_nullable']}")
    print("\n====================\n")
    
    if check_cpu_usage() or check_memory_usage():
        st.info("System is busy, waiting for resources...")
        if not wait_for_cpu_cooldown():
            st.error("System resources are exhausted. Please try again later.")
            return None
        
    db_config = {
        "host": st.secrets["POSTGRES_HOST"],
        "database": st.secrets["POSTGRES_DB"],
        "user": st.secrets["POSTGRES_USER"],
        "password": st.secrets["POSTGRES_PASSWORD"],
        "port": st.secrets["POSTGRES_PORT"]
    }
    
    try:
        vn = EnergyQueryEngine(
            api_key="dummy-key",
            model="phi",
            config={
                "model_type": "ollama",
                "model_name": "phi",
                "base_url": "http://localhost:11434/api"
            }
        )
        
        if not vn.connect_to_database(db_config):
            raise Exception("Failed to connect to database")
        
        return vn
        
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None

# Cached function wrappers
@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    vn = setup_vanna()
    if vn is None:
        return None
    return vn.generate_sql(question=question)

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    # Rate limiting check
    rate_limit_check()
    
    # Wait for CPU to cool down
    if not wait_for_cpu_cooldown():
        st.error("System is too busy. Please try again later.")
        return None
    
    vn = setup_vanna()
    if vn is None:
        return None
    return vn.run_sql(sql=sql)

@st.cache_data(show_spinner="Generating questions ...")
def generate_questions_cached():
    vn = setup_vanna()
    if vn is None:
        return []
    return vn.generate_questions()

def process_query(query):
    # Add CPU check before heavy operations
    if check_cpu_usage():
        time.sleep(0.5)  # Add small delay to let CPU cool down
    
    # Your existing processing code...
    
    # Add periodic CPU checks during intensive operations
    if check_cpu_usage():
        time.sleep(0.5)

def get_schema_info():
    """Get detailed schema information for query optimization"""
    schema_query = """
    SELECT 
        table_name,
        column_name,
        data_type,
        is_nullable
    FROM information_schema.columns 
    WHERE table_schema = 'public'
    AND table_name IN ('tag_readings', 'tags', 'tag_masters', 'meter_details', 'consumer_details')
    ORDER BY table_name, ordinal_position;
    """
    
    try:
        conn = psycopg2.connect(
            host=st.secrets["POSTGRES_HOST"],
            database=st.secrets["POSTGRES_DB"],
            user=st.secrets["POSTGRES_USER"],
            password=st.secrets["POSTGRES_PASSWORD"],
            port=st.secrets["POSTGRES_PORT"]
        )
        
        df = pd.read_sql(schema_query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error getting schema: {str(e)}")
        return None re