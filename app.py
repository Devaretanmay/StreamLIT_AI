import streamlit as st
from vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    PREDEFINED_QUERIES
)

st.set_page_config(layout="wide")

st.title("Energy Consumption Analyzer")

# Energy-specific test queries
test_queries = [
    "Compare energy usage between September and October",
    "Show consumption from August 1st to October 31st", 
    "What's the usage trend for the past 30 days?",
    "Compare this week with last week",
    "Show energy patterns between 9 AM and 5 PM",
    "What's the consumption difference between Q2 and Q3?",
    "Compare morning vs evening usage",
    "Show peak hours consumption this month",
    "What's the weekend vs weekday usage pattern?",
    "Compare first half vs second half of October"
]

# Sidebar with clear categories
st.sidebar.title("📊 Analysis Templates")
for query in PREDEFINED_QUERIES.keys():
    if st.sidebar.button(f"⚡ {query}", help="Optimized energy analysis"):
        st.session_state["my_question"] = query

st.sidebar.title("🔍 Custom Analysis")
for query in [q for q in test_queries if q not in PREDEFINED_QUERIES]:
    if st.sidebar.button(f"📈 {query}", help="AI-powered analysis"):
        st.session_state["my_question"] = query

# Main interface
my_question = st.text_input("What would you like to analyze about your energy consumption?", 
                           value=st.session_state.get("my_question", ""),
                           key="question_input")

if my_question:
    st.write("### Question:")
    st.write(my_question)
    
    # Show if using predefined or generated query
    is_predefined = my_question in PREDEFINED_QUERIES
    st.info(f"Using {'pre-optimized' if is_predefined else 'AI-generated'} query")
    
    # Generate and show SQL
    sql = generate_sql_cached(question=my_question)
    if sql:
        st.write("### Generated SQL:")
        st.code(sql, language="sql")
        
        # Execute SQL and show results
        df = run_sql_cached(sql=sql)
        if df is not None:
            st.write("### Results:")
            st.write(f"Found {len(df)} rows")
            st.dataframe(df)
            
            # Add visualization hints
            if len(df) > 0:
                if 'day' in df.columns or 'hour' in df.columns or 'month' in df.columns:
                    st.line_chart(df.set_index(df.columns[0])[df.columns[1]])
        else:
            st.error("Query execution failed")
    else:
        st.error("Could not generate SQL for that question")
