from langchain_helper import get_few_shot_db_chain
import streamlit as st 

def extract_sql_from_steps(intermediate_steps):
    """Extract the clean SQL query from intermediate steps"""
    if not intermediate_steps or len(intermediate_steps) < 2:
        return "No SQL query found"
    
    # Method 1: Look for the sql_cmd in dictionary (usually at index 2)
    for step in intermediate_steps:
        if isinstance(step, dict) and 'sql_cmd' in step:
            return step['sql_cmd']
    
    # Method 2: The SQL query is usually the second item (index 1)
    if len(intermediate_steps) > 1:
        sql_candidate = intermediate_steps[1]
        if isinstance(sql_candidate, str) and sql_candidate.strip().upper().startswith('SELECT'):
            return sql_candidate
    
    return "Could not extract SQL query"

st.title("AtliQ T Shirts: Database Q&A")

question = st.text_input("Question")

if question:
    chain = get_few_shot_db_chain()
    if chain:
        try:
            response = chain(question)
            answer = response['result']
            st.write("**Answer:**")
            st.write(answer)
            
            # Show clean SQL query
            if st.checkbox("Show SQL query"):
                intermediate_steps = response.get('intermediate_steps', [])
                sql_query = extract_sql_from_steps(intermediate_steps)
                
                st.write("**Generated SQL Query:**")
                st.code(sql_query, language='sql')
            
            # Optional: Show full debug info
            if st.checkbox("Show full debug info"):
                st.write("**Full Intermediate Steps:**")
                st.code(str(response.get('intermediate_steps', 'No intermediate steps available')))
                
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Failed to initialize database connection")