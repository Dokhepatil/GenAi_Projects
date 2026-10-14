import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
from sqlalchemy import create_engine, text

# 1. Load variables from the local .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

# 2. Replicate the LangChain prompt template logic natively (Fixed for dict/JSON inputs)
class CleanPromptTemplate:
    def __init__(self, messages):
        self.messages = messages

    def format(self, user_input_dict):
        formatted_messages = []
        for role, text_content in self.messages:
            # Safely replace keys instead of using .format() to avoid curly-brace errors
            formatted_text = text_content
            for key, value in user_input_dict.items():
                formatted_text = formatted_text.replace(f"{{{key}}}", str(value))
            formatted_messages.append({"role": role, "content": formatted_text})
        return formatted_messages


# 3. Replicate the LangChain ChatGroq wrapper logic natively
class NATIVE_ChatGroq:
    def __init__(self, model, api_key):
        self.model = model
        # Initialize the direct working client internally
        self.client = Groq(api_key=api_key)

    def invoke(self, formatted_messages):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages
        )
        # Wrap response content in an object matching LangChain's structure (.content)
        class ResponseObject:
            def __init__(self, content):
                self.content = content
        return ResponseObject(completion.choices[0].message.content)

# 4. Replicate the LangChain Pipe operator (|) connection logic natively
class NativeChain:
    def __init__(self, prompt, llm):
        self.prompt = prompt
        self.llm = llm

    def invoke(self, input_data):
        formatted = self.prompt.format(input_data)
        return self.llm.invoke(formatted)

# 5. Database Configuration 
DB_USER = "root"
DB_PASSWORD = "root"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "genai"

# Create a connection engine matching your configuration syntax
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

schema = """
customer (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    join_date TEXT
)

sales (
    sale_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product TEXT,
    amount REAL,
    sale_date TEXT,
    FOREIGN KEY(customer_id) REFERENCES customer(customer_id)
)
"""

# 6. Native JSON Output Parsing Helper Function
def native_parse_json(text_data: str) -> dict:
    """Finds and parses a JSON block inside a string markdown container."""
    try:
        json_match = re.search(r"\{.*\}", text_data, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(text_data)
    except Exception:
        return {}

# 7. Replicated Native generate SQL function with structural isolation
def create_query(question: str) -> str:
    schema_prompt = CleanPromptTemplate([
        ("system", "Given the database schema provided, create a valid MySQL query that will answer the question. "
                   "You must respond ONLY with a raw JSON block containing a single key 'query'. "
                   "Do not include any extra introductory text, markdown explanations, or pleasantries."),
        ("user", "Schema:\n{schema}\n\nQuestion:\n{question}\n\nRespond using valid JSON matching: {{\x22query\x22: \x22YOUR_SQL_QUERY_HERE\x22}}")
    ])
    
    schema_chain = NativeChain(schema_prompt, llm)
    
    try:
        response = schema_chain.invoke({
            "schema": schema,
            "question": question
        })
        parsed_json = native_parse_json(response.content)
        return parsed_json.get("query")
    except Exception as e:
        st.error(f"create_query Error: {e}")
        return None

# 8. Execute SQL via SQLAlchemy engine matching your functional spec
def run_query(query: str) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if df.empty:
                return pd.DataFrame(columns=["Status"], data=[["Query executed successfully, but returned 0 rows."]])
            return df
    except Exception as e:
        return pd.DataFrame([[str(e)]], columns=["Database Error"])

# 9. AI Logic: Interpret Results natively without LangChain imports
def interpret_results(question: str, results_df: pd.DataFrame) -> str:
    results_str = str(results_df.to_dict(orient="records"))
    
    summary_prompt = CleanPromptTemplate([
        ("system", "Given the provided database records, generate a concise, human-readable answer answering the user's question directly."),
        ("user", f"Data:\n{results_str}\n\nQuestion: {question}")
    ])
    
    summary_chain = NativeChain(summary_prompt, llm)
    
    try:
        response = summary_chain.invoke({})
        return response.content
    except Exception as e:
        return f"Could not generate automated summary: {str(e)}"


# Streamlit UI Setup
st.set_page_config(page_title="AI SQL Assistant", layout="wide")
st.title("AI-Powered MySQL Assistant")

question = st.text_input("Ask a question about your Sales or Customers:", placeholder="e.g., Who is our top customer by spend?")

# Initialize the global LLM component configuration 
llm = NATIVE_ChatGroq(
    model='openai/gpt-oss-120b', 
    api_key=API_KEY
)

# 10. Button Logic execution workflow
if st.button("Run Analysis"):
    if question.strip():
        with st.spinner("Generating SQL..."):
            generated_sql = create_query(question)
        
        if generated_sql:
            st.markdown("Generated SQL")
            st.code(generated_sql, language="sql")

            with st.spinner("Executing Query..."):
                results = run_query(generated_sql)

            st.markdown("Data Results")
            st.dataframe(results, use_container_width=True)

            if "Database Error" not in results.columns:
                with st.spinner("Summarizing..."):
                    answer = interpret_results(question, results)
                st.markdown("AI Insights")
                st.success(answer)
    else:
        st.warning("Please enter a question first.")
