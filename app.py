import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from decouple import config
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, SequentialChain
import pandas as pd

# LLM model
llm = ChatGroq(model="llama-3.2-90b-text-preview")
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4)

sql_prompt_template = PromptTemplate(
    input_variables=["user_query"],
    template="Generate a SQL query for the following request: {user_query}"
)

# Create chain for SQL query generation
sql_chain = LLMChain(llm=llm, prompt=sql_prompt_template)

# Prompt for generating explanation
explanation_prompt_template = PromptTemplate(
    input_variables=["sql_query"],
    template="Explain the following SQL query in simple terms: {sql_query}"
)

# Create chain for explanation generation
explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt_template)

def sample_data():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 35, 40],
        "city": ["New York", "Los Angeles", "Chicago", "Houston"]
    })

st.set_page_config(
    page_title="ChatGPT Clone",
    page_icon="🤖",
    layout="wide"
)

st.title("ChatGPT Clone")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": [{"type": "text", "content": "Hello there, I'm a ChatGPT clone"}]}
    ]
if "query_run" not in st.session_state:
    st.session_state.query_run = False
if "query_result" not in st.session_state:
    st.session_state.query_result = None
if "current_sql_query" not in st.session_state:
    st.session_state.current_sql_query = ''
if "current_explanation" not in st.session_state:
    st.session_state.current_explanation = ''
if "last_user_prompt" not in st.session_state:
    st.session_state.last_user_prompt = ''

# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        for content_block in message["content"]:
            if content_block["type"] == "text":
                st.write(content_block["content"])
            elif content_block["type"] == "code":
                st.code(content_block["content"], language=content_block.get("language", ""))
            elif content_block["type"] == "dataframe":
                # Reconstruct the DataFrame from the stored dictionary
                df = pd.DataFrame(content_block["content"])
                st.dataframe(df)
            else:
                st.write(content_block["content"])  # Fallback for any other types

user_prompt = st.chat_input()

if user_prompt:
    st.session_state.last_user_prompt = user_prompt
    # Reset state variables when new input is given
    st.session_state.query_run = False
    st.session_state.query_result = None
    st.session_state.current_sql_query = ''
    st.session_state.current_explanation = ''
    st.session_state.messages.append({"role": "user", "content": [{"type": "text", "content": user_prompt}]})
    with st.chat_message("user"):
        st.write(user_prompt)
def generate_query():
    # Generate SQL query
    sql_query = sql_chain.predict(user_query=st.session_state.last_user_prompt)
    # Extract the SQL query if it's enclosed in code blocks
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
    else:
        sql_query = sql_query.strip()
    return sql_query
def run_query():
    st.session_state.query_run = True
    try:
        st.write("Executing the SQL query...")
        # For demonstration, use sample data instead of actual database query
        df = sample_data()
        st.session_state.query_result = df.to_dict()

        # Save the results in a new assistant's message
        result_message = {
            "role": "assistant",
            "content": [
                {"type": "text", "content": "Here are the results of your query:"},
                {"type": "dataframe", "content": st.session_state.query_result}
            ]
        }
        st.session_state.messages.append(result_message)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        # Save the error message in the assistant's messages
        error_message = {
            "role": "assistant",
            "content": [
                {"type": "text", "content": "An error occurred while executing the query."},
                {"type": "text", "content": str(e)}
            ]
        }
        st.session_state.messages.append(error_message)
    finally:
        # Reset state variables after query execution
        st.session_state.current_sql_query = ''
        st.session_state.current_explanation = ''

# Check if we need to generate a new assistant response
if st.session_state.last_user_prompt and not st.session_state.current_sql_query:
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            # Generate SQL query
            sql_query = generate_query()
            # Generate explanation for the SQL query
            explanation = explanation_chain.predict(sql_query=sql_query)

            # Save the current SQL query and explanation
            st.session_state.current_sql_query = sql_query
            st.session_state.current_explanation = explanation

            # Save the assistant's message without the data (initially)
            new_ai_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "content": "### Explanation:"},
                    {"type": "text", "content": explanation},
                    {"type": "text", "content": "### SQL Query:"},
                    {"type": "code", "language": "sql", "content": sql_query}
                ]
            }
            st.session_state.messages.append(new_ai_message)
            # Display the assistant's message
            for content_block in new_ai_message["content"]:
                if content_block["type"] == "text":
                    st.write(content_block["content"])
                elif content_block["type"] == "code":
                    st.code(content_block["content"], language=content_block.get("language", ""))
            st.write("### Run SQL Query:")
            st.button("Run Query", on_click=run_query)
    st.session_state.last_user_prompt = ''  # Reset after processing


