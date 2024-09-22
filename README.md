# ChatDB
# Project Requirements  
1. **Interactive Query Tool** - ChatDB will function like a ChatGPT-style assistant, helping users interactively query SQL and NoSQL databases.
2. **Dynamic Query Suggestion** - It suggests example queries covering various constructs like group by, join, aggregation, and more, dynamically generating them instead of relying on hardcoded queries.
3. **Natural Language Understanding** - ChatDB interprets user queries in natural language, recognizing patterns and mapping them to corresponding database functions.
4. **Query Execution** - Unlike ChatGPT, it can execute database queries directly and display the results for the user.
5. **Database Selection** - Users can choose a database, view its tables, attributes, and sample data before generating and executing queries.
6. **Variety in Suggestion** - Each time users ask for sample queries, ChatDB provides a different set, offering diverse examples to explore multiple query constructs.

# Planned Implementation  
1. **Frontend (Streamlit)** - The user interface will be built using Streamlit, allowing users to interact with ChatDB and input natural language queries.
2. **Query Explanation (Groq LLM)** - After receiving a query, it will be passed to an open-source LLM from Groq, which will generate explanations for the query that can be displayed to the user.
3. **Text-to-SQL Generation (AWS Bedrock with Claude)** - AWS Bedrock will be used to access Claude v2 for converting the natural language queries into SQL queries.
4. **Source (Text-to-SQL)** - [AWS Text-to-SQL Solution](https://aws.amazon.com/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/)
5. **Database (PostgreSQL and MongoDB)** - The generated queries will be tested against both SQL (PostgreSQL) and NoSQL (MongoDB) databases to ensure compatibility and correctness.
6. **Lambda Function for Query Execution** - A lambda function will be called to execute the generated queries on the databases and verify if they work as expected.
7. **Query Validation** - If the generated queries don’t execute correctly, the model will regenerate the queries until valid ones are produced, then display them to the user.
