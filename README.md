**Overview**
-----------

The SQL Chatbot is a conversational AI application that allows users to interact with a database using natural
language queries. The chatbot uses an LLaMA model to generate SQL queries and execute them against a database.

**Features**
------------

* User-friendly interface for setting up a database connection
* Support for various database drivers (e.g., MySQL, PostgreSQL)
* Ability to ask questions and receive answers in the form of SQL queries
* Execution of SQL queries against a connected database

**Technical Requirements**
-------------------------

* Python version >= 3.8 and < 3.11
* LLaMA model (available through the `langchain_community.llms` package)
* `SQLDatabase` class from the `langchain_community.utilities` package
* `create_sql_query_chain` function from the `langchain.chains.sql_database.query` module

**Setup and Usage**
-------------------

1. Change directory `cd datachat`
2. Install required packages using pip: `pip install -r requirements.txt`.
3. Run Llama3 locally using command `ollama run llama3` in different command line.
4. Run the application using command `chainlit run app.py` command in the chatbot.
5. Provide the necessary database details (driver, server, username, password, and database name).
6. Use natural language to ask questions about your database, such as; 
   1. "What are the top 10 orders for this
   quarter?" or 
   2. "Show me all customers from a specific region." or
   3. "How many employees are there in the database?"
7. The chatbot will execute the generated SQL query against the connected database and return the results.

**Troubleshooting**
-------------------

* Check the logs for any errors or exceptions that may have occurred during the setup or execution of the
application.
* Verify that the database connection is established successfully before attempting to execute SQL queries.
* If you encounter issues with the LLaMA model, try updating the package or using a different model.
