import os
import urllib
from pathlib import Path

import requests
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import LlamaCpp, Replicate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sqlalchemy import create_engine

from tqdm import tqdm


class InputType(BaseModel):
    question: str


class SQLDataChat:

    def __init__(self, repo_name: str, filename: str, db_conn_string: str, model_type: str = 'llama_cpp'):
        """
        Initialize SQLDataChat with the given repository of hugging face for the specific model based on the
        system settings like cpu and gpu devices
        :param repo_name: use 'TheBloke/Mistral-7B-Instruct-v0.1-GGUF' for quantized llm
        :param filename: use 'mistral-7b-instruct-v0.1.Q4_K_M.gguf' for better results
        :param db_conn_string: db_connection string for the database connection
        :param model_type: 'llama_cpp' for 'cpu' and 'ollama' or 'llama2' for 'gpu' (cuda) enabled devices
        """
        self.MSSQL_AGENT_PREFIX = """

        You are an agent designed to interact with a SQL database.
        ## Instructions:
        - Given an input question, create a syntactically correct {dialect} query
        to run, then look at the results of the query and return the answer.
        - Unless the user specifies a specific number of examples they wish to
        obtain, **ALWAYS** limit your query to at most {top_k} results.
        - You can order the results by a relevant column to return the most
        interesting examples in the database.
        - Never query for all the columns from a specific table, only ask for
        the relevant columns given the question.
        - You have access to tools for interacting with the database.
        - You MUST double check your query before executing it.If you get an error
        while executing a query,rewrite the query and try again.
        - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
        to the database.
        - DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
        OF THE CALCULATIONS YOU HAVE DONE.
        - Your response should be in Markdown. However, **when running  a SQL Query
        in "Action Input", do not include the markdown backticks**.
        Those are only for formatting the response, not for executing the command.
        - ALWAYS, as part of your final answer, explain how you got to the answer
        on a section that starts with: "Explanation:". Include the SQL query as
        part of the explanation section.
        - If the question does not seem related to the database, just return
        "I don\'t know" as the answer.
        - Only use the below tools. Only use the information returned by the
        below tools to construct your query and final answer.
        - Do not make up table names, only use the tables returned by any of the
        tools below.

        ## Tools:

        """

        template = """Based on the table schema below, write a SQL query that would answer the user's question:
                {schema}

                Question: {question}
                SQL Query:"""

        self.replicate_id = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"

        self.repo = repo_name
        self.filename = filename
        self._download_model()
        self.llm = None
        self.model_type = model_type
        if model_type == 'llama_cpp':
            self.llm = self._initialize_llama_cpp()
        elif model_type == 'ollama':
            self.llm = self._initialize_ollama()
        elif model_type == 'llama2':
            self.llm = self._initialize_llama2()
        else:
            raise ValueError("Unknown model type: model type can be either 'llama_cpp' or 'llama2' or 'ollama'")

        # self.db = self._initialize_db(db_name)  # sqlite3
        self.db = self._initialize_database(db_conn_string)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"Given an input question, "
                           f"convert it to a SQL query. No pre-amble."),
                MessagesPlaceholder(variable_name="history"),
                ("human", template),
            ]
        )

        self.memory = ConversationBufferMemory(return_messages=True)

        sql_chain = self.get_sql_chain()

        sql_response_memory = RunnablePassthrough.assign(output=sql_chain) | self._save

        # Chain to answer
        template_response = """
        Based on the table schema below, question, sql query, and sql response, write a natural language response:
        {schema}

        Question: {question}
        SQL Query: {query}
        SQL Response: {response}"""  # noqa: E501
        prompt_response = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given an input question and SQL response, convert it to a natural "
                    "language answer. No pre-amble.",
                ),
                ("human", template_response),
            ]
        )

        self.chain = (
                RunnablePassthrough.assign(query=sql_response_memory).with_types(
                    input_type=InputType
                )
                | RunnablePassthrough.assign(schema=self._get_schema,
                                             response=lambda x: self.db.run(x["query"]))
                | prompt_response
                | self.llm
        )

    def _download_model(self):
        if not os.path.exists(self.filename):
            print(f"'{self.filename}' not found. Downloading...")
            # Download the file
            with requests.get(f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}",
                              stream=True, allow_redirects=True) as resp:
                resp.raise_for_status()
                total_size = int(resp.headers.get('content-length', 0))
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=self.filename, ascii=True) as pbar:
                    with open(self.filename, "wb") as file:
                        for chunk in resp.iter_content(chunk_size=1024):
                            # Filter out keep-alive new chunks
                            if chunk:
                                # Write the chunk to the file
                                file.write(chunk)
                                # Update the progress bar
                                pbar.update(len(chunk))
            print(f"'{self.filename}' has been downloaded.")
        else:
            print(f"'{self.filename}' already exists in the current directory.")

    def _initialize_llama_cpp(self):
        return LlamaCpp(
            model_path=self.filename,
            n_batch=512,
            n_ctx=2048,
            n_gpu_layers=1,
            # f16_kv MUST set to True
            # otherwise you will run into problem after a couple of calls
            f16_kv=True,
            verbose=True,
        )

    def _initialize_ollama(self, ollama_llm='zephyr'):
        return ChatOllama(model=ollama_llm)

    def _initialize_llama2(self):
        return Replicate(
            model=self.replicate_id,
            model_kwargs={"temperature": 0.01, "max_length": 500, "top_p": 1},
        )

    def _initialize_db(self, sqlite3_db_name: str):
        # TODO: This needs to be implemented logically
        db_path = Path(__file__).parent / sqlite3_db_name
        rel = db_path.relative_to(Path.cwd())
        db_string = f"sqlite:///{rel}"
        db = SQLDatabase.from_uri(db_string, sample_rows_in_table_info=0)
        return db

    def _initialize_database(self, connection_str: str):
        try:
            connection_str = urllib.parse.quote_plus(connection_str)
            connection_str = f"mssql+pyodbc:///?odbc_connect={connection_str}"
            engine = create_engine(connection_str)
            schema = os.environ.get("SCHEMA", None)
            db = SQLDatabase(engine, view_support=True, schema=schema)  # view_support to get all the tables
            print(f"Database connected. Dialect: {db.dialect}")
            print(f"Usable tables: {db.get_usable_table_names()}")
            return db
        except Exception as e:
            raise ValueError(f"Could not connect to database {e}")
        except ConnectionError as e:
            raise ValueError(f"Error initializing database: {e}")

    def _get_schema(self, _):
        return self.db.get_table_info()

    def _run_query(self, query):
        return self.db.run(query)

    def _save(self, input_output):
        output = {"output": input_output.pop("output")}
        self.memory.save_context(input_output, output)
        return output["output"]

    async def generate_response(self, query):
        return self.chain.invoke({"question": query})

    def get_sql_chain(self):
        return (RunnablePassthrough.assign(schema=self._get_schema)
                | self.prompt
                | self.llm.bind(stop=["\nSQLResult:"])
                | StrOutputParser()) \
            if self.model_type == 'llama2' else \
            (RunnablePassthrough.assign(
                schema=self._get_schema,
                history=RunnableLambda(lambda x: self.memory.load_memory_variables(x)["history"]),
            )
             | self.prompt
             | self.llm.bind(stop=["\nSQLResult:"])
             | StrOutputParser()
             )
