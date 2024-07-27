import os

import chainlit as cl
import logging
from chain import SQLDataChat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_sql_config_string(driver, server, db_name, username, password) -> str:
    return (f"DRIVER={driver};"
            f"SERVER={server};"
            f"DATABASE={db_name};"
            f"Trusted_Connection=yes;"
            f"UID={username};"
            f"PWD={password}")


@cl.on_chat_start
async def start():
    await cl.Message("Welcome to the SQL Chatbot. Please wait while connecting with the database.").send()

    db_driver = os.environ.get("DRIVER", None)
    db_server = os.environ.get("SERVER", None)
    db_username = os.environ.get("UID", None)
    db_password = os.environ.get("PASSWORD", None)
    db_name = os.environ.get("DATABASE", None)
    db_uri = get_sql_config_string(db_driver, db_server, db_name, db_username, db_password)
    print(f"Database URI: {db_uri}")


    try:
        file_name = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # "codellama-7b-instruct.Q5_K_M.gguf"
        repo = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"  # "TheBloke/CodeLlama-7B-Instruct-GGUF"  #

        sql_data_chat = SQLDataChat(repo, file_name, db_uri)
        cl.user_session.set("llm", sql_data_chat)
        await cl.Message("Database connection established. You can now ask your questions.").send()

    except Exception as e:
        await cl.Message(f"Failed to connect to the database: {e}").send()
        return


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("llm")
    if chain:
        thinking_message = await cl.Message(content="Thinking...").send()
        try:
            response = await chain.generate_response(message.content)
            logger.info(f"Final response: {response}")
            await thinking_message.update(content=response)

        except Exception as e:
            await thinking_message.update(content=f"Error processing your request: {e}")
    else:
        await cl.Message("Database connection not established. Please use /start to begin.").send()


