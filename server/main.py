from fastapi import FastAPI
import logging
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant.qdrant import QdrantVectorStore
import httpx
import pandas as pd
import tiktoken

def loader(titles: list[str], texts: list[str]) -> list[Document]:
    """Загружает 2 массива одинаковой длины и создает список документов.

    Args:
        titles (list[str]): Список заголовков
        texts (list[str]): Список текстов

    Returns:
        list[Document]: Список объектов Document
    """
    result: list[Document] = []
    for index, title in enumerate(titles):
        result.append(Document( page_content = title, metadata = { "source": texts[index] }))
    return result

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Возвращает количество токенов в текстовой строке."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def docs_num_tokens(docs: list[Document]) -> int:
    """
    Функция вычисляет общее количество токенов в списке документов.

    Args:
        docs (list[Document]): Список объектов Document, для которых необходимо подсчитать токены.

    Returns:
        int: Общее количество токенов во всех документах.
    """
    return sum([num_tokens_from_string(doc.page_content) for doc in docs])

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_PROXY = os.environ.get('OPENAI_PROXY')
QDRANT_URL = os.environ.get('QDRANT_URL')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
QDRANT_collection_name = os.environ.get('QDRANT_collection_name')

df = pd.read_csv("papers.csv")
docs = loader(df['Title'].values, df['Text'].values)
prompt_tokens = docs_num_tokens(docs)
print(f"Количество токенов docs = {prompt_tokens}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions = 256, api_key=OPENAI_API_KEY)
qdrant = QdrantVectorStore.from_documents(docs,
                                          embeddings,
                                          collection_name=QDRANT_collection_name,
                                          location=":memory:",
                                          force_recreate=True)

client = OpenAI(api_key=OPENAI_API_KEY,
                http_client=httpx.Client(proxy=OPENAI_PROXY,
                                          transport=httpx.HTTPTransport(local_address="0.0.0.0")),
                )

def LLM_request(query: str, context: str) -> dict:
    """This function processes a user query using a GPT-4o model.

    Args:
        query (str): The user's question.
        context (str): The relevant document information.

    Returns:
        dict: A dictionary containing the response content, prompt tokens, completion tokens, total tokens, and elapsed time.
    """
    
    system_prompt = f"""DOCUMENT:
    {context}

    QUESTION:
    {query}

    INSTRUCTIONS:
    Answer the users QUESTION using the DOCUMENT text above.
    Keep your answer ground in the facts of the DOCUMENT.
    If the DOCUMENT doesn’t contain the facts to answer the QUESTION return Нет данных.
    """

    start_time = time.time()   # record the time before the request is sent

    LLM = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}],
            model="gpt-4o",
            temperature=0.2,
    )

    elapsed_time = time.time() - start_time    # calculate the time it took to receive the response

    return {
            "content": LLM.choices[0].message.content,
            "prompt_tokens": LLM.usage.prompt_tokens,
            "completion_tokens": LLM.usage.completion_tokens,
            "total_tokens": LLM.usage.total_tokens,
            "elapsed_time": elapsed_time
    }

def docs_related(query: str, top_k: int = 4):
    """
    Finds the most similar documents to a given query using a vector similarity search.

    Args:
        query (str): The search query to find related documents.
        top_k (int, optional): The number of top similar documents to retrieve. Defaults to 4.

    Returns:
        list[Document]: A list of documents that are most similar to the query.
    """
    return qdrant.similarity_search(query, k = top_k)

def keys_related(docs: list[Document]) -> str:
    """
    Returns a string containing the page_content of all documents in the list, separated by newline characters.

    Args:
        docs (list[Document]): The list of documents to extract page_content from.

    Returns:
        str: A string containing the page_content of all documents, separated by newline characters.
    """
    return "\n".join([doc.page_content for doc in docs])

def values_related(docs: list[Document]) -> str:
    """
    Returns a string containing the 'source' metadata of all documents in the list, 
    separated by a specific delimiter.

    Args:
        docs (list[Document]): The list of documents to extract metadata from.

    Returns:
        str: A string containing the 'source' metadata of all documents, 
        separated by a delimiter of '\n\n---\n\n'.
    """
    return "\n\n---\n\n".join([doc.metadata.get("source") for doc in docs])


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI()


@app.post("/chat")
async def completions_create(query: str) -> str:
    """
    Handles a POST request to the /chat endpoint.

    This endpoint is the primary entrypoint for the chatbot. It takes a query string
    as input and returns a response string.

    The function first calls the docs_related function to find the most similar
    documents to the query. It then calls the values_related function to extract
    the 'source' metadata from the documents. Finally, it calls the LLM_request
    function to generate a response using the GPT-4o model.

    Args:
        query (str): The input query string

    Returns:
        str: The response string
    """
    logging.info("Creating completions.")
    Docs = docs_related(query)
    Values = values_related(Docs)
    return LLM_request(query, Values)["content"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="debug", reload=False)