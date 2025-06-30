import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate

## To load environment variables from .env file
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Use only the following context to answer the user's question:

{context}

Question: {question}
"""


def main():
    
    ## Creating CLI (Command Line Interface) for query input
    ## Parser is a component that analyzes a sequence of input
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    ## Prepare the DB
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=10)
    
    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1} - Score: {score:.3f}")
        print(doc.page_content[:300])  # Preview of matched content
    
    ## Eliminating if there were no results or the scores were lower than 0.5
    if len(results) == 0 or results[0][1] < 0.5:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    
    ## Using Gemini API
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    response = llm.invoke(prompt)

    ## Colecting the sources
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    
    ## Printing out the entire response
    formatted_response = f"Response: {response.content}\nSources: {sources}"
    print(formatted_response)
    
if __name__ == '__main__':
    main()