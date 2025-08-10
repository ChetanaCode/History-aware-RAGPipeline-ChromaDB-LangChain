# Here's a complete example_rag_with_chroma.py script demonstrating how to set up a history-aware RAG pipeline 
# using ChromaDB as the vector store with LangChain. This includes document ingestion, embedding, history-aware retrieval 
# using create_history_aware_retriever, and conversational querying:
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

# 1. Setup environment
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Please set OPENAI_API_KEY in your .env"

# 2. Initialize LLM and Embeddings
llm = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Document ingestion and splitting (example URLs or paths)
# Assuming `load_your_documents()` returns a list of LangChain Document objects
# docs = load_your_documents()
# For demo, we'll skip loading actual docs and assume `docs` is defined
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)

from langchain_text_splitters import RecursiveCharacterTextSplitter
loader: PyPDFLoader = PyPDFLoader("EPGPMachineLearningAIBrochure__1688114020619.pdf")
docs: List[Document] = loader.load()
text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits: List[Document] = text_splitter.split_documents(docs)

# Placeholder: replace with actual splits
###splits = []  # Replace with real Document chunks

# 4. Create Chroma vector store and retriever
vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 5. Define history-aware retriever prompt
rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and user question, reformulate into a self-contained query."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=rephrase_prompt
)

# 6. Build the answer-generation chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Based on the provided context, answer the question concisely."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

# 7. Combine into a RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# 8. Example conversation loop
chat_history = []
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ("exit", "quit"):
        break

    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    answer = response.get("answer")
    print(f"\nAssistant: {answer}")

    chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=answer)
    ])
