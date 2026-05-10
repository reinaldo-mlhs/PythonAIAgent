# imports
from langchain_openai import ChatOpenAI

# Set Chroma Vector Store as the Retriever
retriever = vector_store.as_retriever()

# Initialize the LLM instance
llm = ChatOpenAI(model="gpt-4o-mini")

# Create Document Parsing Function to String
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# imports
from langchain_core.prompts import PromptTemplate

# Create the Prompt Template
prompt_template = """Use the context provided to answer 
the user's question below. If you do not know the answer 
based on the context provided, tell the user that you do 
not know the answer to their question based on the context
provided and that you are sorry.

context: {context}

question: {query}

answer: """

# Create Prompt Instance from template
custom_rag_prompt = PromptTemplate.from_template(prompt_template)


# imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Create the RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# Query the RAG Chain
rag_chain.invoke(
  "According to the 2024 state of the union address, Who invaded Ukraine?"
)

# Get an I don't know from the Model
rag_chain.invoke("What is the purpose of life?")