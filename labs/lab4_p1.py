from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.vectorstores import SQLiteVec
from langchain_community.vectorstores import SQLiteVSS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain.ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

DB_PATH = "shrek_vec.db"

# Load and split text
loader = ...("shrek.txt")
documents = loader.load()
splitter = ...(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Initialize SQLiteVSS (stores both text & vectors)
embedding_model = ...(model="mistral")
vector_store = ....from_documents(..., embedding=..., db_path=...)

# Create retriever and LLM chain
retriever = ....as_retriever()
llm = Ollama(model="mistral")
qa_chain = ....from_chain_type(llm, retriever=retriever)

# Chat function
def chat():
    print("Shrek Chatbot: Ask me anything about Shrek!")
    while True:
        # get text from terminal
        query = ....("You: ")
        # implement breakint for "exit" or "bye"
        if query.lower() in [...]:
            print("Shrek Chatbot: Goodbye!")
            break
        response = ....run(...)
        print(f"Shrek Chatbot: {response}")

if __name__ == "__main__":
    chat()

# pip install -U langchain-community sqlite-vec
