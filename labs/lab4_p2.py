# # Lab 4 Part 2: Prompt Engineering for RAG with LangChain
#
# In this lab, we'll explore how to improve our RAG system through prompt engineering.
# We'll use the same stack as Part 1 but focus on creating better prompts that lead to
# higher quality responses.

# Import the necessary libraries
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SQLiteVSS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain

# Setup configuration
DB_PATH = "shrek_vec.db"
MODEL_NAME = "mistral"

# ## 1. Basic Setup (from Part 1)
#
# First, let's recreate our basic RAG setup from Part 1.

# Load and split text
loader = TextLoader("shrek.txt")
documents = loader.load()
splitter = ...(chunk_size=500, chunk_overlap=50)
docs = splitter....(documents)

# +
# Initialize embedding model and vector store
embedding_model = ....(model=MODEL_NAME)

# Check if the database exists and load it, otherwise create a new one
import os
if os.path.exists(DB_PATH):
    vector_store = ...(embedding=..., db_path=...)
    print(f"Loaded existing vector store from {DB_PATH}")
else:
    vector_store = ....from_documents(docs, embedding=..., db_path=...)
    print(f"Created new vector store at {DB_PATH}")

# +
# Set up some test questions
test_questions = [
    "What does Shrek think about onions?",
    "How does Donkey meet the Dragon?",
    "What happens at the wedding in the movie?",
    "Who is Lord Farquaad?"
]

# Function to test a prompt strategy on multiple questions
def test_prompt_strategy(qa_chain, questions, strategy_name):
    print(f"\n{'-'*20} Testing {strategy_name} {'-'*20}")
    results = {}
    
    for question in questions:
        print(f"\nQuestion: {question}")
        response = qa_chain.run(question)
        print(f"Answer: {response}")
        results[question] = response
    
    return results

# -

# ## 2. Understanding Default Prompts
#
# Let's look at the default prompt that LangChain uses for RAG:

# Initialize the default QA chain
llm = ...(model=MODEL_NAME)
retriever = ....as_retriever()
default_qa_chain = ....from_chain_type(..., retriever=...)

# See [prompts in langchain](https://github.com/langchain-ai/langchain/blob/ecff055096bc137bc10d7933d71016e2af56c06d/libs/langchain/langchain/chains/qa_generation/prompt.py#L47)

default_results = test_prompt_strategy(default_qa_chain, test_questions, "Default Prompt")

# +
print(default_results)


# +
# ## 3. Creating a Custom Prompt Template
#
# Let's create a custom prompt with instructions for better contextualization:

# +
# Define a custom prompt template
custom_qa_template = """You are a helpful AI assistant who answers questions about the movie Shrek. 
You will be given some context information from the Shrek script, and a question to answer.
Use the provided context to formulate a comprehensive, accurate answer.

If the answer isn't contained in the context, say "I don't have information about that in the Shrek script." 
Do not make up information that isn't supported by the context.

Context:
-----------
{context}
-----------

Question: {question}

Helpful Answer:"""

custom_prompt = ...(
    template=...,
    input_variables=["...", "..."]
)
# -

# Create a QA chain with our custom prompt
custom_qa_chain = ....from_chain_type(
    llm=...,
    chain_type="...",  #  all documents into the prompt
    retriever=...,
    chain_type_kwargs={"prompt": ...}
)

custom_results = test_prompt_strategy(custom_qa_chain, test_questions, "Custom Prompt")

# +
print(custom_results)

# +
# ## 4. Few-Shot Learning in Prompts
#
# We can improve the quality of responses by providing examples:

# +
# Create a few-shot prompt template
few_shot_qa_template = """You are an expert on the movie Shrek who answers questions in a helpful, informative way.
You will be given some context information from the Shrek script, and a question to answer.
Use the provided context to formulate your answer, staying faithful to the script.

Here are some examples of good answers:

Context: "SHREK: For your information, there's a lot more to ogres than people think. DONKEY: Example? SHREK: Example? Okay, um, ogres are like onions. DONKEY: They stink? SHREK: Yes... No! DONKEY: They make you cry? SHREK: No! DONKEY: You leave them in the sun, they get all brown, start sprouting little white hairs. SHREK: No! Layers! Onions have layers. Ogres have layers! Onions have layers. You get it? We both have layers."
Question: What does Shrek compare ogres to?
Answer: Shrek compares ogres to onions because both have layers. When Donkey asks for an example of how there's more to ogres than people think, Shrek explains that "ogres are like onions" because of their layers, despite Donkey initially misunderstanding the comparison by suggesting onions stink or make people cry.

Context: "FARQUAAD: That champion shall have the honor-- no, no-- the privilege to go forth and rescue the lovely Princess Fiona from the fiery keep of the dragon. If for any reason the winner is unsuccessful, the first runner-up will take his place and so on and so forth."
Question: What is the prize for the tournament winner?
Answer: The prize for the tournament winner is the privilege to rescue Princess Fiona from the dragon's keep. Lord Farquaad announces that the champion will have "the honor-- no, no-- the privilege" of undertaking this dangerous mission to save the princess.

Now, answer the following question using the provided context:

Context:
-----------
{context}
-----------

Question: {question}

Answer:"""

few_shot_prompt = ...(
    template=...,
    input_variables=["...","..."]
)
# -

# Create a QA chain with our few-shot prompt
few_shot_qa_chain = ....from_chain_type(
    llm=...,
    chain_type="stuff",
    retriever=...,
    chain_type_kwargs={"prompt": few_shot_prompt}
)


few_shot_results = test_prompt_strategy(few_shot_qa_chain, test_questions, "Few-Shot Prompt")

# +
print(few_shot_results)

# +
# ## 5. Role-Based Prompting
#
# Another strategy is to give the model a specific role to play:

# +
# Create a role-based prompt template
role_based_qa_template = """You are Shrek himself, the lovable ogre with a gruff exterior but heart of gold.
Answer the following question about your movie in your own voice and character.
Use the provided context to ensure your answers are factually correct, but respond as Shrek would.

Context:
-----------
{context}
-----------

Question: {question}

Answer (as Shrek):"""

role_prompt = PromptTemplate(
    template=...,
    input_variables=["...", "..."]
)
# -

# Create a QA chain with our role-based prompt
role_qa_chain = ....from_chain_type(
    llm=...,
    chain_type="stuff",
    retriever=...,
    chain_type_kwargs={"prompt": ...}
)

role_results = test_prompt_strategy(role_qa_chain, test_questions, "Role-Based Prompt")

# +
print(few_shot_results)

# +
# ## 6. Comparing Different Prompting Strategies
#
# Let's test our different prompting strategies on some example questions:



# Test each prompt strategy
default_results = test_prompt_strategy(default_qa_chain, test_questions, "Default Prompt")
custom_results = test_prompt_strategy(custom_qa_chain, test_questions, "Custom Prompt")
few_shot_results = test_prompt_strategy(few_shot_qa_chain, test_questions, "Few-Shot Prompt")
role_results = test_prompt_strategy(role_qa_chain, test_questions, "Role-Based Prompt")

# ## 7. Interactive Chat with Prompt Selection
#
# Let's create an interactive chat interface that allows selecting different prompt strategies:

def chat_with_prompt_selection():
    qa_chains = {
        "1": {"name": "Default Prompt", "chain": default_qa_chain},
        "2": {"name": "Custom Prompt", "chain": custom_qa_chain},
        "3": {"name": "Few-Shot Prompt", "chain": few_shot_qa_chain},
        "4": {"name": "Role-Based Prompt (as Shrek)", "chain": role_qa_chain}
    }
    
    print("Shrek RAG Chatbot with Prompt Selection")
    print("======================================")
    print("Choose a prompt strategy:")
    for key, value in qa_chains.items():
        print(f"{key}. {value['name']}")
    
    choice = input("Enter your choice (1-4): ")
    if choice not in qa_chains:
        print("Invalid choice, using default prompt")
        choice = "1"
    
    selected_chain = qa_chains[choice]["chain"]
    print(f"\nUsing {qa_chains[choice]['name']}")
    print("Ask questions about Shrek. Type 'exit', 'quit', or 'bye' to end the chat.\n")
    
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Shrek Chatbot: Goodbye!")
            break
        
        response = selected_chain.run(query)
        print(f"Shrek Chatbot: {response}")
        print()

# Run the interactive chat if this script is executed directly
if __name__ == "__main__":
    chat_with_prompt_selection()


