import os
import datetime
import random
# import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate

from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory, SimpleMemory
from langchain.chains.sequential import SimpleSequentialChain
from langchain_experimental.chat_models import Llama2Chat
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage

# Initialize environment and set up for Windows asyncio compatibility
load_dotenv()

prompt_template_questions = """
You are an customer in a pharmacy. You will perform specific tasks: greeting the pharmacist, inquiring 
about medications, understanding prescription requirements, expressing gratitude, and concluding the 
interaction. You does not possess deep knowledge about medications or medical conditions and 
should avoid discussing unrelated or random topics. All responses and queries from you should be 
concise, limited to 1 or 2 sentences. Can you play a role as a customer at Pharmacy. 
And I am a pharmacist. Can you ask me some questions as a customer?  You ask me questions one by one. So we will talk as real conversations?

------------
{text}
------------

Create questions that will you have as an customer in a pharmacy. Make sure not to lose any important information.
Can you play a role as a customer at Pharmacy. And I am a pharmacist. Can you ask me some questions as a customer?  
You ask me questions one by one. So we will talk as real conversations?

QUESTIONS:
"""

PROMPT_QUESTIONS = PromptTemplate(
    template=prompt_template_questions, input_variables=["text"]
)

refine_template_questions = """
You are an customer in a pharmacy. You goal is performing specific tasks: greeting the pharmacist, inquiring 
about medications, understanding prescription requirements, expressing gratitude, and concluding the 
interaction. 
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below. Can you play a role as a customer at Pharmacy. And I am a pharmacist. Can you ask me some questions as a customer?  
You ask me questions one by one. So we will talk as real conversations?
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.

QUESTIONS:
"""

REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)

# Load documents
pdf_path = os.path.join("data", "Basicsofpharmacy.pdf")
loader = PyPDFLoader(pdf_path)
data = loader.load()
# reader = SimpleDirectoryReader("./data")
# documents = reader.load_data()
print("documents loaded successfully.")

# Combine text from Document into one string for question generation
text_question_gen = ""
for page in data:
    text_question_gen += page.page_content

# Initialize Text Splitter for question generation
text_splitter_question_gen = RecursiveCharacterTextSplitter(
    chunk_size=10000, chunk_overlap=50
)

# Split text into chunks for question generation
text_chunks_question_gen = text_splitter_question_gen.split_text(text_question_gen)

# Convert chunks into Documents for question generation
docs_question_gen = [Document(page_content=t) for t in text_chunks_question_gen]

# Initialize Large Language Model for question generation
llm_question_gen = Replicate(
    model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    is_chat_model=True,
    input={"temperature": 0.01, "max_length": 500, "top_p": 1},
)

# Initialize question generation chain
question_gen_chain = load_summarize_chain(
    llm=llm_question_gen,
    chain_type="refine",
    verbose=True,
    question_prompt=PROMPT_QUESTIONS,
    refine_prompt=REFINE_PROMPT_QUESTIONS,
)
print("question_gen_chain.get_prompts:", question_gen_chain.get_prompts())
# Run question generation chain
questions = question_gen_chain.run(docs_question_gen)

# Initialize Large Language Model for answer generation
llm_answer_gen = Replicate(
    model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    model_kwargs={"temperature": 0.01, "max_length": 500, "top_p": 1},
)

# Create vector database for answer generation
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)
print("HuggingFaceEmbeddings loaded successfully.")

# Initialize vector store for answer generation
vector_store = Chroma.from_documents(docs_question_gen, embeddings)
print("Vector store loaded successfully.")

# Initialize retrieval chain for answer generation
answer_gen_chain = RetrievalQA.from_chain_type(
    llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever(k=2)
)
print("answer_gen_chain.get_prompts:", answer_gen_chain.get_prompts())

# create chat prompt template
template_messages = [
    # SystemMessage(content="You are a helpful assistant."),
    # PromptTemplate(
    # input_variables=["text"],
    # template="Hi, I'm the pharmacist, how can I help you?",
    # ),
    MessagesPlaceholder(variable_name="chat_history"),
    # HumanMessagePromptTemplate.from_template("Hi, I'm the pharmacist, how can I help you?" ),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)
llm_chat_gen = llm_question_gen
llm_chat_model = Llama2Chat(llm=llm_chat_gen)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=llm_chat_model, prompt=prompt_template, memory=memory)


# Split generated questions into a list of questions
question_list = questions.split("\n")

# Answer each question and save to a file
count = 0
# while human_answer := input("Enter a answer (q to quit): ") != "q":
#     num = random.randint(0, len(question_list) - 1)
#     question = question_list[num]
#     print("AI Question: ", question)
    
#     # Wait for user input before asking the next question
#     # human_answer = input("Human to answer and press Enter to continue: ")
#     result = chain.run(human_answer)
#     print("AI response ", result)
    
#     answer = answer_gen_chain.run(question)
#     print("AI Answer: ", answer)
#     print("--------------------------------------------------\n\n")
#     count += 1
#     if count == 5:
#         break

# print("Thank you and have a nice day!")

simplechain = SimpleSequentialChain(chains=[question_gen_chain, chain, answer_gen_chain], verbose=True, memory=memory)
output = simplechain.run(docs_question_gen)
print(output)

