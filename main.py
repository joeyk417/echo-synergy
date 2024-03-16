# import asyncio
import os
import tempfile
import datetime

# import streamlit as st
from dotenv import load_dotenv

# from transformers import AutoTokenizer
# from llama_index.core import Settings, ServiceContext, SimpleDirectoryReader
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.replicate import Replicate
# # from llama_index.core.evaluation import DatasetGenerator
# from llama_index.core.llama_dataset.generator import RagDatasetGenerator
# from llama_index.readers.file import PDFReader
import sys

# from prompts import system_prompt, query_wrapper_prompt

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate

# Initialize environment and set up for Windows asyncio compatibility
load_dotenv()

# prompt_template_questions = """
# You are an expert in creating practice questions based on study material.
# Your goal is to prepare a student for their exam. You do this by asking questions about the text below:

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
# Run question generation chain
questions = question_gen_chain.run(docs_question_gen)

# Initialize Large Language Model for answer generation
llm_answer_gen = Replicate(
    model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    input={"temperature": 0.01, "max_length": 500, "top_p": 1},
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

# Split generated questions into a list of questions
question_list = questions.split("\n")

# Answer each question and save to a file
for question in question_list:
    print("Question: ", question)
    answer = answer_gen_chain.run(question)
    print("Answer: ", answer)
    print("--------------------------------------------------\n\n")

# while (prompt := input("Enter a prompt (q to quit): ")) != "q":
#     start_time = datetime.datetime.now()

#     result = canada_engine.query(prompt)

#     end_time = datetime.datetime.now()
#     execution_time = end_time - start_time

#     print(result)
#     print("Execution time:", execution_time.total_seconds())

    # Create a directory for storing answers
# answers_dir = os.path.join(tempfile.gettempdir(), "answers")
# os.makedirs(answers_dir, exist_ok=True)

# Create a single file to save questions and answers
# qa_file_path = os.path.join(answers_dir, "questions_and_answers.txt")
# with open(qa_file_path, "w") as qa_file:
#         # Answer each question and save to the file
#         for idx, question in enumerate(question_list):
#             answer = answer_gen_chain.run(question)
#             qa_file.write(f"Question {idx + 1}: {question}\n")
#             qa_file.write(f"Answer {idx + 1}: {answer}\n")
#             qa_file.write("--------------------------------------------------\n\n")

#########################################################################
# async def generate_dataset(documents, service_context, num_questions_per_chunk):
#     data_generator = RagDatasetGenerator.from_documents(
#         documents=documents,
#         # service_context=service_context,
#         llm=Settings.llm,
#         num_questions_per_chunk=num_questions_per_chunk,
#         show_progress=True
#     )
#     print("data generated successfully.")
#     # questions = await data_generator.generate_dataset_from_nodes()
#     questions = await data_generator.generate_questions_from_nodes()
#     print("question generated successfully.")
#     return questions

# async def main():
#     print("Initializing settings...")

#     # set the LLM
#     # system_prompt = "Your system prompt here"  # Replace "Your system prompt here" with the desired value

#     # Settings.llm = Groq(model="llama2-70b-4096")

#     llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
#     Settings.llm = Replicate(
#     model=llama2_7b_chat,
#     temperature=0.1,
#     system_prompt=system_prompt,
#     query_wrapper_prompt=query_wrapper_prompt,
#     additional_kwargs={"top_p": 1, "max_new_tokens": 300},
#     is_chat_model=True,)

#     Settings.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
#     Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#     service_context = ServiceContext.from_defaults(llm=Settings.llm, embed_model=Settings.embed_model)
#     print("Settings initialized successfully.")

#     # Load documents
#     pdf_path = os.path.join("data", "Good_Customer_Service_Guidance.pdf")
#     documents = PDFReader().load_data(file=pdf_path)
#     # reader = SimpleDirectoryReader("./data")
#     # documents = reader.load_data()
#     print("documents loaded successfully.")

#     # Adjust the chunk size as a strategy to manage API rate limits
#     num_questions_per_chunk = 2  # Consider lowering this if you're hitting rate limits

#     # Generate questions with retry logic
#     try:
#         eval_questions  = generate_dataset, documents, service_context, num_questions_per_chunk
#         print("Questions generated successfully:")

#         for question in eval_questions:
#             print(question)

#         # Split generated questions into a list of questions
#         # question_list = eval_questions .split("\n")

#     except Exception as e:  # Consider catching specific exceptions
#         print(f"Failed to generate questions due to an error: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())
