import os
import random
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain import hub
from langchain.chains import create_retrieval_chain

load_dotenv()

my_secret = os.environ['REPLICATE_API_TOKEN']
app = FastAPI()

# Placeholder for storing conversation history
conversation_history: List[str] = []

# Task status tracker
task_status: Dict[str, str] = {}

# In a real application, this could be a database or cache
question_storage: List[str] = []

class Question(BaseModel):
    text: str  # For receiving question text via POST request

# def generate_new_question(conversation_history):
#       # Check if there is any conversation history
#     # if not conversation_history:
#     #     return "Good morning, What are the opening hours of the pharmacy, and are there any alternative or after-hours services available?"

#     # Get the last question asked
#     # last_question = conversation_history[-1]
   
#     follow_up_question = question_gen_chain.run(docs_question_gen)
#     return follow_up_question

def generate_questions(conversation_history):
  # Placeholder for the actual logic to generate questions
  question_gen_chain.run(docs_question_gen)

def generate_new_questions(conversation_history):
  question_new_gen_chain.run(docs_question_gen)

class Message(BaseModel):
  content: str

prompt_template_questions = """
You are an customer in a pharmacy. You will perform specific tasks: greeting the pharmacist, inquiring 
about medications, understanding prescription requirements, expressing gratitude, and concluding the 
interaction. You does not possess deep knowledge about medications or medical conditions and 
should avoid discussing unrelated or random topics. All responses and queries from you should be 
concise, limited to 1 or 2 sentences. Can you play a role as a customer at Pharmacy. 
And I am a pharmacist. Can you ask me 10 potenial questions as a customer?  

------------
{text}
------------

Create questions that will you have as an customer in a pharmacy. Make sure not to lose any important information.
Can you play a role as a customer at Pharmacy. And I am a pharmacist. Can you ask me one short question as a customer?  
You will ask me 10 potenial questions. So we will talk as real conversations?

QUESTIONS:
"""

PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions,
                                  input_variables=["text"])

refine_template_questions = """
You are an customer in a pharmacy. You goal is performing specific tasks: greeting the pharmacist, inquiring 
about medications, understanding prescription requirements, expressing gratitude, and concluding the 
interaction. 
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below. Can you play a role as a customer at Pharmacy. And I am a pharmacist. Can you ask me some questions as a customer?  
You ask me 10 potenial questions. So we will talk as real conversations?
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

# # This controls how the standalone question is generated.
# # Should take `chat_history` and `question` as input variables.
# ConversationTemplate = (
#     # "Combine the chat history and follow up question into "
#     # "a standalone question. Chat History: {chat_history}"
#     # "Follow up question: {question}"
#     "Can you play a role as a customer at Pharmacy. And I am a pharmacist. Can you ask me 1 short question as a customer based on our chat history"
#     ------------
#     {text}
# ------------
# )

ConversationTemplate = """
"Can you play a role as a customer at Pharmacy. And I am a pharmacist. Can you ask me 1 short question as a customer based on our chat history"
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.

QUESTIONS:
"""
CONVERSATION_PROMPT_QUESTIONS = PromptTemplate(template=ConversationTemplate,
                                  input_variables=["text"])
# Load documents
pdf_path = os.path.join("data", "Basicsofpharmacy.pdf")
loader = PyPDFLoader(pdf_path)
data = loader.load()

## To do, load docs from .data folder
# reader = SimpleDirectoryReader("./data")
# documents = reader.load_data()
# print("documents loaded successfully.")

# Combine text from Document into one string for question generation
text_question_gen = ""
for page in data:
  text_question_gen += page.page_content

# Initialize Text Splitter for question generation
text_splitter_question_gen = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                            chunk_overlap=50)

# Split text into chunks for question generation
text_chunks_question_gen = text_splitter_question_gen.split_text(
    text_question_gen)

# Convert chunks into Documents for question generation
docs_question_gen = [
    Document(page_content=t) for t in text_chunks_question_gen
]

# Initialize Large Language Model for question generation
llm_question_gen = Replicate(
    model=
    "meta/llama-2-7b-chat:f1d50bb24186c52daae319ca8366e53debdaa9e0ae7ff976e918df752732ccc4",
    is_chat_model=True,
    input={
        "temperature": 0.01,
        "max_length": 500,
        "top_p": 1
    },
)

# Initialize question generation chain
question_gen_chain = load_summarize_chain(
    llm=llm_question_gen,
    chain_type="refine",
    verbose=True,
    question_prompt=PROMPT_QUESTIONS,
    refine_prompt=REFINE_PROMPT_QUESTIONS,
)

question_new_gen_chain = load_summarize_chain(
    llm=llm_question_gen,
    chain_type="refine",
    verbose=True,
    question_prompt=CONVERSATION_PROMPT_QUESTIONS,
    refine_prompt=REFINE_PROMPT_QUESTIONS,
)

# Initialize NEXT question generation chain
# new_question_gen_chain = load_summarize_chain(
#     llm=llm_question_gen,
#     chain_type="refine",
#     verbose=True,
#     question_prompt=CONVERSATION_PROMPT_QUESTIONS,
#     refine_prompt=REFINE_PROMPT_QUESTIONS,
# )
# Run question generation chain (save billing)
file_path = 'questions.txt'
 
# open the file in read mode
with open(file_path, 'r') as file_obj:
    # read first character
    first_char = file_obj.read(1)
 
    if not first_char:
        questions = question_gen_chain.run(docs_question_gen)
    else:
        questions = ""

# Initialize Large Language Model for answer generation
llm_answer_gen = Replicate(
    model=
    "meta/llama-2-7b-chat:f1d50bb24186c52daae319ca8366e53debdaa9e0ae7ff976e918df752732ccc4",
    input={
        "temperature": 0.01,
        "max_length": 500,
        "top_p": 1
    },
)

# Create vector database for answer generation
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"})
# print("HuggingFaceEmbeddings loaded successfully.")

# Initialize vector store for answer generation
vector_store = Chroma.from_documents(docs_question_gen, embeddings)
# print("Vector store loaded successfully.")

# Initialize retrieval chain for answer generation
answer_gen_chain = RetrievalQA.from_chain_type(
    llm=llm_answer_gen,
    chain_type="stuff",
    retriever=vector_store.as_retriever(k=2))

# Split generated questions into a list of questions
question_list = questions.split("\n")

# Answer each question and save to a file
# for question in question_list:
# print("List of Questions are : ", question_list)
#  print("Customer Question is : ", question)
# modalAnswer = answer_gen_chain.run(question_list)
#   print("Modal Answer is : ", modalAnswer)
f = open("questionsString.txt", "a")
f.writelines(question_list)
# f.writelines(modalAnswer)

file_path = 'questionsString.txt'

# Read the content of the file
with open(file_path, 'r') as file:
    content = file.read()

# Splitting the content based on a pattern (e.g., digit followed by a dot for numbered list)
import re
questions = re.split(r'\d+\.', content)[1:]  # Skip the first split as it's before the first question

# Add the question numbers back and format each question on a new line
formatted_questions = [f"{i+1}. {question.strip()}" for i, question in enumerate(questions) if question.strip()]

# Join the formatted questions with a newline character
formatted_content = "\n".join(formatted_questions)

# Optionally, save the formatted content to a new file
new_file_path = 'questions.txt'
with open(new_file_path, 'w') as new_file:
    new_file.write(formatted_content)

print("--------------------------------------------------\n\n")
print("-------------------Next Question------------------\n\n")

# Print the list of questions
# for question in questions_list:
# prompt = PromptTemplate.from_template(ConversationTemplate)
# print("-------------------1------------------\n\n")
# question_generator_chain = LLMChain(llm=llm_question_gen, prompt=prompt)
# print("-------------------2------------------\n\n")
# retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
# print("-------------------3------------------\n\n")
# combine_docs_chain = create_stuff_documents_chain(llm_question_gen, retrieval_qa_chat_prompt)
# print("-------------------4------------------\n\n")
# # retriever = FAISS.load_local("vector_db", embeddings).as_retriever()
# retriever = vector_store.as_retriever()
# print("-------------------5------------------\n\n")
# # conversationalRetrievalChain = create_stuff_documents_chain(
# #     combine_docs_chain=combine_docs_chain,
# #     retriever=retriever,
# #     question_generator=question_generator_chain,
# # )

# conversationalRetrievalChain = create_retrieval_chain(retriever, combine_docs_chain)
# print("-------------------6------------------\n\n")
# # Run conversational question generation chain
# nextQuestion = conversationalRetrievalChain.invoke({"input": "We will open from 9am to 7pm"})
# print("-------------------7------------------\n\n")
# print(nextQuestion)

# with open('questions.txt') as f:
#     questions = f.readlines()
#     random_question = random.choice(questions)
# print(random_question)

# @app.post("/start-question-generation/")
# async def start_question_generation(background_tasks: BackgroundTasks):
#     task_id = str(uuid.uuid4())
#     task_status[task_id] = "Pending"
#     # Pass the documents to generate questions from as an argument
#     background_tasks.add_task(generate_questions, task_id, docs_question_gen)
#     return {"task_id": task_id, "message": "Question generation started in the background"}

# @app.get("/task-status/{task_id}")
# async def get_task_status(task_id: str):
#     status = task_status.get(task_id, "Not Found")
#     return {"task_id": task_id, "status": status}

@app.get("/")
def root():
   return {"hello world"}

@app.get("/question/")
async def get_question():
    with open('questions.txt') as f:
     questions = f.readlines()
     random_question = random.choice(questions)
    #  print(random_question)
    return {"question": random_question}

# Endpoint to receive and process the user's answer
@app.post("/answer/")
async def process_answer(answer: str):
    # Ensure conversation_history is properly initialized as a list
    global conversation_history
    if not isinstance(conversation_history, list):
        conversation_history = []

    # Append the user's answer to the conversation history
    conversation_history.append(answer)

    # Generate a new question based on the conversation history
    new_question = generate_questions(conversation_history)

    # Return the new question to the user
    return {"question": new_question}


# Placeholder for generating a new question based on the context of the conversation
# def generate_question(conversation_history):
#     # Placeholder logic, replace with your actual logic
#     return "What else would you like to know?"

# Endpoint to receive the first question and start the conversation
@app.get("/start/")
async def start_conversation():
    # Generate the first question
    first_question = "Can you play a role as a customer at Pharmacy? And I am a pharmacist. Can you ask me one short question as a customer?"

    # Return the first question to the user
    return {"question": first_question}

# Endpoint to receive the next question based on the previous answer
@app.post("/next/")
async def next_question(answer: str):
   conversation_history.append(answer)
# Generate the next question based on the conversation history
   next_question = question_new_gen_chain.run(docs_question_gen)
   print(next_question)
   questions_list = next_question.split("\n")  # Split by newline character
   print(questions_list)
# Filter out empty strings and non-question lines
   questions_list = [
    line.strip()
    for line in questions_list
    if line.strip() and line.strip().endswith("?")
]
   random_new_question = random.choice(questions_list)
   print(random_new_question)
   return {"question": random_new_question}

@app.post("/stream_chat/")
async def process_questions():
  # Ensure questions are generated
  try:
    question_list = question_storage
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
  
  # Process each question and collect answers
  answers = []
  for question in question_list:
    if question.strip():  # Ensure question is not just whitespace
      answer = answer_gen_chain.run(question)  # Process the question
      answers.append({"question": question, "answer": answer})

  # Return the list of question-answer pairs
  return {"responses": answers}

  # Process each question and collect answers
#   answers = []
#   for question in question_list:
#     if question.strip():  # Ensure question is not just whitespace
#       answer = answer_gen_chain.run(question)  # Process the question
#       answers.append({"question": question, "answer": answer})

#   # Return the list of question-answer pairs
#   return {"responses": answers}


#################################################### OLD code for reference
# pdf_indexer = PDFIndexer("Basicsofpharmacy", llm_question_gen, embeddings)
# canada_engine = pdf_indexer.index_pdf(os.path.join("data", "Basicsofpharmacy.pdf"))

# while (answer := input("Conversation: Enter an Answer (q to quit): ")) != "q":
#     start_time = datetime.datetime.now()

#     question = canada_engine.query(answer)

#     end_time = datetime.datetime.now()
#     execution_time = end_time - start_time

#     print("Conversation:", question)
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

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
    