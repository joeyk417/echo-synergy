import os
import uuid
from replit import database
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


load_dotenv()

my_secret = os.environ['REPLICATE_API_TOKEN']
app = FastAPI()

# Task status tracker
task_status: Dict[str, str] = {}

# In a real application, this could be a database or cache
question_storage: List[str] = []

class Question(BaseModel):
    text: str  # For receiving question text via POST request
def generate_questions():
  # Placeholder for the actual logic to generate questions
  questions = question_gen_chain.run(docs_question_gen)
  question_storage.extend(questions)
  # You would then save these questions to a database or cache

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
# Load documents
pdf_path = os.path.join("data", "Basicsofpharmacy.pdf")
loader = PyPDFLoader(pdf_path)
data = loader.load()

## To do, load docs from .data folder
# reader = SimpleDirectoryReader("./data")
# documents = reader.load_data()
print("documents loaded successfully.")

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
    "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
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
# Run question generation chain
questions = question_gen_chain.run(docs_question_gen)

# Initialize Large Language Model for answer generation
llm_answer_gen = Replicate(
    model=
    "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
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
print("HuggingFaceEmbeddings loaded successfully.")

# Initialize vector store for answer generation
vector_store = Chroma.from_documents(docs_question_gen, embeddings)
print("Vector store loaded successfully.")

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
#   modalAnswer = answer_gen_chain.run(question)
#   print("Modal Answer is : ", modalAnswer)
f = open("questionsString.txt", "a")
f.writelines(question_list)
# print("--------------------------------------------------\n\n")
# print("-------------------Next Question------------------\n\n")

# Define the path to your file
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

@app.post("/start-question-generation/")
async def start_question_generation(background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_status[task_id] = "Pending"
    # Pass the documents to generate questions from as an argument
    background_tasks.add_task(generate_questions, task_id, docs_question_gen)
    return {"task_id": task_id, "message": "Question generation started in the background"}

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    status = task_status.get(task_id, "Not Found")
    return {"task_id": task_id, "status": status}

@app.get("/questions/")
async def get_questions():
    # Returns the list of stored questions
    #   db["questions"].append(question)
    # for question in question_list:
    question1 = (db["questions"][0])
    return {"questions": question1}

@app.post("/stream_chat/")
async def process_questions():
  # Ensure questions are generated
  try:
    question_list = question_storage
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
  
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
