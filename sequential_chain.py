import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import Replicate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.sequential import SequentialChain
from langchain.chains.llm import LLMChain
import random

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
print("Documents loaded successfully.")

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

# Create vector database for answer generation
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)
print("HuggingFaceEmbeddings loaded successfully.")

# Initialize vector store for answer generation
vector_store = Chroma.from_documents(docs_question_gen, embeddings)
print("Vector store loaded successfully.")

# TODO: answer generation chain here

question_chain = LLMChain(
    llm=llm_question_gen, prompt=REFINE_PROMPT_QUESTIONS, output_key="question", verbose=True
)

# This is the overall chain where we run these two chains in sequence.
generate_evaluate_chain = SequentialChain(
    chains=[question_chain],#[question_chain, response_chain],
    input_variables= ["text", "existing_answer"],#["text", "human_response"],
    # Here we return multiple variables
    output_variables= ["question"],#["question", "ai_response"],
    verbose=True,
)

# Split generated questions into a list of questions
question_list = questions.split("\n")

for _ in range(5):
    random_question = random.choice(question_list)
    # Do something with the random question
    print(random_question)

    while human_response := input("Enter a answer (q to quit): ") != "q":
        result = generate_evaluate_chain(
                {
                    "text": random_question,
                    "existing_answer":random_question,
                    "human_response": human_response,
                    # "grade": grade,
                    # "tone": tone,
                    # "response_json": json.dumps(RESPONSE_JSON),
                }
            )
        print(result)