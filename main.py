import asyncio
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from llama_index.core import Settings, ServiceContext, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.groq import Groq
from llama_index.llms.replicate import Replicate
# from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.readers.file import PDFReader
import sys
from prompts import system_prompt, query_wrapper_prompt



# Initialize environment and set up for Windows asyncio compatibility
load_dotenv()

async def generate_dataset(documents, service_context, num_questions_per_chunk):
    data_generator = RagDatasetGenerator.from_documents(
        documents=documents, 
        # service_context=service_context, 
        llm=Settings.llm,
        num_questions_per_chunk=num_questions_per_chunk,
        show_progress=True
    )
    print("data generated successfully.")
    # questions = await data_generator.generate_dataset_from_nodes()
    questions = await data_generator.generate_questions_from_nodes()
    print("question generated successfully.")
    return questions

async def main():
    print("Initializing settings...")

    # set the LLM
    # system_prompt = "Your system prompt here"  # Replace "Your system prompt here" with the desired value

    # Settings.llm = Groq(model="llama2-70b-4096")

    llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
    Settings.llm = Replicate(
    model=llama2_7b_chat,
    temperature=0.1,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300},
    is_chat_model=True,)

    Settings.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    service_context = ServiceContext.from_defaults(llm=Settings.llm, embed_model=Settings.embed_model)
    print("Settings initialized successfully.")

    # Load documents
    pdf_path = os.path.join("data", "Good_Customer_Service_Guidance.pdf")
    documents = PDFReader().load_data(file=pdf_path)
    # reader = SimpleDirectoryReader("./data")
    # documents = reader.load_data()
    print("documents loaded successfully.")
    
    # Adjust the chunk size as a strategy to manage API rate limits
    num_questions_per_chunk = 2  # Consider lowering this if you're hitting rate limits

    # Generate questions with retry logic
    try:
        eval_questions  = generate_dataset, documents, service_context, num_questions_per_chunk
        print("Questions generated successfully:")
        
        for question in eval_questions:
            print(question)

        # Split generated questions into a list of questions
        # question_list = eval_questions .split("\n")
            
    except Exception as e:  # Consider catching specific exceptions
        print(f"Failed to generate questions due to an error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
