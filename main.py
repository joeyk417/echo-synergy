import asyncio
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from llama_index.core import Settings, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.evaluation import DatasetGenerator
from llama_index.readers.file import PDFReader
import sys

load_dotenv()

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class RetryError(Exception):
    pass

async def make_request_with_retry(request_func, *args, retries=5, delay=2.0, **kwargs):
    for attempt in range(retries):
        try:
            return await request_func(*args, **kwargs)
        except Exception as e:  # Adapt this exception to your specific needs
            print(f"Attempt {attempt+1} failed with error: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise RetryError("Max retries exceeded.") from e

async def main():
    print("Initializing Groq model...")
    Settings.llm = Groq(model="llama2-70b-4096")
    print("Groq model initialized successfully.")
    
    Settings.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    pdf_path = os.path.join("data", "Canada.pdf")
    documents = PDFReader().load_data(file=pdf_path)
    
    service_context = ServiceContext.from_defaults(llm=Settings.llm, embed_model=Settings.embed_model)
    print("Service context initialized successfully.")

    # Assuming the `DatasetGenerator.from_documents` is the method that eventually triggers the rate-limited API call
    try:
        data_generator = DatasetGenerator.from_documents(documents, service_context=service_context, num_questions_per_chunk=2, show_progress=True)
        questions = await make_request_with_retry(data_generator.agenerate_questions_from_nodes)
        print("Questions generated successfully:")
        for question in questions:
            print(question)
    except RetryError as e:
        print(f"Failed to generate questions: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())


# Settings.llm = Groq(
#     model="llama2-70b-4096",
#     temperature=0.01,
#     system_prompt=system_prompt,
#     # query_wrapper_prompt=query_wrapper_prompt,
#     additional_kwargs={"top_p": 1, "max_new_tokens": 300},
#     is_chat_model=True,
# )