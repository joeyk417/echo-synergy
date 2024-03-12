from dotenv import load_dotenv
import os
from pdf import PDFIndexer
from llama_index.core import Settings
from llama_index.llms.replicate import Replicate

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer

from llama_index.core import StorageContext, VectorStoreIndex, ServiceContext, SimpleDirectoryReader, load_index_from_storage
from llama_index.core.evaluation import DatasetGenerator, ResponseEvaluator, QueryResponseEvaluator

load_dotenv()

# The replicate endpoint
LLAMA_13B_V2_CHAT = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

# set the LLM
llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
Settings.llm = Replicate(
    model=llama2_7b_chat,
    temperature=0.01,
    system_prompt="You are a Q&A assistant. Your goal is to answer questions as accurately as possible is the instructions and context provided.",
    additional_kwargs={"top_p": 1, "max_new_tokens": 300},
)

# set tokenizer to match LLM
Settings.tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf"
)

# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

pdf_indexer = PDFIndexer("canada", Settings.llm, Settings.embed_model)
canada_engine = pdf_indexer.index_pdf(os.path.join("data", "Canada.pdf"))

# while (prompt := input("Enter a prompt (q to quit): ")) != "q":
#     result = canada_engine.query(prompt)
#     print(result)

# Load documents
# reader = SimpleDirectoryReader(input_files=['Canada.pdf'])
# reader = SimpleDirectoryReader("./data")
# documents = reader.load_data()

# service_context = ServiceContext.from_defaults(
#     llm_predictor=Settings.llm, chunk_size_limit=3000
# )

service_context=ServiceContext.from_defaults(llm=Settings.llm, embed_model=Settings.embed_model)

# Generate Question
# data_generator = DatasetGenerator.from_documents(canada_engine, service_context = service_context)
# questions = data_generator.generate_questions_from_nodes()
# print(questions)

# Let's just use a meaningful subset of the shuffled documents.
# random_documents = copy.deepcopy(documents)
# random_shuffle(random_documents)
# random_documents = random_documents[:10]


# Let's reduce the number of questions per chunk.
data_generator = DatasetGenerator.from_documents(
    canada_engine, service_context=service_context, num_questions_per_chunk=2
)
# Let's reduce the number of questions per chunk from 10 to 2.
