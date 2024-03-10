from dotenv import load_dotenv
import os
from pdf import PDFIndexer
from llama_index.core import Settings
from llama_index.llms.replicate import Replicate

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer
from llama_index.llms.replicate import Replicate

load_dotenv()

# REPLICATE_API_TOKEN = "r8_DBCpxyYhxOEbWCqf61d9SbYvp5YxSb51RJuvI"  # Your Relicate API token here
# os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# population_path = os.path.join("data", "population.csv")
# population_df = pd.read_csv(population_path)

# population_query_engine = PandasQueryEngine(
#     df=population_df, verbose=True, instruction_str=instruction_str
# )
# population_query_engine.update_prompts({"pandas_prompt": new_prompt})

# # tools = [
# #     note_engine,
# #     QueryEngineTool(
# #         query_engine=population_query_engine,
# #         metadata=ToolMetadata(
# #             name="population_data",
# #             description="this gives information at the world population and demographics",
# #         ),
# #     ),
#     QueryEngineTool(
#         query_engine=canada_engine,
#         metadata=ToolMetadata(
#             name="canada_data",
#             description="this gives detailed information about canada the country",
#         ),
#     ),
# # ]


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

#agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

pdf_indexer = PDFIndexer("canada", Settings.llm, Settings.embed_model)
canada_engine = pdf_indexer.index_pdf(os.path.join("data", "Canada.pdf"))

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = canada_engine.query(prompt)
    print(result)
    
