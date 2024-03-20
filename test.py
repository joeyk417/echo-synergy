from dotenv import load_dotenv
import os
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.llms.replicate import Replicate
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory


load_dotenv()

# Load documents
pdf_path = os.path.join("data", "Basicsofpharmacy.pdf")
loader = PyPDFLoader(pdf_path)
data = loader.load()
print("Documents loaded successfully.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap=20)
chunks = text_splitter.split_documents(data)

# Create vector database for answer generation
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)
print("HuggingFaceEmbeddings loaded successfully.")

# Initialize vector store for answer generation
def get_vector_store():
    vector_store_name = "vector_store_test"
    if not os.path.exists(vector_store_name):
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings,persist_directory=vector_store_name)
        # vector_store.persist()
        print("Vector store loaded successfully.")
    else:
        vector_store = Chroma(persist_directory=vector_store_name, embedding_function=embeddings)
        print("Vector store loaded successfully.")
    return vector_store

vector_store = get_vector_store()
retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Large Language Model for answer generation
llm = Replicate(
    model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    model_kwargs={"temperature": 0.01, "max_length": 500, "top_p": 1},
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})

# custom_template = """Given the following conversation and a follow-up message, \
#     rephrase the follow-up message to a stand-alone question or instruction that \
#     represents the user's intent, add context if necessary to generate a complete and \
#     unambiguous question or instruction, only based on the history, don't make up messages. \
#     Maintain the same language as the follow up input message.

#     Chat History:
#     {chat_history}

#     Follow Up Input: {question}
#     Standalone question or instruction:"""
    
custom_template = """You are an customer in a pharmacy. You will perform specific tasks: greeting the pharmacist, inquiring 
about medications, understanding prescription requirements, expressing gratitude, and concluding the 
interaction. You does not possess deep knowledge about medications or medical conditions and 
should avoid discussing unrelated or random topics. All responses and queries from you should be 
concise, limited to 1 or 2 sentences. Can you play a role as a customer at Pharmacy. 
And I am a pharmacist. Can you ask me some questions as a customer?  You ask me questions one by one. So we will talk as real conversations?

Chat History:
{chat_history}

Follow Up Input: {question}
"""

sys_prompt = PromptTemplate(
    template=custom_template, input_variables=["chat_history", "question"]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

def get_answer(question, chat_history=[]):
    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            verbose=True,
            memory=memory,
            return_generated_question=True,
            retriever=retriever,
            condense_question_prompt=PromptTemplate.from_template(custom_template),
        )
        result = chain({"question": question, "chat_history": chat_history})

    except Exception as e:
        raise Exception(e)

    return result["answer"]

chat_history=[]
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = get_answer(prompt, chat_history)
    print(result)
    
# Can I take two different allergy medications at the same time, or would that be dangerous?
# chain = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever, verbose = True)
# chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, verbose=True)

# def json_to_chathistory(history):
#     res = []
#     for each in history:
#         res.append((each[0], each[1]))
#     return res

# chat_history = []

# query = "Can I take two different allergy medications at the same time, or would that be dangerous?"
# print("Customer query:", query)
# chain({"question":query, "chat_history":chat_history})

# while (prompt := input("Enter a prompt (q to quit): ")) != "q":
#     result = chain({"question":prompt, "chat_history":chat_history})
#     chat_history = json_to_chathistory(chat_history)
#     print(result.answer)
    
# chat_history = []