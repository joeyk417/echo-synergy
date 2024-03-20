from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain.llms.replicate import Replicate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage

llm = Replicate(
    model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    model_kwargs={"temperature": 0.01, "max_length": 500, "top_p": 1},
)
 
# template = """Act as an experienced high school teacher that teaches {subject}. 
# Always give examples and analogies"""
template ="""You are an customer in a pharmacy. You will perform specific tasks: greeting the pharmacist, inquiring 
about medications, understanding prescription requirements, expressing gratitude, and concluding the 
interaction. You does not possess deep knowledge about medications or medical conditions and 
should avoid discussing unrelated or random topics. All responses and queries from you should be 
concise, limited to 1 or 2 sentences. Can you play a role as a customer at Pharmacy. 
And I am a pharmacist. Can you ask me some questions as a customer?  You ask me questions one by one. So we will talk as real conversations?
For example:
{subject}
"""

human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(template),
        AIMessage(content="Hello, how are you!"),
        HumanMessage(content="Hello, how can I help today?"),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
)

subject = "Can I take two different allergy medications at the same time, or would that be dangerous?"
start_messages = chat_prompt.format_messages(
    subject=subject, 
    text="Hello, how can I help today?"
)
result = llm.predict_messages(start_messages)
print(result.content)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    print("\ntest if subject is being updated:", subject)
    messages = chat_prompt.format_messages(
    subject=subject, 
    text=prompt
    )
    response = llm.predict_messages(messages)
    print(response.content)
    print("\n")
    subject = response.content