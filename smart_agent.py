import datetime
import random
from langchain import hub
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import Replicate
from langchain.agents import AgentExecutor, create_react_agent

# defining a single tool
def meaning_of_time(input=""):
    return 'Time goes on and on and on'

time_tool = Tool(
    name = "Time",
    func=meaning_of_time,
    description="useful for when you need to answer questions about current events. You should ask targeted questions"
)

def meaning_of_life(input=""):
    return 'The meaning of life is 42 if rounded but is actually 42.17658'

life_tool = Tool(
    name='Meaning of Life',
    func= meaning_of_life,
    description="Useful for when you need to answer questions about the meaning of life. input should be MOL "
)

def random_num(input=""):
    return random.randint(0,5)

random_tool = Tool(
    name='Random number',
    func= random_num,
    description="Useful for when you need to get a random number. input should be 'random'"
)

# create an agent with the tools
tools = [time_tool, random_tool, life_tool]

# conversational agent memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    return_messages=True
)

llm = Replicate(
    model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    model_kwargs={"temperature": 0.01, "max_length": 500, "top_p": 1},
)

# create our agent
# conversational_agent = initialize_agent(
#     agent='chat-conversational-react-description',
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     max_iterations=3,
#     early_stopping_method='generate',
#     memory=memory
# )
# Construct the JSON agent
# Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/react")


fixed_prompt = """Assistant is a large language model trained by OpenAI.
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant doesn't know anything about random numbers or anything related to the meaning of life and should use a tool for questions about these topics.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
"""
agent = create_react_agent(llm, tools, fixed_prompt)
# print("prompt:", prompt)

# conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt

# conversational_agent(prompt)

# conversational_agent.run("Hi How are you?")

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    # verbose=True, 
    handle_parsing_errors=True,
    prompt=fixed_prompt,
    # max_iterations=3,
    # max_execution_time=5, 
    memory=memory
)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    start_time = datetime.datetime.now()
    
    # result = conversational_agent.run(prompt)
    result=agent_executor.invoke({"input": prompt})
    
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    
    print(result)
    print("Execution time:", execution_time.total_seconds())
    
memory.clear()