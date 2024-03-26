import datetime
from dotenv import load_dotenv
import re
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.replicate import Replicate


load_dotenv()

template = """Assistant is a large language model.
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions. Do not provide explanations and descriptions on topics.
Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
Follow the below instructions: 
1. You act as a customer, ask a question based on the chat history.
2. Limit your responses to 1 or 2 sentences.
3. Do not have extensive knowledge of medications or medical advice.
4. Do not provide explanations or descriptions on medical topics.
5. Directly say your health concern or ask a question as a customer.
6. Response should be straightforward and brief.
7. Do not provide expression.
8. Reply only with your character's dialogue inside and nothing else.
9. If you don't know the answer, you should say 'I'm not sure, I will check with my GP'.
10. If you are unsure about the response, you should say 'It's ok, I will check with my GP'.
11. If the pharmacist tell you to check with your GP, you should say 'Thanks, I will check with my GP. Bye.'. And stop the conversation.
12. Do not have extensive knowledge of medications or medical advice. - Avoid complex or detailed medication discussions.
13. Cannot simulate payment or insurance-related queries.
14. Should not provide or discuss personal health information.
15. Refrain from engaging in conversations about unrelated or random topics.
16. If the pharmacist asks you to provide more information, you should say 'I'm not sure, I will check with my GP'.
17. If the pharmacist asks you something you don't understand in english, you should say 'Sorry, I don't understand.'

{history}

Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
llm = Replicate(
    model=llama2_7b_chat,
    temperature=0.01,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300},
    )

engine_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=ConversationBufferWindowMemory(k=2),
    llm_kwargs={"max_length": 4096}
)

# character_description = "owns a tavern"
character_description = "acts as a customer and walk into a Pharmacy and want to query the pharmacist"
# character_information = "a band of thieves that has been operating in the area"
character_information = "purchasing over-the-counter medications or supplements here, which ones are recommended for your specific needs"
human_character = "a pharmacist"
# player_first_line = "hello, good to smell a hot meal after a long journey"
player_first_line = "hello, i am the Pharmacist, how can i assist you today?"

output = engine_chain.predict(
human_input=f"I want you to act as a non-player character in a real life situation. Your character {character_description} and has no knowledge about {character_information}. I will type dialogue and you will reply with what the character should say. I want you to only reply with your character's dialogue inside and nothing else. Do not write explanations and any medical information. My character acts as {human_character} and my character's first line of dialogue is '{player_first_line}'."
)
print("AI: ", output)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    start_time = datetime.datetime.now()
    
    #result = canada_engine.query(prompt)
    refine_prompt = f"I want you to continue acting as a customer and answer the human as {human_character} the question: {prompt}. I will type dialogue and you will response with what the character should say and nothing else. Do not write explanations."
    response = engine_chain.predict(human_input=refine_prompt)
    
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    
    content = re.findall(r'"(.+?)"', response)
    print("AI: ", content.pop() if content else "Sorry, I don't understand. Please rephrase your response.")
    print("Execution time:", execution_time.total_seconds())