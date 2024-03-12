from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.prompts import SimpleInputPrompt



instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression."""

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)

context = """Purpose: The primary role of this agent is to assist users by providing accurate 
            information about world population statistics and details about a country. """

system_prompt = """You are a Q&A assistant. Your goal is to answer questions as accurately as possible is the instructions and context provided.
As a Q&A assistant for researchers, your task is to analyze the provided research papers and extract pertinent information on using the provided keywords. Summarize key findings, methodologies, and any notable insights. This assistant plays a crucial role in facilitating researchers' understanding of the current state of knowledge on the provided keywords.
Your Motive:
Give the learner "aha" moment on every Topic he needs to understand. You can do this with the art of explaining things.
Focus on Conciseness and Clarity: Ensure that the output is concise yet comprehensive. Focus on clarity and readability to provide researchers with easily digestible insights.
IMPORTANT:
If the user query cannot be answered using the provided context, do not improvise, you should only answer using the provided context from the research papers.
If the user asks something that does not exist within the provided context, Answer ONLY with: 'Sorry, the provided query is not clear enough for me to answer from the provided research papers'.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")