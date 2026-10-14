# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from groq import Groq
# import os
# from langchain_core.prompts import ChatPromptTemplate


# # 1. Load variables from the local .env file
# load_dotenv()

# # 2. Initialize the client using your secure key
# client = Groq(api_key=os.getenv("API_KEY"))

# llm = ChatGroq(
#     model = "openai/gpt-oss-120b",
#     api_key = client
# )

# prompt = ChatPromptTemplate ([
#     ('system','acts as an python expert'),
#     ('user','what is python')
# ])

# chain = prompt | llm 
# response = chain.invoke()
# print(response.content)


import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load variables from the local .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

# 2. Replicate the LangChain prompt template logic natively
class CleanPromptTemplate:
    def __init__(self, messages):
        self.messages = messages

    def format(self, user_input_dict):
        formatted_messages = []
        for role, text in self.messages:
            # Replace placeholder tokens like {question} with actual values
            formatted_text = text.format(**user_input_dict)
            formatted_messages.append({"role": role, "content": formatted_text})
        return formatted_messages

# 3. Replicate the LangChain ChatGroq wrapper logic natively
class NATIVE_ChatGroq:
    def __init__(self, model, api_key):
        self.model = model
        # Initialize the direct working client internally
        self.client = Groq(api_key=api_key)

    def invoke(self, formatted_messages):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages
        )
        # Wrap response content in an object matching LangChain's structure (.content)
        class ResponseObject:
            def __init__(self, content):
                self.content = content
        return ResponseObject(completion.choices[0].message.content)

# 4. Replicate the LangChain Pipe operator (|) connection logic natively
class NativeChain:
    def __init__(self, prompt, llm):
        self.prompt = prompt
        self.llm = llm

    def invoke(self, input_data):
        formatted = self.prompt.format(input_data)
        return self.llm.invoke(formatted)


# ==========================================
# THIS IS THE EXACT REPLACEMENT FOR YOUR CODE
# ==========================================

# Initialize our custom native wrapper
llm = NATIVE_ChatGroq(
    model="openai/gpt-oss-20b",  # FIXED: Removed the typo string and added an active production model
    api_key=API_KEY
)

system_instructions = """
         -act as a Python expert 
         -answer only python quetions
         -if someone is asking other than python , say : i have only Python knowledge.
         """

# Replicate the prompt architecture
prompt = CleanPromptTemplate([
    ('system', system_instructions ),
    ('user', '{input}')
])

quetion = input('You: ')
# Connect them using our standard chain connector
chain = NativeChain(prompt, llm)

# Execute the call sequence
response = chain.invoke({"input": quetion})
print(response.content)

while True:
    question = input('You: ')
    if question.lower() in ['exit', 'quit', 'bye']:
        print('Goodbye')
        break

    response = chain.invoke({'input': question})
    print(f"\nAssistant:\n", response.content, "\n")
