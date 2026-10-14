import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load variables from the local .env file
load_dotenv()

# 2. Initialize the client using your secure key
client = Groq(api_key=os.getenv("API_KEY"))

# Example #1: Prime Minister Check
completion1 = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Who is the Prime Minister of India?"}]
)
# FIXED: Pointing to index 0 of the choices list
print(completion1.choices[0].message.content)
print("\n" + "="*40 + "\n")

# Example #2: Palindrome function
completion2 = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Write a Python function to check if a string is a palindrome."}]
)
# FIXED: Pointing to index 0 of the choices list
print(completion2.choices[0].message.content)
print("\n" + "="*40 + "\n")

# Example #3: Data Engineering Poem
completion3 = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Write a short motivational poem about learning data engineering."}]
)
# FIXED: Pointing to index 0 of the choices list
print(completion3.choices[0].message.content)
