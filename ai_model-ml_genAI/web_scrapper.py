import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

# 1. Load your API key from the local .env file
load_dotenv()
groq_api_key = os.getenv("API_KEY")

# 2. Fetch the Cricbuzz rankings page data
url = "https://www.cricbuzz.com/cricket-stats/icc-rankings/men/batting"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"} # Prevents website blocks
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Clean up raw text by stripping away unnecessary navbar/javascript noise
for element in soup(["script", "style", "nav", "footer"]):
    element.decompose()
page_text = " ".join(soup.get_text().split())

# 3. Initialize the direct, working Groq client
client = Groq(api_key=groq_api_key)
MODEL_NAME = "openai/gpt-oss-20b"  # Active Groq production model

# 4. Request structured JSON output natively from Groq
completion = client.chat.completions.create(
    model=MODEL_NAME,
    response_format={"type": "json_object"},  # Forces the model to reply in clean, valid JSON
    messages=[
        {
            "role": "system", 
            "content": (
                "You are an expert sports data analyst. Extract the ICC Men's Batting Rankings from the text provided. "
                "You must respond with a JSON object containing a single key 'rankings', which maps to a list of players. "
                "Each player item must be an object with the exact fields: PlayerName, Country, and RankingPoints."
            )
        },
        {
            "role": "user", 
            "content": f"Extract the batting rankings from this page content:\n{page_text[:6000]}" # Truncated to fit token windows safely
        }
    ]
)

# 5. Read the response using proper list indexing to avoid AttributeErrors
raw_json_response = completion.choices[0].message.content
response_data = json.loads(raw_json_response)

# 6. Print nicely formatted JSON data
print(json.dumps(response_data, indent=2))
