import os
import json
from dotenv import load_dotenv
from groq import Groq

# 1. Load your API key from the local .env file
load_dotenv()
groq_api_key = os.getenv("API_KEY")

# 2. Initialize the direct, working Groq client
client = Groq(api_key=groq_api_key)
MODEL_NAME = "openai/gpt-oss-20b"  # Active Groq production model

# Sample SOW paragraph
sow_text = """
Project Title: Data Warehouse Modernization
Client: Acme Corp
Project ID: DW-2026-001
Date: 10/01/2026

Scope: The vendor will migrate Acme Corp’s legacy data warehouse to Azure Synapse Analytics. Deliverables include data migration scripts, ETL pipelines, and gold-layer reporting dashboards. The project is expected to be completed within 3 months.

Responsibilities: The vendor will provide technical resources and documentation. The client will ensure access to legacy systems and provide test data.

Payment Terms: The total cost is $120,000, payable in three installments of $40,000 each at the end of every project month.
"""

# 3. Simulate LangChain's JsonOutputParser formatting guidelines natively
format_instructions = (
    "Return a JSON object with the following fields: project_title, client, project_id, "
    "date, scope (including migration_target and deliverables as a list), "
    "responsibilities (vendor and client), and payment_terms (total_cost and installments)."
)

# 4. Request structured JSON output directly from Groq using native features
completion = client.chat.completions.create(
    model=MODEL_NAME,
    response_format={"type": "json_object"},  # Forces the model to reply in clean, valid JSON
    messages=[
        {
            "role": "system", 
            "content": f"Extract meaningful details from the following SOW paragraph and output in JSON format.\n{format_instructions}"
        },
        {
            "role": "user", 
            "content": f"Statement of Work text:\n{sow_text}"
        }
    ]
)

# 5. Read the response using proper list indexing [0] to avoid AttributeErrors
raw_json_response = completion.choices[0].message.content
response_data = json.loads(raw_json_response)

# 6. Print nicely formatted JSON
print(json.dumps(response_data, indent=2))
