import os
import json
from dotenv import load_dotenv
from groq import Groq

# 1. Load the environment variables from your local .env file
load_dotenv()
api_key = os.getenv("API_KEY")

# 2. Initialize the direct Groq Client
client = Groq(api_key=api_key)

# Using an active production model on Groq's platform
MODEL_NAME = "openai/gpt-oss-20b" 

invoice_text = """
Date: 10/01/2026  
Patient Name: Sibesh Patel  
Patient ID: P-10245  
Doctor: Dr. Alok Mehta  
Department: Cardiology  
Bill #: HB-2026-001  

Services Rendered:  
Consultation Fee - 300  
X-Ray Imaging - 250  
Blood Test - 150  
Medication Charges - 200  
Room Charges (2 days) - 1000  
Service Tax (5%) - 95  

Payment Method: Credit Card  
Total Amount: 1995  
"""

# 3. Request structured JSON output directly from Groq
completion = client.chat.completions.create(
    model=MODEL_NAME,
    response_format={"type": "json_object"},  # Forces the model to reply in clean JSON
    messages=[
        {
            "role": "system", 
            "content": (
                "You are an expert medical billing assistant. Extract data from the provided invoice text. "
                "You must respond with a JSON object containing keys for: date, patient_name, patient_id, "
                "doctor, department, bill_number, services_rendered (as a list of objects with service and cost), "
                "payment_method, and total_amount."
            )
        },
        {
            "role": "user", 
            "content": f"Extract details from this invoice:\n{invoice_text}"
        }
    ]
)

# 4. Read and print the parsed structured JSON payload
# FIXED: Added [0] index to choices to resolve the AttributeError
raw_json_response = completion.choices[0].message.content
parsed_data = json.loads(raw_json_response)

print(json.dumps(parsed_data, indent=4))
