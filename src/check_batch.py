import os
from openai import OpenAI
import json
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_PJ")
client = OpenAI(api_key=OPENAI_API_KEY)

response = client.batches.list(limit=100)
print(response)
# check = client.batches.retrieve("batch_686c820937a88190a0780776a2b7c356")
# # print(check)
# result_content_response = client.files.content(check.output_file_id)
# result_content = result_content_response.read().decode('utf-8')
# for line in result_content.strip().split('\n'):
#     result_json = json.loads(line)
#     custom_id = result_json.get('custom_id', '')
#     response_body = result_json.get('response', {}).get('body', {})
    
#     if not custom_id or not response_body:
#         continue
        
#     csv_path, timestep_start = custom_id.split('|')
#     content = response_body.get('choices', [{}])[0].get('message', {}).get('content')
    
#     if content:
#         if content == "no contact":
#             content = "No contact.;no contact"
#         annotation, label = content.rsplit(";", 1)
        # print(annotation, label)
#batch_686c7e38d9c88190b3bf96b458fba40f
# batches = [b for b in response]
# for batch in batches:
#     client.batches.cancel(batch.id)
#     print(f"Cancelled Batch ID: {batch.id}")