import requests
import json

url = "http://localhost:8000/v1/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer useless"
}

your_question = "Where is Cornell University?"

data = {
    "model": "meta-llama/Llama-3.1-8B",
    "prompt": f"""
              <|begin_of_text|><|start_header_id|>system<|end_header_id|>
              Cutting Knowledge Date: December 2023
              Today Date: 23 July 2024
              You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
              {your_question}<|eot_id|>
              <|start_header_id|>assistant<|end_header_id|>
              """,
    "max_tokens": 10240,
    "temperature": 0,
    "ignore_eos": "False"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.json())
