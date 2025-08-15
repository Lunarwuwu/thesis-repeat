import os
import json
import urllib.request
import urllib.error
import ssl
from human_eval.data import write_jsonl, read_problems
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat" 

def get_python_code(code):
    code = code.replace("\r", "")
    if "```python" in code:
        code_start_idx = code.index("```python")
        code = code[code_start_idx:].replace("```python", "").strip()
        end_idx = code.find("```") if "```" in code else len(code)
        code = code[:end_idx].strip()

    return code

def generate_one_completion(prompt: str) -> str:
    print(prompt)
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json", 
    }

    payload_dict = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
        "stream": False
    }

    data    = json.dumps(payload_dict).encode('utf-8')
    req     = urllib.request.Request(DEEPSEEK_API_URL, data=data, headers=headers, method='POST')
    context = ssl._create_default_context() 

    try:
        with urllib.request.urlopen(req, context=context, timeout=1000) as response:
            if response.status != 200:
                print(f"HTTP error: {response.status} {response.reason}")
                try:
                    error_body = response.read().decode('utf-8')
                    print(f"Error details: {error_body}")
                except Exception as read_err:
                    print(f"Failed to read error response body: {read_err}")
                return ""

           
            response_body_bytes = response.read()
            response_body_str = response_body_bytes.decode('utf-8')
            response_data = json.loads(response_body_str)

            if response_data.get("choices") and len(response_data["choices"]) > 0:
                completion = response_data["choices"][0].get("message", {}).get("content", "")
                # return completion.strip()
                return get_python_code(completion)
            else:
                print(f"Warning: No valid API response found. completion: {response_data}")
                return ""

    except urllib.error.HTTPError as e:
        print(f"HTTP error occurred when calling the DeepSeek API: {e.code} {e.reason}")
        try:
            error_body = e.read().decode('utf-8')
            print(f"Error details: {error_body}")
        except Exception as read_err:
            print(f"Failed to read HTTP Error response body: {read_err}")
        return ""
    except urllib.error.URLError as e:
        print(f"URL error occurred when calling the DeepSeek API: {e.reason}")
        return ""
    except json.JSONDecodeError:
        print(f"Error parsing API response JSON (the response may not be valid JSON)")
        return ""
    except Exception as e:
        print(f"An unknown error occurred while calling the DeepSeek API: {e}")
        return ""



def format_prompt(problem):
    return f'>>> Problem:\n{problem["prompt"]}\n>>> Code:\n```python'


def main():
    problems = read_problems()
    samples = []

    for task_id, problem in list(problems.items()):  
        prompt = problem['prompt']
        print("---------------------------------------------")
        print(f"Processing task {task_id}:")

        solution_code = generate_one_completion(format_prompt(problems[task_id]))
        samples.append({
            'task_id': task_id,
            'completion': solution_code,
        })
    file_name = "samples_direct.jsonl"
    write_jsonl(file_name, samples)
    print(f"Done. Samples saved to {file_name}")

if __name__ == '__main__':
    main()
