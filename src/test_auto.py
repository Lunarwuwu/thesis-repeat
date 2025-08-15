import os
import json
import urllib.request
import ssl
import http.client
import time
import socket
from urllib.error import URLError
from human_eval.data import write_jsonl, read_problems


def get_env_or_raise(var_name):
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"Environment variable {var_name} is not set")
    return value

DEEPSEEK_API_KEY = get_env_or_raise("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"


def call_deepseek(prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(DEEPSEEK_API_URL, data=data, headers=headers, method='POST')
    ctx = ssl.create_default_context()

    with urllib.request.urlopen(req, context=ctx, timeout=300) as response:
        if getattr(response, "status", 200) != 200:
            raise urllib.error.HTTPError(
                DEEPSEEK_API_URL, response.status, response.reason, response.headers, None
            )
        try:
            raw_bytes = response.read()
        except http.client.IncompleteRead as e:
            raw_bytes = e.partial

    resp_body = raw_bytes.decode('utf-8', errors='ignore')
    resp_data = json.loads(resp_body)
    return resp_data['choices'][0]['message']['content']


def call_deepseek_with_retry(prompt: str, max_tokens: int = 1024, temperature: float = 0.0, retries: int = 3) -> str:

    for attempt in range(1, retries + 1):
        try:
            return call_deepseek(prompt, max_tokens, temperature)
        except (http.client.IncompleteRead, URLError, socket.timeout, json.JSONDecodeError) as e:
            if attempt == retries:
                print(f"Final attempt failed ({e.__class__.__name__}): {e}")
                raise
            wait = 2 ** attempt
            print(f"Attempt {attempt} failed ({e.__class__.__name__}), retrying in {wait}s...")
            time.sleep(wait)


def extract_code(markdown: str) -> str:
    for fence in ("```python", "```"):
        if fence in markdown:
            start = markdown.index(fence) + len(fence)
            end = markdown.find("```", start)
            end = end if end != -1 else len(markdown)
            return markdown[start:end].strip()
    return markdown.strip()


def generate_test_cases(problem_prompt: str) -> str:

    test_request = (
        problem_prompt
        + "\n\nPlease write pytest test cases covering edge cases and typical inputs."
    )
    raw = call_deepseek_with_retry(test_request)
    return extract_code(raw)


def generate_solution(problem_prompt: str, test_code: str) -> str:

    solution_request = (
        problem_prompt
        + "\n\nHere are the pytest tests:\n```"
        + test_code
        + "```\nImplement the function so that these tests all pass."
    )
    raw = call_deepseek_with_retry(solution_request)
    return extract_code(raw)


def main():
    problems = read_problems()
    samples = []
    failures = []

    for task_id, problem in problems.items():
        prompt = problem['prompt']
        print(f"Processing task {task_id}:")
        try:
            test_code = generate_test_cases(prompt)
            print("Generated tests:\n", test_code)

            solution_code = generate_solution(prompt, test_code)
            print("Generated solution:\n", solution_code)

            samples.append({
                'task_id': task_id,
                'tests': test_code,
                'completion': solution_code,
            })
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            failures.append(task_id)
            continue

    write_jsonl('samples_auto.jsonl', samples)
    print("Done. Samples saved to samples_auto.jsonl")
    if failures:
        print("The following tasks have failed, please rerun them manually:", failures)


if __name__ == '__main__':
    main()
