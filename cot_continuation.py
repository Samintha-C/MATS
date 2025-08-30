import os
import time
import json
import requests
from openai import OpenAI

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
NOVITA_API_KEY = os.environ.get("NOVITA_API_KEY")

# Models/endpoints
OR_BASE = "https://openrouter.ai/api/v1"
OR_MODEL = "deepseek/deepseek-r1-distill-qwen-14b"
NOVITA_BASE = "https://api.novita.ai/openai"
NOVITA_MODEL = "deepseek/deepseek-r1-distill-qwen-14b"

# Default generation parameters
GEN_TEMPERATURE = 0.6
GEN_TOP_P = 0.95
MAX_CONT_TOKENS = 32000

def or_complete(prompt: str,
                max_tokens: int = 1000000,
                temperature: float = 0.6,
                top_p: float = 0.95,
                stop=None) -> str:
    """
    OpenRouter /completions (non-chat) call, robust to transient errors and empty choices.
    """
    print("REACHESOR")
    url = f"{OR_BASE}/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional attribution headers:
        # "HTTP-Referer": "https://your.app",
        # "X-Title": "Your App Name",
    }
    payload = {
        "model": OR_MODEL,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
    }
    if stop:
        payload["stop"] = stop

    delay = 2.0
    for attempt in range(4):
        print('gets to attempt')
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        # retry on common transient statuses
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay); delay *= 2; continue

        # try to parse JSON regardless of status
        try:
            data = r.json()
        except Exception:
            raise RuntimeError(f"OpenRouter non-JSON response [{r.status_code}]: {r.text[:500]}")

        # print raw for diagnosis (trim long bodies)
        print("RAW RESPONSE:", json.dumps(data, indent=2)[:1800])

        if r.status_code != 200:
            msg = (data.get("error") or {}).get("message") or data
            raise RuntimeError(f"OpenRouter error [{r.status_code}]: {msg}")

        choices = data.get("choices") or []
        if not choices or "text" not in choices[0]:
            raise RuntimeError(f"OpenRouter returned no choices/text: {data}")

        return choices[0]["text"]

    raise RuntimeError("OpenRouter failed after retries")

def novita_complete(prompt: str,
                    model: str = NOVITA_MODEL,
                    max_tokens: int = 32000,
                    temperature: float = 0.6,
                    top_p: float = 0.95,
                    stop=None,
                    timeout_s: float = 360.0,
                    max_retries: int = 4) -> str:
    """Complete using Novita API"""
    url = "https://api.novita.ai/openai/v1/completions"
    headers = {"Authorization": f"Bearer {NOVITA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": 1,
        "stream": False,
    }
    if stop:
        payload["stop"] = stop

    backoff = 2.0
    for attempt in range(max_retries):
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        print("NOVITA STATUS:", r.status_code)
        if r.status_code in (429, 500, 502, 503, 504):
            if attempt == max_retries - 1:
                raise RuntimeError(f"Novita transient error [{r.status_code}]: {r.text[:400]}")
            time.sleep(backoff); backoff *= 2; continue
        try:
            data = r.json()
        except Exception:
            raise RuntimeError(f"Novita non-JSON response [{r.status_code}]: {r.text[:400]}")
        if r.status_code != 200:
            msg = (data.get("error") or {}).get("message") or str(data)
            raise RuntimeError(f"Novita error [{r.status_code}]: {msg}")
        choices = data.get("choices") or []
        if not choices or "text" not in choices[0]:
            raise RuntimeError(f"Novita returned no choices/text: {json.dumps(data)[:400]}")
        return choices[0]["text"]
    raise RuntimeError("Novita failed after retries")

def continuation_prompt(problem_text: str, cot_prefix: str, forced_answer: bool = False) -> str:
    """
    Match the paper/original style: a single completion prompt with <think> block.
    Keep the prefix inside <think> so the model continues naturally in the same style.
    """
    prompt = (
        "Solve this math problem step by step. You MUST put your final answer in \\boxed{}.\n"
        f"Problem: {problem_text}\n"
        "Solution:\n<think>\n"
        f"{cot_prefix.rstrip()}\n"
    )
    if forced_answer:
        # Close thinking and force the box. (Also fix the original typo "answers".)
        prompt += "</think>\n\nTherefore, the final answer is \\boxed{"
    return prompt

def continue_from_prefix_qwen(problem_text: str,
                              prefix: str,
                              max_tokens: int = MAX_CONT_TOKENS,
                              temperature: float = GEN_TEMPERATURE,
                              top_p: float = GEN_TOP_P,
                              forced_answer: bool = False,
                              use_novita: bool = True) -> str:
    """
    Calls the /completions endpoint with a single prompt; returns the raw text completion.
    """
    user_prompt = continuation_prompt(problem_text, prefix, forced_answer=forced_answer)
    print("PROMPT >>>\n", user_prompt[:1200], "\n<<< END PROMPT")
    # Optional: stop sequences to avoid the model jumping to unrelated headers
    stop = ["\nProblem:", "\n# ", "\n\n\n"]
    
    if use_novita:
        return novita_complete(
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )
    else:
        return or_complete(
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )

def continue_from_prefix_openrouter(problem_text: str,
                                   prefix: str,
                                   max_tokens: int = MAX_CONT_TOKENS,
                                   temperature: float = GEN_TEMPERATURE,
                                   top_p: float = GEN_TOP_P,
                                   forced_answer: bool = False) -> str:
    """Wrapper for OpenRouter continuation"""
    return continue_from_prefix_qwen(
        problem_text, prefix, max_tokens, temperature, top_p, forced_answer, use_novita=False
    )
