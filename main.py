import os
import sys
import json
import asyncio
import logging
import time
import re
import subprocess
import urllib.request
import aiohttp
from dotenv import load_dotenv

# Load local environment variables (if any, for development only)
load_dotenv()

# Setup logging to stdout only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("agent")

# Import system modules
from classifier import classify_prompt
from validators import validate_category_output
from client import FireworksClient, get_max_tokens, get_emergency_fallback

# Model configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")

# DEV_MODE toggle (defaulting to False for production submission)
DEV_MODE = os.environ.get("DEV_MODE", "false").lower() == "true"

# Global local model state
local_server_process = None
local_disabled = True

# System Prompt Caching
from client import SYSTEM_PROMPTS

def format_chatml(system: str, user: str) -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

def get_local_prompt(category: str, prompt: str) -> str:
    system_prompt = SYSTEM_PROMPTS.get(category, "Answer the query.")
    user_prompt = prompt + "\n\nAnswer only. No explanation, no chain-of-thought, no preamble, no restating the question."
    return format_chatml(system_prompt, user_prompt)

def start_local_server() -> subprocess.Popen:
    global local_disabled
        
    # Path to winget-installed llama-server on Windows
    win_bin = r"C:\Users\cheer\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
    binary = win_bin if os.path.exists(win_bin) else "llama-server"
    
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Local GGUF model file not found at {MODEL_PATH}. Operating in remote-only mode.")
        local_disabled = True
        return None
        
    try:
        logger.info(f"Launching local llama-server from: {binary} using {MODEL_PATH}...")
        proc = subprocess.Popen(
            [binary, "-m", MODEL_PATH, "--port", "8085", "-c", "1024", "-t", "4"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Poll health endpoint: http://127.0.0.1:8085/health
        health_url = "http://127.0.0.1:8085/health"
        for i in range(30):
            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=1.0) as response:
                    if response.status == 200:
                        logger.info("Local llama-server started successfully and is healthy!")
                        local_disabled = False
                        return proc
            except Exception:
                pass
            time.sleep(0.5)
            
        logger.warning("Local llama-server failed to report healthy in 15 seconds. Terminating process.")
        proc.terminate()
        local_disabled = True
        return None
    except Exception as e:
        logger.error(f"Failed to launch local llama-server process: {e}")
        local_disabled = True
        return None

async def call_local_model(system_prompt: str, user_prompt: str, max_tokens: int = 150) -> str:
    url = "http://127.0.0.1:8085/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload, timeout=15.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    raise ValueError(f"HTTP Error {resp.status} from local llama-server")
        except Exception as e:
            logger.error(f"HTTP request to local llama-server failed: {e}")
            raise e

async def verify_local_answer(category: str, prompt: str, answer: str) -> bool:
    """
    Runs a zero-cost local self-verification check to evaluate confidence.
    """
    global local_disabled
    if local_disabled:
        return True
        
    try:
        if category == "named_entity_recognition":
            try:
                start = answer.find('{')
                end = answer.rfind('}')
                if start != -1 and end != -1:
                    data = json.loads(answer[start:end+1])
                    for entity in data.get("entities", []):
                        text = entity.get("text", "")
                        if text and text.lower() not in prompt.lower():
                            logger.warning(f"NER Verification: Extracted entity '{text}' not found in source text.")
                            return False
            except Exception:
                return False
                
        return True
    except Exception as e:
        logger.warning(f"Self-verification helper encountered error: {e}")
        return True

def sanitize_response(text: str, category: str) -> str:
    """
    Strip chain-of-thought preamble and trailing noise from model outputs.
    Code models sometimes expose their reasoning when used for non-code tasks.
    Applies for sentiment/factual/summarization/logic categories.
    """
    if not text:
        return text
    lines = text.strip().splitlines()
    # Strip lines that start with "The user wants..." or "The user asks..." or self-narration
    skip_prefixes = (
        "the user wants", "the user asks", "the user said",
        "let me", "i need to", "i will", "i should", "i must",
        "thinking:", "thought:", "analysis:", "reasoning:",
        "step 1", "step 2", "step 3",
    )
    cleaned = []
    for line in lines:
        if line.strip().lower().startswith(skip_prefixes):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    # For sentiment: if the response has "But the" or trailing clause cuts off mid-sentence, truncate
    if category == "sentiment_classification" and result:
        # Take only the first 2 sentences max
        sentences = re.split(r'(?<=[.!?])\s+', result)
        result = " ".join(sentences[:2]).strip()
    return result if result else text

def classify_model_roles(allowed_models):
    """
    Parses allowed_models into code, reasoning, cheap, and mid roles.
    """
    code_model = None
    reasoning_model = None
    cheap_model = None
    mid_model = None
    
    # code role: contains "code" case-insensitively
    code_models = [m for m in allowed_models if "code" in m.lower()]
    if code_models:
        code_model = code_models[0]
        
    non_code_models = [m for m in allowed_models if m not in code_models]
    if not non_code_models:
        non_code_models = allowed_models.copy()
        
    # Heuristics-based parameter size parsing helper
    def get_param_size(model_name):
        name_lower = model_name.lower()
        size = 0.0
        
        # 1. Large proprietary/commercial models prioritized as reasoning models
        if "minimax" in name_lower:
            size = 500.0
        else:
            # Base parameter size extraction from patterns like -8b, -70b, -405b, -1.5b
            size_match = re.search(r"[-_](\d+(?:\.\d+|p\d+)?)b", name_lower)
            if size_match:
                size_str = size_match.group(1).replace("p", ".")
                try:
                    size = float(size_str)
                except:
                    pass
            else:
                # Default medium parameter size if no size pattern exists
                size = 10.0
                
        # 2. Adjust size based on quantization suffix to ensure proper cheap/mid ranking
        # a4b is extremely cheap/small
        if "a4b" in name_lower:
            size = size * 0.1 if size > 0 else 1.0
        # nvfp4, fp4, int4, int8 are quantized and cheaper than base counterparts
        elif any(q in name_lower for q in ["fp4", "nvfp4", "int4", "int8", "quant"]):
            size = size * 0.5 if size > 0 else 5.0
            
        return size

    non_code_sorted = sorted(non_code_models, key=get_param_size)
    
    if non_code_sorted:
        cheap_model = non_code_sorted[0]
        reasoning_model = non_code_sorted[-1]
        
    if len(non_code_sorted) > 2:
        mid_model = non_code_sorted[1]
    elif len(non_code_sorted) == 2:
        mid_model = non_code_sorted[1]
    else:
        mid_model = cheap_model
        
    if not code_model:
        code_model = reasoning_model or cheap_model
        
    return {
        "code": code_model,
        "reasoning": reasoning_model,
        "cheap": cheap_model,
        "mid": mid_model
    }

def write_output_results(results_map, output_path):
    """
    Writes the current state of results atomically.
    """
    try:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temp_path = output_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(list(results_map.values()), f, indent=2)
        os.replace(temp_path, output_path)
    except Exception as e:
        logger.error(f"Failed to write results to output path: {e}")

async def execute_task_pipeline(task_id, prompt, roles, client, local_sem, remote_sem):
    """
    The full cascade routing implementation.
    """
    global local_disabled
    
    # 1. Classification
    local_callable = None
    if not local_disabled:
        local_callable = lambda p, max_tokens=15: call_local_model("You are a helpful classification assistant.", p, max_tokens)
        
    category = await classify_prompt(prompt, local_llm_callable=local_callable)
    
    easy_categories = [
        "factual_knowledge", 
        "sentiment_classification", 
        "summarization", 
        "named_entity_recognition"
    ]
    
    tier_used = "unknown"
    model_name = "local"
    answer = ""
    validation_pass = False
    start_time = time.time()
    
    # 2. Local Tier (Easy categories)
    if category in easy_categories and not local_disabled:
        async with local_sem:
            tier_used = "local"
            try:
                system_prompt = SYSTEM_PROMPTS.get(category, "Answer the query.")
                user_prompt = prompt + "\n\nAnswer only. No explanation, no chain-of-thought, no preamble, no restating the question."
                max_tokens = min(35 if category != "named_entity_recognition" else 70, get_max_tokens(category, prompt))
                answer = await asyncio.wait_for(
                    call_local_model(system_prompt, user_prompt, max_tokens),
                    timeout=40.0
                )
                validation_pass = validate_category_output(category, prompt, answer)
                if validation_pass:
                    verification_pass = await verify_local_answer(category, prompt, answer)
                    if verification_pass:
                        logger.info(f"Task {task_id}: Local model passed validation and self-verification.")
                    else:
                        logger.info(f"Task {task_id}: Local model self-verification failed. Escalating...")
                        validation_pass = False
                else:
                    logger.info(f"Task {task_id}: Local model failed structural validation. Escalating to cheap remote...")
            except asyncio.TimeoutError:
                logger.warning(f"Task {task_id}: Local model execution timed out. Escalating...")
            except Exception as e:
                logger.warning(f"Task {task_id}: Local model execution failed: {e}. Escalating...")
            
    # 3. Remote Tier Cascade (Easy categories on escalation or local disabled)
    if category in easy_categories and not validation_pass:
        async with remote_sem:
            # Step 3a: Cheap remote role
            if roles.get("cheap"):
                tier_used = "cheap-remote"
                model_name = roles["cheap"]
                try:
                    answer = await client.call_api(
                        model=roles["cheap"],
                        category=category,
                        prompt=prompt,
                        timeout=9.0
                    )
                    validation_pass = validate_category_output(category, prompt, answer)
                    if validation_pass:
                        logger.info(f"Task {task_id}: Cheap remote model passed validation.")
                except Exception as e:
                    logger.warning(f"Task {task_id}: Cheap remote model failed: {e}. Escalating...")
                
            # Step 3b: Mid remote role (escalate once)
            if not validation_pass and roles.get("mid"):
                tier_used = "mid-remote"
                model_name = roles["mid"]
                try:
                    answer = await client.call_api(
                        model=roles["mid"],
                        category=category,
                        prompt=prompt,
                        timeout=9.0
                    )
                    validation_pass = validate_category_output(category, prompt, answer)
                    if validation_pass:
                        logger.info(f"Task {task_id}: Mid remote model passed validation.")
                except Exception as e:
                    logger.warning(f"Task {task_id}: Mid remote model failed: {e}. Escalating...")
                    
            # Fallback 1: Reasoning remote role
            if not validation_pass and roles.get("reasoning"):
                tier_used = "reasoning-fallback"
                model_name = roles["reasoning"]
                try:
                    logger.info(f"Task {task_id}: Easy category falling back to reasoning model: {model_name}")
                    answer = await client.call_api(
                        model=roles["reasoning"],
                        category=category,
                        prompt=prompt,
                        timeout=12.0
                    )
                    validation_pass = validate_category_output(category, prompt, answer)
                    if validation_pass:
                        logger.info(f"Task {task_id}: Easy category reasoning fallback passed validation.")
                except Exception as e:
                    logger.warning(f"Task {task_id}: Easy category reasoning fallback failed: {e}. Escalating...")

            # Fallback 2: Code remote role
            if not validation_pass and roles.get("code"):
                tier_used = "code-fallback"
                model_name = roles["code"]
                try:
                    logger.info(f"Task {task_id}: Easy category falling back to code model: {model_name}")
                    answer = await client.call_api(
                        model=roles["code"],
                        category=category,
                        prompt=prompt,
                        timeout=12.0
                    )
                    validation_pass = validate_category_output(category, prompt, answer)
                    if validation_pass:
                        logger.info(f"Task {task_id}: Easy category code fallback passed validation.")
                except Exception as e:
                    logger.warning(f"Task {task_id}: Easy category code fallback failed: {e}")

            # Final Emergency Fallback
            if not validation_pass:
                logger.warning(f"Task {task_id}: All remote models failed for Easy Category. Falling back to emergency placeholder.")
                answer = get_emergency_fallback(category, prompt)
                        
    # 4. Direct-to-remote categories
    if category not in easy_categories:
        async with remote_sem:
            validation_pass = False
            if category in ["math_reasoning", "logical_reasoning"]:
                # Try reasoning model first
                if roles.get("reasoning"):
                    tier_used = "direct-remote"
                    model_name = roles["reasoning"]
                    try:
                        answer = await client.call_api(
                            model=roles["reasoning"],
                            category=category,
                            prompt=prompt,
                            timeout=14.0
                        )
                        validation_pass = validate_category_output(category, prompt, answer)
                        if validation_pass:
                            logger.info(f"Task {task_id}: Reasoning model passed validation.")
                    except Exception as e:
                        logger.warning(f"Task {task_id}: Reasoning model failed: {e}. Escalating...")
                
                # Fallback 1: mid model
                if not validation_pass and roles.get("mid"):
                    tier_used = "mid-fallback"
                    model_name = roles["mid"]
                    try:
                        logger.info(f"Task {task_id}: Math/Logic falling back to mid model: {model_name}")
                        answer = await client.call_api(
                            model=roles["mid"],
                            category=category,
                            prompt=prompt,
                            timeout=12.0
                        )
                        validation_pass = validate_category_output(category, prompt, answer)
                        if validation_pass:
                            logger.info(f"Task {task_id}: Math/Logic mid fallback model passed validation.")
                    except Exception as e:
                        logger.warning(f"Task {task_id}: Math/Logic mid fallback failed: {e}. Escalating...")

                # Fallback 2: cheap model
                if not validation_pass and roles.get("cheap"):
                    tier_used = "cheap-fallback"
                    model_name = roles["cheap"]
                    try:
                        logger.info(f"Task {task_id}: Math/Logic falling back to cheap model: {model_name}")
                        answer = await client.call_api(
                            model=roles["cheap"],
                            category=category,
                            prompt=prompt,
                            timeout=10.0
                        )
                        validation_pass = validate_category_output(category, prompt, answer)
                        if validation_pass:
                            logger.info(f"Task {task_id}: Math/Logic cheap fallback model passed validation.")
                    except Exception as e:
                        logger.warning(f"Task {task_id}: Math/Logic cheap fallback failed: {e}. Escalating...")

                # Fallback 3: code model
                if not validation_pass and roles.get("code"):
                    tier_used = "code-fallback"
                    model_name = roles["code"]
                    try:
                        logger.info(f"Task {task_id}: Math/Logic falling back to code model: {model_name}")
                        answer = await client.call_api(
                            model=roles["code"],
                            category=category,
                            prompt=prompt,
                            timeout=12.0
                        )
                        validation_pass = validate_category_output(category, prompt, answer)
                        if validation_pass:
                            logger.info(f"Task {task_id}: Math/Logic code fallback model passed validation.")
                    except Exception as e:
                        logger.warning(f"Task {task_id}: Math/Logic code fallback failed: {e}")

                # Final Emergency Fallback
                if not validation_pass:
                    logger.warning(f"Task {task_id}: All models failed for Math/Logic. Falling back to emergency placeholder.")
                    answer = get_emergency_fallback(category, prompt)
                            
            elif category in ["code_debugging", "code_generation"]:
                # Try code model first
                if roles.get("code"):
                    tier_used = "direct-remote"
                    model_name = roles["code"]
                    try:
                        answer = await client.call_api(
                            model=roles["code"],
                            category=category,
                            prompt=prompt,
                            timeout=14.0
                        )
                        validation_pass = validate_category_output(category, prompt, answer)
                        if validation_pass:
                            logger.info(f"Task {task_id}: Code model passed validation.")
                    except Exception as e:
                        logger.warning(f"Task {task_id}: Code model failed: {e}. Escalating...")

                # Fallback 1: mid model
                if not validation_pass and roles.get("mid"):
                    tier_used = "mid-fallback"
                    model_name = roles["mid"]
                    try:
                        logger.info(f"Task {task_id}: Code falling back to mid model: {model_name}")
                        answer = await client.call_api(
                            model=roles["mid"],
                            category=category,
                            prompt=prompt,
                            timeout=12.0
                        )
                        validation_pass = validate_category_output(category, prompt, answer)
                        if validation_pass:
                            logger.info(f"Task {task_id}: Code mid fallback model passed validation.")
                    except Exception as e:
                        logger.warning(f"Task {task_id}: Code mid fallback failed: {e}. Escalating...")

                # Fallback 2: cheap model
                if not validation_pass and roles.get("cheap"):
                    tier_used = "cheap-fallback"
                    model_name = roles["cheap"]
                    try:
                        logger.info(f"Task {task_id}: Code falling back to cheap model: {model_name}")
                        answer = await client.call_api(
                            model=roles["cheap"],
                            category=category,
                            prompt=prompt,
                            timeout=10.0
                        )
                        validation_pass = validate_category_output(category, prompt, answer)
                        if validation_pass:
                            logger.info(f"Task {task_id}: Code cheap fallback model passed validation.")
                    except Exception as e:
                        logger.warning(f"Task {task_id}: Code cheap fallback failed: {e}. Escalating...")

                # Fallback 3: reasoning model
                if not validation_pass and roles.get("reasoning"):
                    tier_used = "reasoning-fallback"
                    model_name = roles["reasoning"]
                    try:
                        logger.info(f"Task {task_id}: Code falling back to reasoning model: {model_name}")
                        answer = await client.call_api(
                            model=roles["reasoning"],
                            category=category,
                            prompt=prompt,
                            timeout=12.0
                        )
                        validation_pass = validate_category_output(category, prompt, answer)
                        if validation_pass:
                            logger.info(f"Task {task_id}: Code reasoning fallback model passed validation.")
                    except Exception as e:
                        logger.warning(f"Task {task_id}: Code reasoning fallback failed: {e}")

                # Final Emergency Fallback
                if not validation_pass:
                    logger.warning(f"Task {task_id}: All models failed for Code. Falling back to emergency placeholder.")
                    answer = get_emergency_fallback(category, prompt)
                            
    # Log information to stdout only
    latency = time.time() - start_time
    approx_tokens = len(answer) // 4
    print(f"TASK_LOG: task_id={task_id} | category={category} | tier={tier_used} | "
          f"model={model_name} | approx_tokens={approx_tokens} | "
          f"validation={'PASS' if validation_pass else 'FAIL'} | latency={latency:.2f}s", flush=True)
          
    return {"task_id": task_id, "answer": answer}

async def process_single_task(task, roles, client, results_map, output_path, local_sem, remote_sem):
    task_id = task["task_id"]
    prompt = task["prompt"]
    
    if not prompt or not prompt.strip():
        logger.warning(f"Task {task_id}: Empty prompt.")
        results_map[task_id] = {"task_id": task_id, "answer": "No content provided."}
        write_output_results(results_map, output_path)
        return
        
    try:
        res = await execute_task_pipeline(task_id, prompt, roles, client, local_sem, remote_sem)
        results_map[task_id] = res
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        results_map[task_id] = {"task_id": task_id, "answer": get_emergency_fallback("factual_knowledge", prompt)}
    finally:
        # Update output file on every task completion
        write_output_results(results_map, output_path)

async def main():
    # Read environment variables
    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL")
    allowed_models_env = os.environ.get("ALLOWED_MODELS")
    
    if not api_key or not base_url or not allowed_models_env:
        logger.error("Missing required environment variables: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS")
        sys.exit(1)
        
    allowed_models = [m.strip() for m in allowed_models_env.split(",") if m.strip()]
    
    # 1. Parse and assign roles dynamically
    roles = classify_model_roles(allowed_models)
    
    # Log configuration table
    print("=== CONFIGURATION TABLE ===")
    for role, model in roles.items():
        print(f"Role '{role}': {model}")
    print("===========================")
    
    # 2. Setup client
    client = FireworksClient(api_key=api_key, base_url=base_url)
    
    # Run startup model connectivity check — prune dead models from the role assignment
    logger.info("Starting Fireworks connectivity healthcheck for all allowed models...")
    live_models = []
    dead_models = []
    for model_name in allowed_models:
        try:
            logger.info(f"Healthcheck: sending test ping to {model_name}...")
            await client.call_api(
                model=model_name,
                category="factual_knowledge",
                prompt="hello",
                timeout=4.0
            )
            logger.info(f"Healthcheck for '{model_name}': SUCCESS (200)")
            live_models.append(model_name)
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "NOT_FOUND" in err_str or "not found" in err_str.lower():
                logger.warning(f"Healthcheck for '{model_name}': DEAD (404 — removing from cascade)")
                dead_models.append(model_name)
            else:
                # Non-404 failure (rate limit, timeout) — keep the model, may recover
                logger.warning(f"Healthcheck for '{model_name}': SOFT_FAIL (keeping in cascade) — {e}")
                live_models.append(model_name)
    
    if dead_models:
        logger.warning(f"Pruned {len(dead_models)} dead models from cascade: {dead_models}")
    
    # Re-classify roles using only live models
    if live_models:
        roles = classify_model_roles(live_models)
        print("=== UPDATED ROLE TABLE (live models only) ===")
        for role, model in roles.items():
            print(f"Role '{role}': {model}")
        print("=============================================")
    else:
        logger.error("All models failed healthcheck — cannot process tasks.")

            
    # Set up paths
    input_path = "/input/tasks.json"
    output_path = "/output/results.json"
    
    # Check for local test override paths
    if not os.path.exists("/input") or not os.path.exists("/output"):
        input_path = "./input/tasks.json"
        output_path = "./output/results.json"
        os.makedirs("./input", exist_ok=True)
        os.makedirs("./output", exist_ok=True)
        
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at {input_path}")
        sys.exit(1)
        
    with open(input_path, "r") as f:
        tasks = json.load(f)
        
    if not isinstance(tasks, list):
        logger.error("Input file must be a JSON array of tasks.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(tasks)} tasks to process.")
    
    # 2. Initialize results map with placeholders for fallback safety
    results_map = {
        task["task_id"]: {
            "task_id": task["task_id"], 
            "answer": "System interrupted before task could complete."
        } for task in tasks
    }
    
    # Initial write to guarantee valid JSON file at any moment
    write_output_results(results_map, output_path)
    
    # 4. Start local server (if GGUF model is present)
    proc = start_local_server()
    
    # 5. Run tasks concurrently
    max_local_concurrency = int(os.environ.get("MAX_LOCAL_CONCURRENCY", "3"))
    max_remote_concurrency = int(os.environ.get("MAX_REMOTE_CONCURRENCY", "12"))
    
    logger.info(f"Using split concurrency limits: local CPU: {max_local_concurrency}, remote I/O: {max_remote_concurrency}")
    
    local_sem = asyncio.Semaphore(max_local_concurrency)
    remote_sem = asyncio.Semaphore(max_remote_concurrency)
    
    async def worker(task):
        await process_single_task(task, roles, client, results_map, output_path, local_sem, remote_sem)
            
    try:
        await asyncio.gather(*(worker(task) for task in tasks))
    finally:
        # Wrap final write in try/finally to flush results on interruption
        write_output_results(results_map, output_path)
        if proc:
            logger.info("Stopping local llama-server process...")
            try:
                proc.terminate()
                proc.wait(timeout=5.0)
            except Exception as ex:
                logger.warning(f"Error terminating local llama-server: {ex}")
                try:
                    proc.kill()
                except:
                    pass
        logger.info(f"Finished. Results written to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
