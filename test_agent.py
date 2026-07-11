import unittest
import json
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Configure env vars for importing main
os.environ["FIREWORKS_API_KEY"] = "fake_key"
os.environ["FIREWORKS_BASE_URL"] = "https://api.fireworks.ai/inference/v1"
os.environ["ALLOWED_MODELS"] = "accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/llama-v3p1-70b-instruct,accounts/fireworks/models/qwen2p5-coder-32b-instruct"

from classifier import classify_prompt
from validators import validate_category_output, coerce_ner_output
from client import FireworksClient, get_emergency_fallback
from deterministic_solvers import solve_ner_deterministically
from main import classify_model_roles
import main

class TestGeneralPurposeAgent(unittest.IsolatedAsyncioTestCase):

    def test_role_classification(self):
        """
        Tests parsing allowed_models into appropriate roles.
        """
        allowed = [
            "accounts/fireworks/models/llama-v3p1-405b-instruct",
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/qwen2p5-coder-32b-instruct"
        ]
        roles = classify_model_roles(allowed)
        self.assertEqual(roles["code"], "accounts/fireworks/models/qwen2p5-coder-32b-instruct")
        self.assertEqual(roles["reasoning"], "accounts/fireworks/models/llama-v3p1-405b-instruct")
        self.assertEqual(roles["cheap"], "accounts/fireworks/models/llama-v3p1-8b-instruct")
        self.assertEqual(roles["mid"], "accounts/fireworks/models/llama-v3p1-70b-instruct")

    def test_role_classification_fallback(self):
        """
        Tests fallback behavior when only two models are present.
        """
        allowed = [
            "llama-v3p1-8b-instruct",
            "qwen2p5-coder-32b"
        ]
        roles = classify_model_roles(allowed)
        self.assertEqual(roles["code"], "qwen2p5-coder-32b")
        self.assertEqual(roles["cheap"], "llama-v3p1-8b-instruct")
        self.assertEqual(roles["reasoning"], "llama-v3p1-8b-instruct")
        self.assertEqual(roles["mid"], "llama-v3p1-8b-instruct")

    def test_role_classification_launch_day_fixture(self):
        """
        Tests parsing the exact launch day model list provided in feedback.
        """
        allowed = [
            "minimax-m3", 
            "kimi-k2p7-code", 
            "gemma-4-31b-it", 
            "gemma-4-26b-a4b-it", 
            "gemma-4-31b-it-nvfp4"
        ]
        roles = classify_model_roles(allowed)
        self.assertEqual(roles["code"], "kimi-k2p7-code")
        self.assertEqual(roles["cheap"], "gemma-4-26b-a4b-it")
        self.assertEqual(roles["reasoning"], "minimax-m3")
        self.assertEqual(roles["mid"], "gemma-4-31b-it-nvfp4")

    async def test_task_classification_rules(self):
        """
        Tests that keyword-based classifier maps typical prompts correctly.
        """
        # Code Generation
        self.assertEqual(
            await classify_prompt("Write a python function to find the fibonacci sequence."),
            "code_generation"
        )
        # Code Debugging
        self.assertEqual(
            await classify_prompt("Find the bug in this function:\n```python\ndef f(x): return x/0\n```"),
            "code_debugging"
        )
        # Math Reasoning
        self.assertEqual(
            await classify_prompt("Solve for x: 3x + 5 = 20"),
            "math_reasoning"
        )
        # Logical Reasoning
        self.assertEqual(
            await classify_prompt("Which state of the grid satisfies the puzzle constraints?"),
            "logical_reasoning"
        )
        # Sentiment
        self.assertEqual(
            await classify_prompt("Classify the sentiment of this review as positive or negative."),
            "sentiment_classification"
        )
        # Summarization
        self.assertEqual(
            await classify_prompt("Summarise the main points of this article in 50 words."),
            "summarization"
        )
        # NER
        self.assertEqual(
            await classify_prompt("Extract all the named entities and locations from this paragraph."),
            "named_entity_recognition"
        )
        # Factual Knowledge (Default fallback)
        self.assertEqual(
            await classify_prompt("What is the capital city of France?"),
            "factual_knowledge"
        )

    async def test_factual_routing_classification(self):
        """
        Tests 5+ distinct factual-knowledge sample prompts to confirm they route to
        factual_knowledge, not sentiment_classification.
        """
        factual_prompts = [
            "What is the capital of Japan?",
            "Who was the first president of the United States?",
            "When did the Titanic sink?",
            "Where is Mount Everest located?",
            "How many planets are in the Solar System?",
            "Explain the theory of general relativity in simple terms."
        ]
        for prompt in factual_prompts:
            category = await classify_prompt(prompt)
            self.assertEqual(category, "factual_knowledge")

    async def test_math_routing_edge_cases(self):
        """Conversational / arithmetic prompts must route to math_reasoning, not factual."""
        math_prompts = [
            "What is -15 + 27?",
            "What is the average of 12, 18, and 30?",
            "Hey, if someone bought twelve apples and gave away four, how many would they have left?",
            "Tom has 3 apples, buys 12 more, then eats 7. How many apples does he have?",
            "There are 17 students and each needs 4 handouts. How many handouts should I print?",
        ]
        for prompt in math_prompts:
            self.assertEqual(await classify_prompt(prompt), "math_reasoning")

    async def test_conversational_ner_routing(self):
        prompts = [
            "From this blurb, pull out the people, companies, and places: Sundar Pichai leads Google.",
            "Can you extract named entities from this customer note? Dr. Anya Sharma visited Mayo Clinic.",
            "List the people and companies in: Satya Nadella leads Microsoft.",
        ]
        for prompt in prompts:
            self.assertEqual(await classify_prompt(prompt), "named_entity_recognition")

    async def test_summarization_not_math_with_percent(self):
        prompt = (
            "Summarize for a busy manager (under 20 words): Our Q3 revenue rose 8% "
            "year-over-year driven by enterprise subscriptions, while consumer churn ticked up slightly."
        )
        self.assertEqual(await classify_prompt(prompt), "summarization")

    async def test_probability_coin_is_logic_not_math(self):
        prompt = (
            "You flip a fair coin three times and get heads each time. "
            "What is the probability the next flip is heads? Reply with a fraction."
        )
        self.assertEqual(await classify_prompt(prompt), "logical_reasoning")

    async def test_edge_cases_classification(self):
        """
        Tests weird unicode prompts, super long inputs, prompts with no clear category,
        and multiple languages to ensure robust classification behavior.
        """
        # Weird Unicode and emojis
        unicode_prompt = "🌟 Translate and extract named entities: Mr. Takahashi works at Sony in Tokyo 🇯🇵."
        self.assertEqual(await classify_prompt(unicode_prompt), "named_entity_recognition")

        # Super long input (5000+ characters)
        long_prompt = "summarize " + ("hello " * 1000)
        self.assertEqual(await classify_prompt(long_prompt), "summarization")

        # Prompts with no clear category (should fallback safely)
        no_clear_prompt = "xyz123abc !!!"
        self.assertIn(await classify_prompt(no_clear_prompt), [
            "factual_knowledge", "sentiment_classification", "summarization",
            "named_entity_recognition", "code_debugging", "logical_reasoning",
            "code_generation", "math_reasoning"
        ])

        # Multiple languages (Spanish and French queries)
        spanish_factual = "Cual es la capital de España?"
        self.assertEqual(await classify_prompt(spanish_factual), "factual_knowledge")

        french_factual = "Quelle est la capitale de la France?"
        self.assertEqual(await classify_prompt(french_factual), "factual_knowledge")

    def test_ner_validator(self):
        """
        Tests that NER validator accepts valid schemas and rejects invalid ones.
        """
        valid_ner = '{"entities": [{"text": "Google", "type": "ORG"}, {"text": "London", "type": "LOCATION"}]}'
        invalid_ner_json = '{"entities": [{"text": "Google"}]}' # Missing type
        invalid_ner_key = '{"names": [{"text": "Google", "type": "ORG"}]}' # Wrong key
        invalid_ner_structure = '{"entities": "Google"}'
        
        self.assertTrue(validate_category_output("named_entity_recognition", "", valid_ner))
        self.assertFalse(validate_category_output("named_entity_recognition", "", invalid_ner_json))
        self.assertFalse(validate_category_output("named_entity_recognition", "", invalid_ner_key))
        self.assertFalse(validate_category_output("named_entity_recognition", "", invalid_ner_structure))
        self.assertFalse(validate_category_output("named_entity_recognition", "", "Not JSON"))

    def test_sentiment_validator(self):
        """
        Tests sentiment verification against expected labels.
        """
        prompt = "Classify this review (positive/negative/neutral): I loved it!"
        
        self.assertTrue(validate_category_output("sentiment_classification", prompt, "positive - because the user loved it."))
        self.assertTrue(validate_category_output("sentiment_classification", prompt, "The sentiment is negative because it was bad."))
        # Now rejected to force cheap model to produce justification
        self.assertFalse(validate_category_output("sentiment_classification", prompt, "positive"))

        # Justification requested
        prompt_with_just = "Determine the sentiment (positive/negative) and provide a justification."
        self.assertTrue(validate_category_output("sentiment_classification", prompt_with_just, "positive because it has excellent features"))
        # Now rejected to force cheap model to produce justification
        self.assertFalse(validate_category_output("sentiment_classification", prompt_with_just, "positive"))

        client = FireworksClient("fake", "https://example.invalid")
        normalized = client._scrub_cot(
            "Mixed — the design is excellent and the battery is poor.",
            "sentiment_classification",
        )
        self.assertIn("because", normalized.lower())
        self.assertTrue(validate_category_output("sentiment_classification", prompt, normalized))

    def test_ner_partial_answers_fall_through_and_heading_repair(self):
        # Missing event/product/date candidates must not be accepted as a
        # confident deterministic extraction.
        event_result = solve_ner_deterministically(
            "NER task — list entities with types from: Serena Williams won Wimbledon in London in 2012."
        )
        self.assertIsNotNone(event_result)
        event_entities = json.loads(event_result)["entities"]
        self.assertIn(
            {"text": "Wimbledon", "type": "EVENT"},
            event_entities,
        )
        self.assertIsNotNone(solve_ner_deterministically(
            "Can you extract named entities from this customer note? "
            "Dr. Anya Sharma at Mayo Clinic in Rochester prescribed Lisinopril on March 3, 2024."
        ))
        clinic_result = solve_ner_deterministically(
            "Can you extract named entities from this customer note? "
            "Dr. Anya Sharma at Mayo Clinic in Rochester prescribed Lisinopril on March 3, 2024."
        )
        clinic_entities = {e["text"] for e in json.loads(clinic_result)["entities"]}
        self.assertIn("Lisinopril", clinic_entities)
        self.assertIn("Mayo Clinic", clinic_entities)

        repaired, ok = coerce_ner_output(
            "**People:** Sundar Pichai\n"
            "**Companies:** Google, Alphabet\n"
            "**Places:** Mountain View"
        )
        self.assertTrue(ok)
        data = json.loads(repaired)
        self.assertEqual(
            {e["text"] for e in data["entities"]},
            {"Sundar Pichai", "Google", "Alphabet", "Mountain View"},
        )

    def test_code_debugging_bare_def_scrub(self):
        client = FireworksClient("k", "https://example.com")
        kimi_style = (
            "Here is the fixed binary search with empty array handling:\n\n"
            "def bsearch(a, x):\n"
            "    if not a:\n"
            "        return -1\n"
            "    lo, hi = 0, len(a)\n"
            "    while lo < hi:\n"
            "        mid = (lo+hi)//2\n"
            "        if a[mid] < x:\n"
            "            lo = mid+1\n"
            "        else:\n"
            "            hi = mid\n"
            "    return lo\n"
        )
        scrubbed = client._scrub_cot(kimi_style, "code_debugging")
        self.assertIn("```python", scrubbed)
        self.assertTrue(
            validate_category_output("code_debugging", "def bsearch(a, x): pass", scrubbed)
        )


    def test_logic_answer_prefix_stripped(self):
        client = FireworksClient("k", "https://example.com")
        self.assertEqual(client._scrub_cot("Answer: juice", "logical_reasoning"), "juice")
        self.assertEqual(client._scrub_cot("Answer: 1/2", "logical_reasoning"), "1/2")

    def test_code_debug_deterministic_not_used_in_pipeline(self):
        """Generic empty-input patch must not bypass remote code models."""
        from deterministic_solvers import solve_code_debug_deterministically
        prompt = (
            "Fix off-by-one error in sum:\n"
            "def total(nums):\n"
            "    s = 0\n"
            "    for i in range(len(nums)):\n"
            "        s += nums[i+1]\n"
            "    return s\n"
            "Also handle empty input."
        )
        # Solver may return a patch — pipeline must not call it (removed in v27).
        patch = solve_code_debug_deterministically(prompt)
        import main
        src = open(main.__file__, encoding="utf-8").read()
        self.assertNotIn("solve_code_debug_deterministically", src)

        """
        Tests summary length constraints.
        """
        prompt_limit = "Summarize in 5 words or less: the quick brown fox jumps over the lazy dog."
        self.assertTrue(validate_category_output("summarization", prompt_limit, "Quick brown fox jumps.")) # 4 words
        # 10 words (exceeds limit + buffer)
        self.assertFalse(validate_category_output("summarization", prompt_limit, "The quick brown fox jumps over the lazy sleeping dog today."))

    def test_factual_validator(self):
        """
        Tests factual response validation.
        """
        prompt = "What is gravity?"
        self.assertTrue(validate_category_output("factual_knowledge", prompt, "Gravity is a fundamental interaction."))
        self.assertFalse(validate_category_output("factual_knowledge", prompt, "What is gravity?")) # degenerate repetition

    def test_reasoning_validator(self):
        """
        Tests math/logic response structure validation.
        """
        self.assertTrue(validate_category_output("math_reasoning", "", "144"))
        self.assertTrue(validate_category_output("logical_reasoning", "", "The owner is Sam."))
        self.assertFalse(validate_category_output("math_reasoning", "", ""))

    def test_code_validator(self):
        """
        Tests syntax and formatting validation for code.
        """
        prompt = "Write a python function."
        valid_python = "```python\ndef add(a, b):\n    return a + b\n```"
        invalid_python = "```python\ndef add(a, b):\n    return a + \n```" # Syntax error
        no_blocks = "def add(a, b): return a + b"
        
        bare_valid_python = "def add(a, b): return a + b"       # valid bare Python — now accepted
        bare_invalid_python = "def add(a b): return a + b"       # syntax error even bare — still rejected

        self.assertTrue(validate_category_output("code_generation", prompt, valid_python))
        self.assertFalse(validate_category_output("code_generation", prompt, invalid_python))
        self.assertTrue(validate_category_output("code_generation", prompt, bare_valid_python))   # auto-wrapped
        self.assertFalse(validate_category_output("code_generation", prompt, bare_invalid_python)) # syntax error


class TestMainLoop(unittest.IsolatedAsyncioTestCase):

    async def test_atomic_writing_and_io(self):
        """
        Mocking client and running the loop to check that input is processed
        and output is correctly created containing every task ID.
        """
        input_data = [
            {"task_id": "t1", "prompt": "Identify sentiment: Happy!"},
            {"task_id": "t2", "prompt": "Solve math: 2+2=?"}
        ]
        
        # Write temporary input file
        os.makedirs("./input", exist_ok=True)
        with open("./input/test_tasks.json", "w") as f:
            json.dump(input_data, f)
            
        mock_client = AsyncMock()
        mock_client.call_api.return_value = "positive - because it is happy."
        
        roles = {
            "code": "model-code",
            "reasoning": "model-reasoning",
            "cheap": "model-cheap",
            "mid": "model-mid"
        }
        
        results_map = {
            t["task_id"]: {"task_id": t["task_id"], "answer": "System interrupted"} for t in input_data
        }
        
        output_path = "./output/results.json"
        
        local_sem = asyncio.Semaphore(3)
        remote_sem = asyncio.Semaphore(12)
        # Test task executor pipeline
        with patch('main.local_disabled', True): # Force remote only
            await main.process_single_task(input_data[0], roles, mock_client, results_map, output_path, local_sem, remote_sem)
            await main.process_single_task(input_data[1], roles, mock_client, results_map, output_path, local_sem, remote_sem)
            
        # Read written output
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "r") as f:
            output_data = json.load(f)
            
        self.assertEqual(len(output_data), 2)
        self.assertEqual(output_data[0]["task_id"], "t1")
        self.assertEqual(output_data[1]["task_id"], "t2")
        
        # Clean up files
        if os.path.exists("./input/test_tasks.json"):
            os.remove("./input/test_tasks.json")
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    unittest.main()
