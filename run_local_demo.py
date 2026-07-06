import os
import json

def main():
    # 1. Create dummy input directory and tasks.json
    input_dir = "./input"
    os.makedirs(input_dir, exist_ok=True)
    tasks_path = os.path.join(input_dir, "tasks.json")
    
    test_tasks = [
        {
            "task_id": "task_ner_1",
            "prompt": "Extract entities: Steve Jobs founded Apple Inc. in California."
        },
        {
            "task_id": "task_sentiment_1",
            "prompt": "Classify review sentiment (positive/negative/neutral) and justify: The customer service was exceptionally helpful and friendly!"
        },
        {
            "task_id": "task_summary_1",
            "prompt": "Summarize in 5 words or less: Deep learning is subset of machine learning using multi-layer neural networks."
        },
        {
            "task_id": "task_factual_1",
            "prompt": "What is the boiling point of water in Celsius?"
        },
        {
            "task_id": "task_math_1",
            "prompt": "Solve for y: 2y + 8 = 18. What is the value of y?"
        },
        {
            "task_id": "task_code_1",
            "prompt": "Write a python function to compute the factorial of a given integer."
        },
        {
            "task_id": "task_logic_1",
            "prompt": "If all A are B, and all B are C, are all A necessarily C? Explain logic."
        },
        {
            "task_id": "task_ner_2",
            "prompt": "Extract names and organizations: Elon Musk is the CEO of Tesla."
        },
        {
            "task_id": "task_sentiment_2",
            "prompt": "Classify review sentiment (positive/negative/neutral): This food tastes terrible."
        },
        {
            "task_id": "task_summary_2",
            "prompt": "Summarize in 8 words or less: Python is a high-level programming language known for its readability."
        },
        {
            "task_id": "task_factual_2",
            "prompt": "Who wrote the play Hamlet?"
        },
        {
            "task_id": "task_debug_1",
            "prompt": "Fix the syntax error in this code: print('Hello world)"
        }
    ]
    
    with open(tasks_path, "w") as f:
        json.dump(test_tasks, f, indent=2)
    print(f"Created sample task file at: {os.path.abspath(tasks_path)}")
    print("\nNext steps to run a live test:")
    print("1. Download the local model:  python download_model.py")
    print("2. Run the routing agent:     python main.py")
    print("3. Inspect the outputs:       ./output/results.json")

if __name__ == "__main__":
    main()
