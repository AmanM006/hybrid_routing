import os
import json

def main():
    # 1. Create dummy input directory and tasks.json
    input_dir = "./input"
    os.makedirs(input_dir, exist_ok=True)
    tasks_path = os.path.join(input_dir, "tasks.json")
    
    test_tasks = []
    for i in range(1, 26):
        if i % 4 == 1:
            test_tasks.append({
                "task_id": f"task_ner_{i}",
                "prompt": f"Extract entities: John Doe works at TechCorp in Tokyo. Index {i}."
            })
        elif i % 4 == 2:
            test_tasks.append({
                "task_id": f"task_sentiment_{i}",
                "prompt": f"Classify sentiment (positive/negative/neutral): The food was excellent at restaurant {i}."
            })
        elif i % 4 == 3:
            test_tasks.append({
                "task_id": f"task_summary_{i}",
                "prompt": f"Summarize in 5 words or less: Machine learning is study of computer algorithms that improve automatically. Index {i}."
            })
        else:
            test_tasks.append({
                "task_id": f"task_factual_{i}",
                "prompt": f"What is the capital of France? Query number {i}."
            })
            
    with open(tasks_path, "w") as f:
        json.dump(test_tasks, f, indent=2)
    print(f"Created sample task file with 25 tasks at: {os.path.abspath(tasks_path)}")
    print("\nNext steps to run a live test:")
    print("1. Download the local model:  python download_model.py")
    print("2. Run the routing agent:     python main.py")
    print("3. Inspect the outputs:       ./output/results.json")

if __name__ == "__main__":
    main()
