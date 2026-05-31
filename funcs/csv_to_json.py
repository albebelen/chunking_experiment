import pandas as pd
import json


testset = pd.read_csv("../testset/testset_generated_en.csv").fillna("")

questions = [] 

for idx, row in testset.iterrows():
    answer_type = str(row.get("query_length", "short")).lower()
    test_set = int(row.get("test_set", 1))

    generated_id = f"ts{test_set}_{answer_type}_{idx + 1:02d}"

    question_obj = {
        "id": generated_id,
        "test_set": 1,
        "answer_type": answer_type,
        "difficulty": "easy" if answer_type == "short" else "medium" if answer_type == "medium" else "hard",
        "question": row.get("user_input", ""),
        "answer": "",
        "ground_truth": row.get("reference", ""),
        "article_refs": [],
        "contexts": (
            row.get("reference_contexts", [])
            if isinstance(row.get("reference_contexts", []), list)
            else [row.get("reference_contexts", "")]
        ),
        "adversarial": False # for now
    }

    questions.append(question_obj)

with open("../testset/en_testset.py", "w", encoding="utf-8") as f:
    f.write("questions = ")
    f.write(json.dumps(questions, indent=4, ensure_ascii=False))