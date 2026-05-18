from ragas import evaluate 
from ragas.metrics import context_precision, context_recall, faithfulness
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from time import time 
import os 

from funcs.utils import read_document, retrieve_top_k
from questions.questions_dataset import questions

from chunking.subdocument_chunking import subdocument_chunking

OLLAMA_CLOUD_URL = "https://ollama.com"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_BASE_MODEL = "gpt-oss:120b-cloud"
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
OLLAMA_API_KEY = "c036e8dbcabc49e28bcdeb7ca52cb800.IIPzB42NuCYS0gXLELXKzjtz"

llm = Ollama(
    model=OLLAMA_BASE_MODEL,
    base_url=OLLAMA_CLOUD_URL,
    temperature=0,
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)
embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBEDDING_MODEL, 
    base_url=OLLAMA_BASE_URL,
)

document = read_document('docs/CELEX_32006L0054_EN_TXT.pdf')

start_time = time()

chunks = subdocument_chunking(document, 'docs/CELEX_32006L0054_EN_TXT.pdf')

for q in questions:
    retrieved = retrieve_top_k(chunks, q["question"], embeddings, k=5)
    q["contexts"].append(retrieved)

    provided_context = "\n\n".join(retrieved)

    prompt = f"""
    Answer only based on provided context.

    Answer format: {q["answer_type"]}

    Context: {provided_context}

    Question: {q["question"]}
    """

    response = llm.invoke(prompt)

    if isinstance(response, str):
        q["answer"] = response.strip()
    else:
        q["answer"] = str(response).strip()

    print("Question: " + q["question"] + "\n")
    print("Answer: " + response + "\n")

stop_time = time()
elapsed_time = stop_time - start_time

dataset = Dataset.from_dict({
    "user_input": [q["question"] for q in questions],
    "question": [q["question"] for q in questions],
    "answer": [q["answer"] for q in questions],
    "ground_truth": [q["ground_truth"] for q in questions],
    "contexts": [[str(c) for c in q["contexts"]] for q in questions],
    "metadata": [{"difficulty": q["difficulty"], "adversarial": q["adversarial"], "language": "IT"} for q in questions]
})

result = evaluate(
    dataset,
    metrics=[context_precision, context_recall, faithfulness],
    llm=llm,
    embeddings=embeddings,
    raise_exceptions=False
)

i = 1

with open('outputs/subdocument_chunking_ts.txt', 'w') as output:
    for chunk in chunks:
        output.write('chunk ' + str(i) + ' : ' + chunk  + '\n')
        i+= 1

file_size = os.path.getsize('outputs/subdocument_chunking_ts.txt')

print(result)
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print("File size: " + str(file_size))