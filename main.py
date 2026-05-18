from ragas import evaluate 
from ragas.metrics import context_precision, context_recall, faithfulness
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from time import time 
import os 
import chromadb

from testset.questions_set import questions
from funcs.utils import read_clean_doc, retrieve_top_k
from chunking.fixed_size_chunking import fixed_size_chunking
from libs.MyEmbeddingFunction import MyEmbeddingFunction

OLLAMA_CLOUD_URL = "https://ollama.com"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_BASE_MODEL = "gpt-oss:120b-cloud"
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
OLLAMA_API_KEY = "c036e8dbcabc49e28bcdeb7ca52cb800.IIPzB42NuCYS0gXLELXKzjtz"


# env setup
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

ollama_ef = MyEmbeddingFunction(model=OLLAMA_EMBEDDING_MODEL)

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="chunked_document", embedding_function=ollama_ef)


# start chunking
document = read_clean_doc('documents/CELEX_32006L0054_EN_TXT.pdf') # todo: pulire documento

start_time = time()

chunks = fixed_size_chunking(document, is_eng=False)



collection.add(
    ids = [f"id{i}" for i in range(1, len(chunks) + 1)],
    documents = chunks
)

collection_data = collection.get()
ids = collection_data.get("ids", [])
documents = collection_data.get("documents", [])

for i in range(len(documents)):
    print(f"ID: {ids[i]}")
    print(f"Content: {documents[i]}")
    print("-" * 50)

for q in questions:
    q["contexts"] = retrieve_top_k(collection, q["question"])
    #q["contexts"].append(retrieved)

    provided_context = q["contexts"]

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

with open('outputs/fixed_size_chunking.txt', 'w') as output:
    for chunk in chunks:
        output.write('chunk ' + str(i) + ' : ' + chunk  + '\n')
        i+= 1

file_size = os.path.getsize('outputs/fixed_size_chunking.txt')

print(result)
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print("File size: " + str(file_size))