import pymupdf4llm
import chromadb

def read_clean_doc(file_path):
    cleaned_text = pymupdf4llm.to_markdown(file_path)
    return cleaned_text

''' this method simulates retrieval from a vector database
1. converts texts into vectors
2. converts question into a numerical representation
3. cosine_similarity computes the similarity betweeb query and each chunks
4. retrieve the top k  chunks 
returns the best chunks
'''
# use for other chunking except subdoc and pageindex
# per questo tipo di documenti k = 3 è quella ideale
def retrieve_top_k(collection, queries, k=3):
  results = collection.query(
     query_texts=[queries],
     n_results = k
  )
  return results["documents"]