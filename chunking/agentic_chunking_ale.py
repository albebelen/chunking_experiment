from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_CLOUD_URL = "https://ollama.com"
OLLAMA_BASE_MODEL = "gpt-oss:120b-cloud"
OLLAMA_API_KEY = "0ab2a4e81a764cdca9ea9e9ec8349b76.A4xqN-KsGKNnDQ2iqQAJyuj5"

def agentic_chunking(document, model, is_eng=False):  
    llm = Ollama(
    model=OLLAMA_BASE_MODEL,
    base_url=OLLAMA_CLOUD_URL,
    temperature=0,
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)

    mini_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    mini_chunks = mini_splitter.split_text(document)
    
    final_chunks = []
    current_chunk = mini_chunks[0]


    for i in range(1, len(mini_chunks)):
        next_mini = mini_chunks[i]
        
        prompt = (
            f"Current Content: {current_chunk}\n\n"
            f"Next Sentence: {next_mini}\n\n"
            "Does the 'Next Sentence' continue the exact same legal topic or Article as the 'Current Content'? "
            "Respond with 'YES' to merge them or 'NO' to start a new chunk. Answer ONLY with YES or NO."
        )
        
        response = llm.invoke(prompt)
        
        decision = response.strip()
        
        if "YES" in decision:
            current_chunk += " " + next_mini
        else:
            final_chunks.append(current_chunk)
            current_chunk = next_mini

    final_chunks.append(current_chunk)
    return final_chunks