from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

OLLAMA_CLOUD_URL = "https://ollama.com"
OLLAMA_BASE_MODEL = "gpt-oss:120b-cloud"
OLLAMA_API_KEY = "c036e8dbcabc49e28bcdeb7ca52cb800.IIPzB42NuCYS0gXLELXKzjtz"

def agentic_chunk_block(llm, text_block):
    """The Agent: Asks Ollama to logically split a block of text."""
    
    # prompt = f"""
    # You are an expert legal document structuring agent.
    # Your task is to read the following text and split it into logical, self-contained chunks.
    # Each chunk should cover exactly one legal concept, recital, or article.
    
    # TEXT TO ANALYZE:
    # {text_block}
    
    # OUTPUT FORMAT:
    # You must respond ONLY with a valid JSON object. Do not add markdown formatting, do not say "Here is the JSON".
    # Strictly use this format:
    # {{
    #     "chunks": [
    #         "first logical chunk here",
    #         "second logical chunk here"
    #     ]
    # }}
    # """

    prompt = f"""
        Sei un agente esperto nella strutturazione di documenti legali.
        Il tuo compito è leggere il testo seguente e suddividerlo in blocchi (chunk) logici e autonomi.
        Ogni blocco deve riguardare esattamente un concetto legale, un considerando o un articolo.
        
        TESTO DA ANALIZZARE:
        {text_block}
        
        FORMATO DI OUTPUT:
        Devi rispondere ESCLUSIVAMENTE con un oggetto JSON valido. Non aggiungere formattazione Markdown, non scrivere frasi come "Ecco il JSON".
        Usa tassativamente questo formato:
        {{
            "chunks": [
                "primo blocco logico qui",
                "secondo blocco logico qui"
            ]
        }}
    """
    
    try:
        response = llm.invoke(prompt)
        # Parse the JSON string back into a Python dictionary
        data = json.loads(response)
        return data.get("chunks", [])
    except json.JSONDecodeError:
        print("LLM Agent failed. Falling back to original block.")
        return [text_block]    

def agentic_chunking(document, model, is_eng=False):  
    llm = Ollama(
        model=model,
        base_url=OLLAMA_CLOUD_URL,
        temperature=0,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    )

    pre_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    pre_chunks= pre_splitter.split_text(document)
    
    final_chunks = []
    
    for i, block in enumerate(pre_chunks, start=1):
        smart_chunks = agentic_chunk_block(llm, block)
        final_chunks.extend(smart_chunks)
    
    return final_chunks

