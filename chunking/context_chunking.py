from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

def context_enriched_chunking(document, model, is_eng=False):
    OLLAMA_CLOUD_URL = "https://ollama.com"
    #OLLAMA_BASE_MODEL = "gpt-oss:120b-cloud"
    OLLAMA_API_KEY = "c036e8dbcabc49e28bcdeb7ca52cb800.IIPzB42NuCYS0gXLELXKzjtz"

    llm = Ollama(
        model=model,
        base_url=OLLAMA_CLOUD_URL,
        temperature=0,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    base_chunks = text_splitter.split_text(document)

    enriched_chunks = []

    for i, chunk in enumerate(base_chunks):
        prompt = (f"Here is a chunk from an EU Directive:\n\n{chunk}\n\n"
            "Provide a 1-sentence summary context to situate this chunk within the "
            "overall document to improve search retrieval with key terms or concept as metadata. "
            "Return ONLY the context and metadata.")

        response = llm.invoke(prompt)

        context_summary = response.strip()
        enriched_content = f"CONTEXT: {context_summary}\n\nContent: {chunk}"
        enriched_chunks.append(enriched_content)

    return enriched_chunks