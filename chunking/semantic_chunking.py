from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

def semantic_chunking(document, threshold, is_eng=False): 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold # ???
    )
    
    chunks = text_splitter.create_documents([document])    
    
    #return chunks[0].page_content
    # return [{
    #         "chunk_content": c.page_content,
    #         "metadata": {"source": "docs/CELEX_32006L0054_EN_TXT.pdf"}
    #     }
    #     for c in chunks]
    
    return [c.page_content for c in chunks ]