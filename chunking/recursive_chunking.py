from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_chunking(document, chunk_size=1200, overlap=150, is_eng=False):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap, # overlap chunk to mitigate loss of info
        length_function=len
    )

    chunks = text_splitter.split_text(document)
    return chunks