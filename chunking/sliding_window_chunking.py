def sliding_window_chunking(document, window_size = 150, overlap = 50, is_eng = False):
    words = document.split()
    
    chunks = []
    step_size = window_size - overlap

    for i in range(0, len(words), step_size):
        chunk_words = words[i:i+window_size]

        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        if i+window_size >= len(words):
            break

    return chunks