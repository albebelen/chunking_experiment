import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

def sentence_chunking(document, sentences_per_chunk=4, overlap = 1, is_eng = False):
    lang = "english" if is_eng else "italian"

    sentences = sent_tokenize(document, language=lang)

    chunks = []
    step_size = sentences_per_chunk - overlap

    for i in range(0, len(sentences), step_size):
        chunk_group = sentences[i : i + sentences_per_chunk]
        chunk_text = " ".join(chunk_group)
        chunks.append(chunk_text)

        if i + sentences_per_chunk >= len(sentences):
            break
    
    return chunks