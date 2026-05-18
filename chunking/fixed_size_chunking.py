import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

def fixed_size_chunking(raw_text, chunk_size=500, is_eng=False):   
    lang = "english" if is_eng else "italian"
 
    words = word_tokenize(raw_text, language=lang)
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    return [" ".join(chunk) for chunk in chunks]