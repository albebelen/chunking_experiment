import pymupdf4llm

def read_clean_doc(file_path):
    cleaned_text = pymupdf4llm.to_markdown(file_path)
    return cleaned_text