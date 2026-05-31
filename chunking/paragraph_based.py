def paragraph_chunking(document, is_eng=False):

    paragraphs = document.split('\n\n')

    chunks = []

    # Filter out empty lines or tiny leftover Markdown artifacts (like "---" or single letters)
    for p in paragraphs:
        cleaned_parag = p.strip()
        
        if len(cleaned_parag) > 15:
            chunks.append(cleaned_parag)

    return chunks