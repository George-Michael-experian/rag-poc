def chunk_document(text, chunk_size=500, overlap= 50):
    ### Split document into overlapping chunks###
    chunks = []
    start = 0 
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        #try to break at sentence boundary 

        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
        
        chunks.append(chunk.strip())
        start = end - overlap # move back by over lap for next chunk
    print(chunks)
    return chunks 

chunk_document("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.", chunk_size=100, overlap=20)