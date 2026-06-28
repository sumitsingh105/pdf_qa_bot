import fitz

def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    documents = []

    for page_num, page in enumerate(doc):
        text = page.get_text()

        cleaned_text = text.strip()

        documents.append({
            "page": page_num + 1,
            "text": cleaned_text
        })

    return documents