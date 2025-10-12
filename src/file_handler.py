import streamlit as st
from PyPDF2 import PdfReader
from docx import Document

def get_text_from_files(uploaded_files):
    """
    Reads and extracts text from a list of uploaded files (PDF, DOCX, TXT).

    Args:
        uploaded_files (list): A list of files uploaded via Streamlit's file_uploader.

    Returns:
        str: A single string containing all the extracted text from the documents.
    """
    text = ""
    for file in uploaded_files:
        try:
            if file.name.endswith(".pdf"):
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            elif file.name.endswith(".docx"):
                doc = Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            elif file.name.endswith(".txt"):
                text += file.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file {file.name}: {e}")
            # Optionally, you might want to skip the file or handle the error differently
            continue
    return text
