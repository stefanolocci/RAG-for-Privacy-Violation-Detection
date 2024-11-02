import os
import pdfplumber  # For parsing PDF documents
from pdfminer.pdfparser import PDFSyntaxError  # To catch PDF-specific syntax errors

# Directories for abstracts and intros
ABSTRACTS_DIR = "abstracts"
INTROS_DIR = "intros"
PDF_FOLDER = "downloaded_papers"

# Create the directories if they do not exist
os.makedirs(ABSTRACTS_DIR, exist_ok=True)
os.makedirs(INTROS_DIR, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)  # Ensure the PDF folder exists as well


def extract_text_from_pdf(pdf_path):
    """
    Extracts the full text from a PDF using pdfplumber.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except PDFSyntaxError as e:
        print(f"Skipping {pdf_path} due to PDFSyntaxError: {e}")
        return None
    except Exception as e:
        print(f"Skipping {pdf_path} due to an error: {e}")
        return None


def extract_abstract_and_introduction(text):
    """
    Extracts the Abstract and Introduction from the text.
    The abstract ends when the introduction starts, and the introduction ends when another chapter starts.
    """
    abstract = ""
    introduction = ""

    # Normalize text to handle different cases for section headings
    text = text.lower()

    # Find indices of relevant sections
    abstract_start = text.find("abstract")
    intro_start = text.find("introduction")

    # Assume "Related Work" or other section starts the next chapter
    next_section_start = min(
        [text.find(section) for section in ["related work", "related works", "conclusion", "references", "methods"] if
         text.find(section) != -1],
        default=len(text)
    )

    # Extract Abstract (from "Abstract" to "Introduction")
    if abstract_start != -1 and intro_start != -1:
        abstract = text[abstract_start:intro_start].strip()

    # Extract Introduction (from "Introduction" to next section heading)
    if intro_start != -1 and next_section_start != -1:
        introduction = text[intro_start:next_section_start].strip()

    return abstract, introduction


def save_text_to_file(text, filename, folder):
    """
    Saves the given text to a .txt file.
    """
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def process_pdfs_in_folder(folder_path):
    """
    Process each PDF in the given folder:
    - Extracts Abstract and Introduction
    - Saves them to appropriate folders
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing {pdf_path}...")

            # Extract text from the PDF
            pdf_text = extract_text_from_pdf(pdf_path)

            if pdf_text is None:
                continue  # Skip the PDF if it couldn't be processed

            # Extract Abstract and Introduction
            abstract, introduction = extract_abstract_and_introduction(pdf_text)

            # Define filenames for the abstract and introduction txt files
            abstract_filename = filename.replace(".pdf", "_abstract.txt")
            intro_filename = filename.replace(".pdf", "_intro.txt")

            # Save the abstract and introduction to their respective folders
            if abstract:
                save_text_to_file(abstract, abstract_filename, ABSTRACTS_DIR)
            if introduction:
                save_text_to_file(introduction, intro_filename, INTROS_DIR)


# Ensure the PDF folder exists and process PDFs
process_pdfs_in_folder(PDF_FOLDER)
