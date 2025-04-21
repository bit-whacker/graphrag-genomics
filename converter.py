import os
import argparse
from PyPDF2 import PdfReader

def extract_text_from_pdfs(pdf_dir, output_dir):
    """
    Extract text from all PDF files in a directory and save each as a text file.

    :param pdf_dir: Directory containing PDF files.
    :param output_dir: Directory where text files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the PDF directory
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            text_output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")

            try:
                # Read the PDF and extract text
                reader = PdfReader(pdf_path)
                extracted_text = ""

                for page in reader.pages:
                    extracted_text += page.extract_text()

                # Save the extracted text to a text file
                with open(text_output_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(extracted_text)

                print(f"Extracted text from '{filename}' and saved to '{text_output_path}'")

            except Exception as e:
                print(f"Failed to process '{filename}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from PDF files in a directory.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing PDF files.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where extracted text files will be saved.")

    args = parser.parse_args()

    extract_text_from_pdfs(args.input_dir, args.output_dir)
