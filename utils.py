import pandas as pd
import os
import re
from openai import OpenAI


def merge_files(tsv_file, csv_file, output_file):
    """
    Merge a TSV file containing emails and a CSV file containing labels and explanations by 'ID'.
    Outputs a new TSV file with the columns: ID, Text, Label, Explanation.
    """

    # Load the TSV file with emails
    emails_df = pd.read_csv(tsv_file, sep='\t', dtype={'ID': int, 'emails': str})

    # Load the CSV file with labels and explanations
    labels_df = pd.read_csv(csv_file, dtype={'ID': int, 'Label': str, 'Explanation': str})

    # Merge the two dataframes on the 'ID' column
    merged_df = pd.merge(emails_df, labels_df, on='ID', how='inner')

    # Rename the columns for clarity
    merged_df.rename(columns={'emails': 'Text'}, inplace=True)

    # Write the merged dataframe to a new TSV file
    merged_df.to_csv(output_file, sep='\t', index=False)

    print(f"Merged file saved to {output_file}")


def clean_text(text):
    """
    Clean the input text by removing unwanted characters and fixing spacing issues.
    """
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Replace multiple spaces and newlines with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Fix common word concatenation issues caused by missing spaces
    text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase letters
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Add space between numbers and letters
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Add space between letters and numbers

    # Remove any leftover weird symbols or unwanted artifacts
    text = re.sub(r'[^\w\s.,;?!-]', '', text)

    # Strip leading/trailing whitespace
    return text.strip()


def process_and_clean_directory(input_dir, output_dir):
    """
    Process all text files in the input directory, clean them, and save the cleaned versions
    in the output directory.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each text file in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Read the content of the file
            with open(input_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Clean the text
            cleaned_text = clean_text(text)

            # Save the cleaned text to the output directory
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(cleaned_text)

    print(f"All files in {input_dir} have been cleaned and saved to {output_dir}")


# Run the cleaning process
if __name__ == "__main__":
    rewrite_text("introductions_cleaned", "introductions_cleaned_rewritten")
