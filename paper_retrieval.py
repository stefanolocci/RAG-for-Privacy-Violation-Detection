import requests
import json
import os
from bs4 import BeautifulSoup

# Directory to save downloaded papers
DOWNLOAD_DIR = "downloaded_papers"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Semantic Scholar API URL and query terms DATE OF RETRIEVAL: 15 October 2024
API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
QUERY_TERMS = 'privacy | "sensitive data nlp" | "privacy llm" | "sensitive data"'
QUERY_TERMS = 'privacy | "privacy-preserving" | "sensitive data" | "sensitive information" | "confidential data" | "confidential information" | "personal data" | "personally-identifiable information" | PII'
# QUERY_TERMS_FULL = '((privacy | "privacy-preserving" | "sensitive data" | "sensitive information" | "confidential data" | "confidential information" | "personal data" | "personally-identifiable information") & ("natural language processing" | NLP | "large language models" | LLMs | "machine learning" | AI))'
FIELDS = "title,abstract,authors,openAccessPdf,url"
LIMIT = 100 #MAX Limit without API key


def fetch_papers_from_api(query, fields, limit):
    """
    Fetches papers from the Semantic Scholar API based on the given query.
    """
    params = {
        "query": query,
        "fields": fields,
        "limit": limit
    }
    response = requests.get(API_URL, params=params)
    return response.json()


def save_json_to_file(data, filename):
    """
    Saves the given JSON data to a file.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {filename}")


def download_papers_from_json(file_name):
    """
    Parses the JSON file to extract paper information and download the corresponding PDFs.
    """
    with open(file_name, "r") as f:
        data = json.load(f)

        for paper in data.get("data", []):
            # Extracting the relevant fields from the paper
            paper_id = paper.get("paperId")
            title = paper.get("title", "unknown_title")
            paper_url = paper.get("url")

            # Check if openAccessPdf exists and is not None
            open_access_pdf = paper.get("openAccessPdf")
            pdf_url = open_access_pdf.get("url") if open_access_pdf else None

            # Print paper information
            print(f"\nPaper ID: {paper_id}")
            print(f"Title: {title}")
            print(f"URL: {paper_url}")
            print(f"PDF URL: {pdf_url if pdf_url else 'No open access PDF available'}")
            print(f"Authors: {[author['name'] for author in paper.get('authors', [])]}")

            # If no open access PDF, try to scrape the Semantic Scholar page
            if not pdf_url:
                print(f"Scraping Semantic Scholar page for {title}...")
                pdf_url = scrape_semantic_scholar_for_pdf(paper_url)

            # Download if PDF URL is found
            if pdf_url:
                print(f"Downloading: {title}")
                download_pdf(pdf_url, title)
            else:
                print(f"No PDF found for {title}")


def download_pdf(url, title):
    """
    Downloads the PDF from the given URL and saves it with a sanitized title.
    """
    try:
        response = requests.get(url)
        file_path = os.path.join(DOWNLOAD_DIR, f"{sanitize_title(title)}.pdf")

        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Saved to {file_path}")
    except Exception as e:
        print(f"Failed to download {title}: {e}")


def sanitize_title(title):
    """
    Sanitizes the paper title to create a valid filename.
    """
    return "".join(x for x in title if x.isalnum() or x in "._- ").strip()


def scrape_semantic_scholar_for_pdf(paper_url):
    """
    Scrapes the Semantic Scholar page to extract the PDF link.
    """
    try:
        response = requests.get(paper_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for the PDF link in the anchor with the correct attributes
        pdf_link = soup.find('a', {'data-test-id': 'paper-link', 'href': True})

        if pdf_link:
            return pdf_link['href']
        else:
            print(f"No PDF link found on page: {paper_url}")
            return None
    except Exception as e:
        print(f"Error scraping {paper_url}: {e}")
        return None


def main():
    # Fetch papers from the API
    papers_data = fetch_papers_from_api(QUERY_TERMS, FIELDS, LIMIT)

    # Save the response data to a JSON file
    json_filename = "eligible_KB_papers_3.json"
    save_json_to_file(papers_data, json_filename)

    # Calculate and print the number of papers retrieved
    num_papers = len(papers_data.get("data", []))
    print(f"Number of papers retrieved: {num_papers}")

    # Download the papers
    download_papers_from_json(json_filename)


if __name__ == "__main__":
    main()
