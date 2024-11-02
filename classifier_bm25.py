import os
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from llama_index.core import Document, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

# Set up LlamaIndex OpenAI API key
Settings.llm = LlamaOpenAI(model="gpt-3.5-turbo")  # Set your preferred model
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
os.environ[
    "OPENAI_API_KEY"] = "MyOpenAIKey"


def load_knowledge_base(experiment):
    """
    Loads documents from the 'abstracts', 'introductions', or both directories based on the experiment type.
    Returns a list of documents after processing.
    """
    print("Loading KB...")
    kb_docs = []

    if experiment == "abs" or experiment == "abs_intros":
        abs_dir = "abstracts_cleaned_rewritten"
        kb_docs.extend(load_and_process_dir(abs_dir))

    if experiment == "intros" or experiment == "abs_intros":
        intro_dir = "introductions_cleaned_rewritten"
        kb_docs.extend(load_and_process_dir(intro_dir))

    return kb_docs


def load_and_process_dir(directory):
    """
    Helper function to load files from a directory, remove newlines, split by periods, and return a list of documents.
    """
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                content = file.read().replace('\n', ' ')  # Remove newlines
                docs.extend(content.split("."))  # Split by periods and add to list
    return docs


def create_bm25_retriever(docs):
    """
    Create a BM25 retriever using LlamaIndex.
    """
    # Convert strings to Document objects
    documents = [Document(text=doc) for doc in docs]

    # Initialize document store with nodes
    splitter = SentenceSplitter(chunk_size=512)
    nodes = splitter.get_nodes_from_documents(documents)

    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=10,  # Set the number of relevant documents to retrieve
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    return bm25_retriever


def semantic_search(query, retriever):
    """
    Perform semantic search using the BM25 retriever, returning the top-k relevant documents.
    """
    retrieved_nodes = retriever.retrieve(query)
    return [node.text for node in retrieved_nodes]


def create_zero_shot_openai_prompt_new(input_text, relevant_docs):
    """
    Create the OpenAI API prompt using relevant documents and input text.
    """
    return f"""Your task is to analyze a text to determine whether it exposes private or sen-
            sitive information, and provide an explanation of your answer. The text to be
            analyzed is the following:
            {input_text}
            Format your answer by providing both a label and a clear and thorough expla-
            nation, as follows:
            Label: it should be “violation detected” or “no violation detected”. 
            Explanation: it should be a complete and well-reasoned analysis, referencing the
            provided documents to justify why the text shows violations or not.
    """


def create_openai_prompt_new(input_text, relevant_docs):
    """
    Create the OpenAI API prompt using relevant documents and input text.
    """
    return f"""
        Your task is to analyze a text to determine whether it exposes private or sen-
        sitive information, and provide an explanation of your answer. The text to be
        analyzed is the following:  {input_text}
        Base your answer on the following passages: {' '.join(relevant_docs)}
        Format your answer by providing both a label and a clear and thorough expla-
        nation, as follows:
        Label: it should be “VIOLATION DETECTED” or “NO VIOLATION DETECTED”.
        Explanation: it should be a complete and well-reasoned analysis, referencing the
        provided documents to justify why the text shows violations or not.
    """

def analyze_dataset_and_save(csv_file, retriever, openai_client, output_file):
    """
    Iterate over the dataset, perform semantic search for each text entry, generate analysis using OpenAI,
    and save the result in a new TSV file with columns: ID, Text, Label, Explanation.
    """
    df = pd.read_csv(csv_file, sep="\t")

    # Initialize a list to store the results
    results = []

    for idx, row in tqdm(df.iterrows()):
        input_text = row['Text'].replace('\n', '')
        id_value = row['ID']  # Get the ID for the row

        # Perform semantic search
        relevant_docs = semantic_search(input_text, retriever)
        # print(relevant_docs)

        # Create OpenAI prompt
        prompt = create_openai_prompt_new(input_text, relevant_docs)

        # Generate analysis using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=512,
            messages=[{"role": "system", "content": prompt}]
        )

        # Extract the text from the response
        answer = response.choices[0].message.content

        # Extract Label and Explanation from the answer
        label_match = re.search(r'Label:\s*(VIOLATION DETECTED|NO VIOLATION DETECTED)', answer)
        explanation_match = re.search(r'Explanation:\s*(.*)', answer, re.DOTALL)

        label = label_match.group(1) if label_match else "Unknown"
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"

        # Save the result as a dictionary
        result = {
            'ID': id_value,
            'Text': input_text,
            'Label': label,
            'Explanation': explanation
        }

        results.append(result)

    # Convert results to a DataFrame and save as a TSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, sep='\t', index=False)
    print(f"Results saved to {output_file}")


def main(experiment):
    # Load documents based on experiment
    kb_docs = load_knowledge_base(experiment)
    print(f"Loaded {len(kb_docs)} documents from the {experiment} experiment")

    # Create BM25 retriever
    retriever = create_bm25_retriever(kb_docs)
    print("BM25 retriever created")

    # Initialize OpenAI API client
    openai_client = OpenAI(
        api_key="MyOpenAIKey")  # Replace with your actual API key

    # Analyze dataset (iterating over 'text' column in CSV)
    dataset = "enron_mail_3k_annotated_gpt35_full_50.tsv"  # Path to your CSV file
    output_file = "new_res/abs/enron_mail_3k_annotated_gpt4o_full_BM25.tsv"
    analyze_dataset_and_save(dataset, retriever, openai_client, output_file=output_file)


if __name__ == "__main__":
    # You can choose "abs", "intros", or "abs_intros" as experiment type
    exp = "abs"  # Example: can be "abs", "intros", or "abs_intros"
    main(exp)
