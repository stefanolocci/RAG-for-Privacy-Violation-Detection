import pandas as pd
import re


# Function to extract the body of the email
def extract_email_body(message):
    # Split the message at "X-FileName:" and take the second part if it exists
    split_message = re.split(r'X-FileName:.*\n', message, maxsplit=1)
    if len(split_message) > 1:
        return split_message[1].strip()  # Return the email body, stripping leading/trailing whitespace
    else:
        print(message)  # If no split happened, return the original message

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('datasets/emails.csv')

# Step 2: Apply the parser to each row in the 'message' column and create a new column 'emails'
df['emails'] = df['message'].apply(extract_email_body)

# Step 3: Save only the 'emails' column into a new CSV file
df[['emails']].to_csv('datasets/enron_mail_full.tsv', sep='\t', index=False)
