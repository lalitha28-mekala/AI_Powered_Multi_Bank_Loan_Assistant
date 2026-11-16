import os
from langchain_community.document_loaders import PyPDFLoader
print(os.listdir("data"))

loader = PyPDFLoader("data/BOB/BOB_HL.pdf") # File path
docs= loader.load()
print(docs[0].page_content)
print(docs[0].metadata)



from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
SBI Home Loan

SBI provides home loans to eligible customers.

The applicant should have a regular income.

The applicant must submit Aadhaar Card and PAN Card.

The maximum loan tenure is 30 years.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
   chunk_overlap=20
)

chunks = splitter.split_text(text)
print(len(chunks))
print(len(text))
for chunk in chunks:
    print(chunk)
    
    print("********")