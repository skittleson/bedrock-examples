# https://medium.com/@dminhk/langchain-question-answering-with-amazon-bedrock-1e6519735633

# some knowledge
# pip install python-magic-bin
from langchain.document_loaders import UnstructuredURLLoader
urls = ["https://en.wikipedia.org/wiki/Dune_(2021_film)"]
loader = UnstructuredURLLoader(urls=urls)
docs = loader.load()

#step 2
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(docs)

# step 3
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import Chroma

import boto3

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)
bedrock_embedding = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan-embed-text-v1",
)

vectorstore = Chroma.from_documents(
documents=all_splits,
embedding=bedrock_embedding
)

# step 4
question = "Who directed Dune?"

docs = vectorstore.similarity_search(question)
len(docs)

for item in docs:
    print(item.page_content)
    print(item.metadata)

# Step 5
from langchain.chains import RetrievalQA
from langchain.llms import Bedrock

model_id = "anthropic.claude-v2"
model_kwargs =  { 
    "max_tokens_to_sample": 8191,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

llm = Bedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs
)

query = (f"\n\nHuman:{question}\n\nAssistant:")

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
response = qa_chain({"query": query})

from pprint import pprint
pprint(response['result'])