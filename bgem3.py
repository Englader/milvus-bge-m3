from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection
)
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from os import listdir
from os.path import isfile, join


onlyfiles = [f for f in listdir("./pdf_files") if isfile(join("./pdf_files/", f))]
pagesText = []

for file in onlyfiles:
    try:
        reader = PdfReader(join("./pdf_files/", file))
    except PdfReadError:
        print("invalid PDF file")
    else:
        for page in reader.pages: 
            text = page.extract_text() 
            # print(text)
            pagesText.append(text.replace("\n", " ")) 

# print(pagesText)

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3', # Specify the model name
    device='cpu', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)


connections.connect("default", host="127.0.0.1", port="19530")


fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="index", dtype=DataType.INT64)
]

schema = CollectionSchema(fields, description="Document Embeddings Collection")
collection_name = "doc_embeddings"
collection = Collection(name=collection_name, schema=schema)
# utility.drop_collection(collection_name)

if utility.has_collection(collection_name):
    collection.release()


if collection.has_index():
    collection.drop_index()

index_params = {
    "index_type": "IVF_FLAT",  # Inverted File System
    "metric_type": "L2",  # Euclidean distance
    "params": {"nlist": 128}
}

collection.create_index("embedding", index_params)
print("New index created successfully.")

collection.load()

docs_embeddings = bge_m3_ef.encode_documents(pagesText)

# print(docs_embeddings)
entities= []

for i in range(len(docs_embeddings["dense"])):
    entities.append({"embedding": docs_embeddings["dense"][i], "index": i})

insert_result = collection.insert(entities)

collection.load()

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

queries = ["What was written in the notice board in the selfish giant?"]
query_embeddings = bge_m3_ef.encode_queries(queries)  # Example: search with the first two embeddings
result = collection.search(
    query_embeddings['dense'], 
    "embedding", 
    search_params, 
    limit=6, 
    output_fields=["index"]
)
# Print the text chunks corresponding to the top search results
for hits in result:
    for hit in hits:
        print(f"hit: {hit}, text chunk: {pagesText[hit.entity.get('index')]}")

first_index = result[0][0].entity.get('index')
first_chunk = pagesText[first_index]


import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def get_gpt4_response(context, question, model='gpt-3.5-turbo', temperature=0.1):

    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role":"system",
                "content":"""You are expert at answering questions about stories from childrens book. A user gives you context and then he asks you a question. 
                You answer the question only based from the context that is provided.
                """
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        temperature=temperature,
    )
    response = chat_completion.choices[0].message.content
    return response


# context = first_chunk
# question = queries[0]
# answer = get_gpt4_response(context, question)
#
#
# print(f"Question: {question}")
# print(f"Answer from llm: {answer}")
# print()

context = pagesText
question = "Summarize these two children stories for me. Return answer as a list of dictionaries with fields: {'title':'Summary of (the name of the story)', 'summary':'summary of the story'}"
answer = get_gpt4_response(context, question)


print(f"Question: {question}")
print(f"Answer from llm: {answer}")
print()


context = answer
question = """Create a new childrens story based on the summary of these two children stories for me.
Return answer in JSON format with fields: {'title':'', 'story':''}
"""
answer = get_gpt4_response(context, question, model='gpt-4o', temperature=0.8)


print(f"Question: {question}")
print(f"Answer from llm: {answer}")
print()
