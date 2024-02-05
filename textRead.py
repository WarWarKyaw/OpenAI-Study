from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

loader = DirectoryLoader(
    "./input/",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'autodetect_encoding': True}
)
data = loader.load()
# print(data)

text_splitter = CharacterTextSplitter(
    separator='\n\n',
    chunk_size=900,
    chunk_overlap=0,
    length_function=len
)
documents = text_splitter.create_documents([doc.page_content for doc in data])

with open("./output/text_chunks.txt", "w", encoding="utf-8") as file:
    for text in documents:
        file.write(text.page_content)
        file.write('\n--------------------------------------\n')

db = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(openai_api_key='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'),
    persist_directory='testdb'
)

if db:
    db.persist()
    db = None
else:
    print("Chroma DB has not been initialized.")
