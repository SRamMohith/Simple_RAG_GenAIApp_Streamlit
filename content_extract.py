from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(source_type,source):
    if source_type=='Web link':
        loader = WebBaseLoader(source)
    elif source_type=='PDF':
        with open(source.name, mode='wb') as w:
            w.write(source.getvalue())
        loader = PyPDFLoader(source.name)

    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=250)
    docs = splitter.split_documents(document)

    return docs