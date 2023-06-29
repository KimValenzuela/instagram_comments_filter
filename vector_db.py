from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain

def loader_docs(model_embed):
    embedding = OpenAIEmbeddings(
        model = model_embed,
        chunk_size = 1000,
    )
    loader = DirectoryLoader(path='comments_file/', glob='*.txt', loader_cls=TextLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs) 

    vectordb = Chroma.from_documents(
        documents = docs, 
        embedding = embedding,
        persist_directory = 'bd'
    )
    vectordb.persist()

    return vectordb


def make_chain(model, temperature, vectordb, query_user):
    llm = OpenAI(
        model = model,
        temperature=temperature
    )

    retriever = vectordb.as_retriever()

    template_query = """
    Esta es una publicación de instagram, en donde quiero que respondas la siguiente pregunta en base a los comentarios almacenados.
    Pregunta: 
    """

    template = """
    {summaries}
    {question}
    """ 

    prompt_temp = PromptTemplate(input_variables=['summaries', 'question'], template=template)
    prompt_value = prompt_temp.format(summaries=template_query,question=query_user)

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        chain_type = 'stuff',
        retriever=retriever,
        chain_type_kwargs={
            'prompt': prompt_temp
        },
    )

    response = qa_chain(prompt_value)
    return response['answer']


