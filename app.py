import openai
from dotenv import load_dotenv
import os
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import langchain
from annoy import AnnoyIndex
import chardet  # Necesitarás instalar esta biblioteca

langchain.verbose = False

load_dotenv()

def get_openai_embedding(text, openai_api_key):
    response = openai.Embedding.create(
        input=text,
        engine="text-similarity-babbage-001",
        api_key="your_openai_api_key_here"
    )
    return response['data']['embedding']

def create_annoy_index(text_chunks, openai_api_key, vector_size=4096):
    annoy_index = AnnoyIndex(vector_size, 'angular')
    for i, chunk in enumerate(text_chunks):
        vector = get_openai_embedding(chunk, openai_api_key)
        annoy_index.add_item(i, vector)
    annoy_index.build(10)
    return annoy_index

def process_text(text, embeddings):
    return text

def read_file_content(file):
    content = file.read()
    # Usa chardet para detectar la codificación del archivo
    detected_encoding = chardet.detect(content)['encoding']
    # Decodifica el contenido del archivo usando la codificación detectada
    return content.decode(detected_encoding)

def main():
    st.title("SSTC CHAT")
    html = st.file_uploader("Sube tu archivo HTML", type="html")
    rss = st.file_uploader("Sube tu archivo RSS (máx. 3 archivos)", type="xml", accept_multiple_files=True)
    text = ""

    if html is not None:
        html_content = read_file_content(html)
        soup = BeautifulSoup(html_content, 'html.parser')
        text += soup.get_text()

    if rss:
        for rss_file in rss:
            rss_content = read_file_content(rss_file)
            soup = BeautifulSoup(rss_content, 'xml')
            text += soup.get_text()

    if text:
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        knowledge_base = process_text(text, embeddings)
        query = st.text_input('Escribe tu pregunta para los archivos...')
        cancel_button = st.button('Cancelar')

        if cancel_button:
            st.stop()

        if query and knowledge_base:
            query_vector = embeddings(input_text=query)
            indices, _ = knowledge_base.get_nns_by_vector(query_vector, 10, include_distances=True)
            model = "gpt-3.5-turbo-instruct"
            temperatura = 0
            llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperatura=temperatura)
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cost:
                start_time = datetime.now()
                response = "Aquí va la respuesta generada"
                end_time = datetime.now()

                st.write(response)
                st.write(f"Tokens consumidos: {cost.total_tokens}")
                st.write(f"Tiempo de transacción: {end_time - start_time}")

if __name__ == "__main__":
    main()

