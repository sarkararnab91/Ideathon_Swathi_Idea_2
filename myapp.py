import streamlit as st
import os
import dotenv
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from htmlTemplates import css, bot_template, user_template
from langchain.text_splitter import CharacterTextSplitter # Importing from langchain to split the texts into chunks #
from langchain.vectorstores import FAISS                  # Importing to create vector-store #
from langchain.chat_models import ChatOpenAI              # OpenAI LLM (Chargable)
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings # Free version Embedding from Huggingface instruct embedding # OpenAI LLM (Chargable)
from langchain.memory import ConversationBufferMemory     # Initialize memory key 
from langchain.chains import ConversationalRetrievalChain # have some memory to vector store
from langchain.llms import HuggingFaceHub



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    OpenAI = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    model_name = "hkunlp/instructor-xl"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=OpenAI)
    return vectorstore

def get_conversation_chain(vectorstore):
    OpenAI_llm = ChatOpenAI()              #chargable
    #hf_llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})  #Free of cost
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI_llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    
    load_dotenv() # enable API keys  under Main 
    
    # starting page with title, icon #
    st.set_page_config(page_title="Service Enquiry",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    #It is good to initialize when using session_state in streamlit 
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
        
    # User question area # 
    st.header("Service Enquiry:books:")
    user_question = st.text_input("Ask a question about GE Appliance:")
    if user_question:
        handle_userinput(user_question)
     
    # Place to upload the document
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain from #vectorstore

                st.session_state.conversation = get_conversation_chain(vectorstore) #st.session_state : not initiallization 
                    

if __name__ == '__main__':
    main()
