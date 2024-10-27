import streamlit as st
from streamlit import session_state as ss
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from content_extract import load_document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
load_dotenv(find_dotenv())

st.header("Simple GenAI Chatbot")
st.markdown('Please set your source and after a few processing tasks you can start using the chatbot. Smaller sources are reccommended for faster results. The LLM will try to answer the queries based on the source. The chat histroy will not be saved for your next visit.')

with st.sidebar:
  st.markdown('Please enter your [Groq](https://groq.com/) API key to connect with the LLM.\nIf you donot have one please create by visiting the website')
  groq_api = st.text_input(label='Enter your Groq API key',type='password',value=os.environ.get("GROQ_API_KEY", None) or ss.get("GROQ_API_KEY", ""))
  session_id = st.text_input(label='Session Name (will not be saved for next visit)')
  llm_name = st.selectbox('Select LLM',('llama3-8b-8192','gemma-7b-it','llama3-70b-8192','gemma2-9b-it'))
  source_type = st.selectbox('Type of source',('Web link','PDF'),index=None)
  if source_type=='Web link':
    source = st.text_input(label='Link')
  elif source_type=='PDF':
    source = st.file_uploader("Upload pdf",type=["pdf"],help="Scanned documents are not supported yet!")

check = True
if not groq_api:
  st.warning('Enter your Groq API in the sidebar')
  check=False
if (not source_type) or (not source):
  st.warning('Set your source information in the sidebar')
  check=False

if check:
  llm = ChatGroq(model_name=llm_name,groq_api_key=groq_api)

  docs = load_document(source_type,source)

  os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
  embedd = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  vectorstore = FAISS.from_documents(documents=docs,embedding=embedd)
  retriever = vectorstore.as_retriever(search_kwargs={'k':3})

  if 'store' not in ss:
    ss.store = {}

  if 'messages' not in ss:
    ss['messages'] = [
      {'role':'assistant','content':'Hi! How can I help you today?'}
    ]

  for msg in ss["messages"]:
    st.chat_message(msg['role']).write(msg['content'])

  if inp:=st.chat_input():
    ss['messages'].append({'role':'user','content':inp})
    st.chat_message('user').write(inp)

    system_prompt = 'You are an assistant for Q&A tasks. Use the following peices of retrived context to answer the users quires. You can answer the queries without using this context as well but please mention that your not using the context information to answer.  If you cannot answer anything please mention it.\n{context}'
    context_q_sys_prompt = "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    context_q_prompt = ChatPromptTemplate.from_messages([
        ('system',context_q_sys_prompt),
        MessagesPlaceholder('chat_history'),
        ('human','{input}')
    ])
    hist_aware_retriver = create_history_aware_retriever(llm,retriever,context_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ('system',system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human','{input}')
    ])
    qa_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(hist_aware_retriver,qa_chain)

    def get_chat_history(session_id)->BaseChatMessageHistory:
      if session_id not in ss.store:
        ss.store[session_id] = ChatMessageHistory()
      return ss.store[session_id]

    conv_rag_chain = RunnableWithMessageHistory(
        rag_chain,get_chat_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )
    
    response = conv_rag_chain.invoke(
      {'input':inp},
      config = {'configurable':{'session_id':session_id}}
    )
    st.chat_message('assistant').write(response['answer'])
    ss['messages'].append({'role':'assistant','content':response['answer']})
else:
  st.stop()