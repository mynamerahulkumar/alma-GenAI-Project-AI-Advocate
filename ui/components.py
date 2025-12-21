import streamlit as st 
from typing import List,Optional
import tempfile
import os


def init_session_state():
    """Initialize streamlit session state variable"""
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    
    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_intialized=False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files=[]
        

def display_chart_history():
    """Display all messages in the chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            #Show sources if available
            if message.get("sources"):
                with st.expander("sources"):
                    for source in message["sources"]:
                        st.write(f"-{source}")
    

def add_messages(role:str,content:str,sources:List[str]=None):
    """
    Add a message to chat history
    Args:
    role:'user' or 'assistant'
    content:'Message content'
    sources:Optional list of source documents
    """   
    message={"role":role,"content":content}
    if sources:
        message["sources"]=sources
    st.session_state.messages.append(message) 
     

def clear_chat_history():
    """clear chat history"""
    st.session_state.messages=[]
    

def save_uploaded_file(uploaded_file)->str:
    """
     save an uploaded file to temporary location
     Args:
     uploaded_file:streamlit uploadedfile object
     Returns:
      Path to saved files
      
    """
    #create temp directory if it doesn't exist
    temp_dir=tempfile.mkdtemp()
    file_path=os.path.join(temp_dir,uploaded_file.name)
    with open(file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

def display_sidebar_info():
    """Display information in the sidebar"""
    with st.sidebar:
        st.header("📖 About")
        st.markdown("""
                 This is an **AI Advocate RAG Chatbot** that can:

                    📄 Answer questions from your documents
                    🔍 Search the web using Tavily
                    💬 Have natural conversations
                    
                    How to use:

                Upload PDF or TXT files
                Wait for processing
                Ask questions!
                    """)
        st.divider()
        
        #show upload status
        st.header("📁 Uploaded Files")
        if st.session_state.uploaded_files:
            for file in st.session_state.uploaded_files:
                st.write(f"{file}")
        else:
            st.write("No files uploaded yet")
            
        if st.button("Clear chat history"):
            clear_chat_history()
            st.rerun()
 

def display_file_uploader():
    """Display the file upload widget and return uploaded files"""
    uploaded_files=st.file_uploader(
        "Upload your documents (PDF or TXT)"
        type=["pdf","txt"]
        accecpt_multiples_files=True
        help="Upload documents to chat with"
    ) 
    return uploaded_files          

def display_processing_status(message:str,status:str="info"):
    """
    Display a status message
    Args:
    message:Status message
    status:Type -"info" 
    
    """
    if status=="success":
        st.success(message)
    elif status=="warning":
        st.warning(message)
    elif status=="error":
        st.error(message)
    else:
        st.info(message)

    
def create_web_search_toggle()->bool:
    """Create a toggle for web search"""
    return st.toggle("Enabale web search",
                     value=False,
                     help="When enabled,chatbot will also search web for answers")
    
    
        
