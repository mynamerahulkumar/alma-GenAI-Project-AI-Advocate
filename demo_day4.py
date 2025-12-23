#!/usr/bin/env python3
"""
Demo Script - Day 4: Streamlit UI
==================================
This script runs the complete Streamlit application.

Run with: streamlit run demo_day4.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from ui.components import (
    init_session_state,
    display_chat_history,
    add_message,
    display_sidebar_info,
    display_file_uploader,
    display_processing_status,
    create_web_search_toggle
)
from ui.chat_interface import ChatInterface


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="RAG Chatbot - Day 4 Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Initialize chat interface
    if "chat_interface" not in st.session_state:
        st.session_state.chat_interface = ChatInterface()
    
    chat = st.session_state.chat_interface
    
    # Display sidebar
    display_sidebar_info()
    
    # Main content
    st.title("🤖 RAG Chatbot - Day 4 Demo")
    st.markdown("**Build Your Own AI-Powered Chatbot with RAG**")
    
    # File upload section
    with st.expander("📤 Upload Documents", expanded=not st.session_state.vector_store_initialized):
        st.markdown("""
        Upload PDF or TXT files to create your knowledge base.
        The chatbot will answer questions based on your documents.
        """)
        
        uploaded_files = display_file_uploader()
        
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                with st.spinner("📄 Processing documents... This may take a minute..."):
                    try:
                        num_chunks = chat.process_uploaded_files(uploaded_files)
                        display_processing_status(
                            f"✅ Success! Processed {len(uploaded_files)} file(s) into {num_chunks} chunks",
                            "success"
                        )
                    except Exception as e:
                        display_processing_status(
                            f"❌ Error processing files: {str(e)}",
                            "error"
                        )
    
    # Web search toggle
    col1, col2 = st.columns([1, 5])
    with col1:
        use_web_search = create_web_search_toggle()
    with col2:
        if use_web_search:
            st.caption("🌐 Web search enabled - can answer real-time questions")
        else:
            st.caption("📄 Document search only")
    
    st.divider()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        add_message("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            try:
                # Stream the response
                response = st.write_stream(
                    chat.get_response(prompt, use_web_search=use_web_search)
                )
                
                # Get sources
                sources = chat.get_sources(prompt)
                
                # Show sources if available
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write(f"- {source}")
                
                # Add assistant message to history
                add_message("assistant", response, sources)
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                add_message("assistant", error_msg)


if __name__ == "__main__":
    main()
