import streamlit as st
import pandas as pd
from builder import create_vector, qa_chain
import tempfile
import os

st.title('Langchain with custom dataset')

# st.sidebar()

if 'vec' not in st.session_state:
        st.session_state.vec = None
        st.session_state.model_trained= False

file = st.file_uploader("Upload only a .csv file")


if file:
    # Check if the file has the expected format (you can adjust this check based on your requirements)
    if not file.name.endswith('.csv'):
        st.error("Please upload a file with the .csv extension.")
    else:

        df = pd.read_csv(file, encoding='iso-8859-1')

        st.dataframe(df)

        if st.button('Trian Model'):
            try:

                st.write("Training to start...")

                # Save the uploaded file to a temporary directory
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, file.name)

                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(file.getvalue())

                vec = create_vector(temp_file_path)
                st.write("Successfully completed!")
                st.session_state.vec = vec
                st.session_state.model_trained = True

                # Clean up temporary files and directory
                os.remove(temp_file_path)
                os.rmdir(temp_dir)

            except Exception as train_error:
                st.error(f'Error during model training: {str(train_error)}')

    
if st.session_state.model_trained:
    try:
        # Access the vec variable from session_state
        
        question = st.text_input("Question:")
        st.write("Always add question mark '?' at the end of every question")
        if question and st.session_state.vec is not None:
            chain = qa_chain(st.session_state.vec)
            response = chain(question)

            st.header("Answer")
            st.write(response["result"])

    except Exception as qa_error:
        st.error(f'Error during question and answering: {str(qa_error)}')
