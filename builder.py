from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

llm = GooglePalm(google_api_key=api_key, temperature=0)

embedding = HuggingFaceEmbeddings()

vectordb_file_path = "faiss_index"

def create_vector(uploaded_file):

    try:
        # Assuming you have 'FAISS' and 'embedding' defined somewhere in your code
        loader = CSVLoader(file_path=uploaded_file, source_column='prompt', encoding='iso-8859-1')
        data = loader.load()

        vector_db = FAISS.from_documents(documents=data, embedding=embedding)

        if vector_db is None:
            raise ValueError("Failed to create vector_db.")
        
        return vector_db

    except Exception as e:
        # Log the exception or print a helpful message for debugging
        print(f"Error in create_vector: {str(e)}")
        return None


def qa_chain(vectordb):

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "Your request is beyound my scope of assessment." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
        chain_type="stuff",
        retriever= retriever,
        input_key= "query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain



if __name__ == '__main__':
    vec = create_vector("C:/Users/user/Desktop/LangChain/FUPRE_DATA.csv")
    print(type(vec))
    chain = qa_chain(vec)

    print(type(chain))
    print(chain("Where is Fupre?"))
    # chain = qa_chain()
    # print(chain('WHo is the HOD of computer science'))
    # pass
  

