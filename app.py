import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from streamlit_chat import message as st_message
from langchain import PromptTemplate

st.set_page_config(
    page_title="PDF question answering",
    page_icon="drl logo.png",
    layout="wide"
)
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: black;'>PDF Question Answering</h2>", unsafe_allow_html=True)
    st.markdown('''
    ## About
    Guess What!! This tool will help you get rid of reading and understanding large PDFs. Just upload it here and get to know what you want. Interesting right, want to try? Go ahead and check.
    ''')
    add_vertical_space(2)
    st.image('drl logo.png')
    # st.markdown("<h3 style='text-align: center; color: black;'>Dr Reddy's Laboratories</h3>", unsafe_allow_html=True)

def main():
    st.markdown('#### Upload the PDF')
    load_dotenv()
    pdf=st.file_uploader("",type="pdf")
    if pdf is not None:
        if "history" not in st.session_state:
            st.session_state.history = []
        pdf_reader= PdfReader(pdf)

        text=""
        for page in pdf_reader.pages:
            text +=page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        st.success('Understood it completely')
        query = st.text_input("Enter your query here:")
        if query:
            docs = VectorStore.similarity_search(query=query)
            prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            Answer:"""
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )            
            llm = OpenAI(max_tokens=256,n=1,temperature=0,top_p=1.0,frequency_penalty=0.5,presence_penalty=0.5,best_of=2,model='text-davinci-003')
            chain=load_qa_chain(llm,chain_type="stuff",prompt=PROMPT)
            answer = chain({"input_documents": docs, "question": query}, return_only_outputs=True)['output_text']
            st.markdown(answer)
            st.session_state.history.append({"message": query, "is_user": True})
            st.session_state.history.append({"message": answer, "is_user": False})
        for i, chat in enumerate(st.session_state.history):
            st_message(**chat, key=str(i)) #unpacking
                    

if __name__ == '__main__':
    main()