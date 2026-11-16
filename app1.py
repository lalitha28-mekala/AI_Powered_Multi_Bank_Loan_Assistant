import os
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ----------------------------
# Load Environment Variables
# ----------------------------

load_dotenv()

os.environ["USER_AGENT"] = "Mozilla/5.0"


# ----------------------------
# Streamlit Config
# ----------------------------

st.set_page_config(
    page_title="Bank AI Assistant",
    page_icon="🏦",
    layout="wide"
)


# ----------------------------
# Title
# ----------------------------

st.title("🏦 Multi Bank AI Assistant")
st.write(
    """
    Ask questions related to SBI, HDFC, Bank of Baroda 
    loans, charges, eligibility, documents and FAQs.
    """
)


# ----------------------------
# Load Chroma Database
# ----------------------------

@st.cache_resource
def load_vector_db():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


    vector_db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    return vector_db



vector_db = load_vector_db()



# ----------------------------
# Retriever
# ----------------------------

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k":5
    }
)



# ----------------------------
# Custom Prompt
# ----------------------------


prompt_template = """

You are a helpful banking assistant.

Answer the user's question only using the provided context.

Rules:

1. Do not create information from your own knowledge.
2. If answer is not available in documents say:
   "I could not find this information in the available bank documents."
3. Give simple customer-friendly answers.
4. Mention bank name whenever available.
5. For comparison questions compare only available information.
6. Provide source document name at the end.


Context:

{context}


Question:

{question}


Answer:

"""


PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=[
        "context",
        "question"
    ]
)



# ----------------------------
# LLM
# ----------------------------


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)



# ----------------------------
# RAG Chain
# ----------------------------


qa_chain = RetrievalQA.from_chain_type(

    llm=llm,

    retriever=retriever,

    chain_type="stuff",

    chain_type_kwargs={
        "prompt":PROMPT
    },

    return_source_documents=True

)



# ----------------------------
# Chat Memory
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages=[]



for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])




# ----------------------------
# User Input
# ----------------------------


user_question = st.chat_input(
    "Ask your banking question..."
)



if user_question:


    st.session_state.messages.append(
        {
            "role":"user",
            "content":user_question
        }
    )


    with st.chat_message("user"):
        st.write(user_question)



    with st.chat_message("assistant"):


        with st.spinner("Searching bank documents..."):


            response = qa_chain.invoke(
                {
                    "query":user_question
                }
            )


            answer=response["result"]


            st.write(answer)



            # ----------------------------
            # Sources
            # ----------------------------


            st.subheader("📚 Sources")


            sources=set()


            for doc in response["source_documents"]:


                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                sources.add(source)



            for s in sources:

                st.write(
                    "📄",
                    s
                )



    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )