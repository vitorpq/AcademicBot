import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

arquivo = "arquivos/database.md"
loader = UnstructuredMarkdownLoader(arquivo)
documents = loader.load()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

#MD Header splits
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)

all_sections = []

for doc in documents:
    sections = markdown_splitter.split_text(doc.page_content)
    all_sections.extend(sections)

vectorstore = FAISS.from_documents(
    documents=all_sections,
    embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
)

prompt = ChatPromptTemplate.from_template(
    '''Responda as perguntas se baseando no contexto fornecido.
    Warning: Você é um chatbot para a faculdade Ruy Barbosa de Salvador. Falar como um estudante de graduação.
    Tentar sempre apresentar a resposta em tópicos ou listas para facilitar ao estudante seguir um passo-a-passo.
    
    contexto: {contexto}
    
    pergunta: {pergunta}''')


retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 25})
setup = RunnableParallel({
    'pergunta': RunnablePassthrough(),
    'contexto': retriever
})

def join_documents(input):
    input['contexto'] = '\n\n'.join([c.page_content for c in input['contexto']])
    return input

setup = RunnableParallel({
    'pergunta': RunnablePassthrough(),
    'contexto': retriever
}) | join_documents


chain = setup | prompt | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash") | StrOutputParser()

# Streamlit UI Styling
st.markdown(
    """
    <style>
    /* Define o fundo roxo escuro para toda a página */
    .stApp {
        background-color: #88185D !important;
    }

    /* Personaliza o título */
    h1, h2, h3 {
        color: #FFA21C !important;
        text-align: center;
    }

    /* Personaliza os botões */
    div.stButton > button:first-child {
        background-color: #FFA21C !important;
        color: white !important;
        font-size: 18px;
        border-radius: 8px;
        border: 2px solid white;
        padding: 10px 24px;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF8C00 !important;
    }

    /* Personaliza os inputs */
    div.stTextInput > label {
        font-size: 18px;
        font-weight: bold;
        color: #FFA21C !important;
    }

    /* Personaliza a área do chat */
    div.stChatMessage {
        background-color: #5C1042 !important;
        border-radius: 8px;
        padding: 10px;
    }

    /* Personaliza os textos gerais */
    body, p, div {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.image("logo-UNIRUY-Branco.png", width=350)
st.sidebar.text("ChatBot de dúvidas para alunos do Centro Universitário Ruy Barbosa - UNIRUY")
st.sidebar.text("Desenvolvido pelo Prof. Vítor E. Andrade")
st.sidebar.text("https://github.com/vitorpq")
# Streamlit Chat Interface
st.title("AcademicBot Chat")
# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
chat_container = st.container()
with chat_container:
    for message in reversed(st.session_state.chat_history):
        role, content = message["role"], message["content"]
        if role == "user":
            st.markdown(f"<div class='user-bubble'>{content}</div>", unsafe_allow_html=True)
        elif role == "assistant":
            st.markdown(f"<div class='chat-bubble'>{content}</div>", unsafe_allow_html=True)


# Input box for user message
if user_input := st.chat_input("Digite sua pergunta aqui..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response using the chain
    response = chain.invoke(user_input)

    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
