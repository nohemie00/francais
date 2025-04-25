import streamlit as st
import os
import tempfile
import base64
import io
import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain.memory import ConversationBufferMemory
from supabase import create_client
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import uuid


# Streamlit 페이지 설정
st.set_page_config(
    page_title="Prof. Francais",
    page_icon="🇫🇷",
    layout="wide"
)

# 스타일 커스터마이징 (배경 파랑 + 글자 흰색)
st.markdown("""
    <style>
    .stApp {
        background-color: #0047AB; /* 전체 배경 */
        color: white;
    }
    .stMarkdown p, .stTextInput > div > div > input {
        color: white !important;
    }
    .stTextInput > div > div {
        background-color: #0055cc !important;
    }
    .stChatMessage {
        background-color: #0055cc;
        color: white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatInputContainer {
        background-color: #003b7a;
        padding: 1rem;
        border-radius: 10px;
    }
    button[kind="primary"] {
        background-color: #FF4B4B;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# 커버 이미지
st.image("https://raw.githubusercontent.com/nohemie00/francais/main/assets/FRANCAIS_.png", use_container_width=True)

# 사이드바에 제목 추가
with st.sidebar:
    st.markdown("<h2 style='color:#4F8BF9;'>🧑‍🏫 Prof. Francais FR</h2>", unsafe_allow_html=True)
    st.markdown("쉽고 재미있게 프랑스어를 배우도록 도와주는 Noy 선생님이에요.")
    st.markdown("""
    - ✅ 문법 교정  
    - ✅ 발음 설명  
    - ✅ 회화 연습  
    - ✅ 문화 설명  
    - ✅ 고급 불어
    """)
    if st.button("💬 대화 초기화"):
        st.session_state.messages = []

# 환경 변수 설정
if 'OPENAI_API_KEY' not in st.secrets:
    st.sidebar.warning('OpenAI API 키를 설정해주세요!')
    OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')
else:
    OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

if 'SUPABASE_URL' not in st.secrets:
    st.sidebar.warning('Supabase URL을 설정해주세요!')
    SUPABASE_URL = st.sidebar.text_input('Supabase URL')
else:
    SUPABASE_URL = st.secrets['SUPABASE_URL']

if 'SUPABASE_KEY' not in st.secrets:
    st.sidebar.warning('Supabase Key를 설정해주세요!')
    SUPABASE_KEY = st.sidebar.text_input('Supabase Key', type='password')
else:
    SUPABASE_KEY = st.secrets['SUPABASE_KEY']

# 필요한 키가 모두 있는지 확인
if not (OPENAI_API_KEY and SUPABASE_URL and SUPABASE_KEY):
    st.warning('모든 API 키를 입력해주세요!')
    st.stop()

# Supabase 클라이언트 초기화
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f'Supabase 연결 오류: {e}')
    st.stop()

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536,
    api_key=OPENAI_API_KEY
)

# LLM 모델 설정
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

# 프롬프트 템플릿
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
주어진 대화 기록과 새로운 질문을 참고하여 독립적인 질문을 만드세요.

대화 기록: {chat_history}
새로운 질문: {question}

독립적인 질문:""")

QA_PROMPT = PromptTemplate.from_template("""
당신은 프랑스인이며 다른 나라에서 온 사람에게 프랑스어를 가르치는 전문가입니다. 
다음 내용을 참고하여 사용자의 질문에 친절하고 전문적으로 답변해주세요.
특히 프랑스 예절과 헷갈리기 쉬운 문법에 관련된 내용은 반드시 강조해서 설명해주세요. 
번역이나 조언을 할 땐 고급 불어 표현도 추가로 알려주세요.
친절하고 젊은 선생님의 통통 튀는 어조로 답변해주세요. 

다음과 같은 성격과 특징을 살려 답변해주세요:

1. 틀린 문법으로 물어보면 수정해주고 추가 예시를 들어줌:
   - 수정한 다음 더 좋은 문장이 있으면 제시함.

2. 답변을 할 때는 프랑스어와 한국어를 같이 사용:
   - 무슨 언어로 물어보든 먼저 프랑스어로 대답함.
   - 프랑스어로 대답한 뒤 한국어로 번역한 대답을 추가함.

3. 초심자를 위한 사려 깊은 면모:
   - 깊이 있는 생각과 통찰력 표현
   - 때로는 고급 단어나 문학적 표현 사용하고 이 부분을 인지하도록 알려줌.

4. 젊고 통통 튀는 젊은 선생님:
   - 다시 물어볼 때는 짧게.
   - 말이 늘어지는 느낌 없이 간결하고 명확한 어조
   - 가끔 귀여운 느낌으로 반말 섞어서 하기
   

참고 내용:
{context}

질문: {question}

답변:""")

# 벡터 스토어 초기화
try:
    vectorstore = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="embeddings",
        query_name="match_embeddings"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    st.error(f'벡터 스토어 초기화 오류: {e}')
    st.stop()

# 대화 이력 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 메모리 초기화
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 대화형 검색 체인 생성
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=False,
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    combine_docs_chain_kwargs={'prompt': QA_PROMPT}
)

# 메인 제목
st.title("Noy와 함께 우아당탕 프랑스어 🇫🇷")

# 채팅 인터페이스
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("편하게 질문해. 나 한국어도 잘해."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = qa_chain.invoke({"question": prompt})
            answer = response['answer']
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_message = f"오류가 났어. 잠시만!: {str(e)}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


