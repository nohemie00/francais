import streamlit as st
import json, re, ast, uuid, traceback
import numpy as np
from typing import List, Any, Optional
from supabase import create_client
from langchain.schema import Document, BaseRetriever
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Streamlit 페이지 설정
st.set_page_config(
    page_title="France Curator Mme.Noy",
    page_icon="🇫🇷",
    layout="wide"
)

# 스타일 커스터마이징 (배경 파랑 + 글자 흰색)
st.markdown("""
    <style>
    .stApp {
        background-color: #0047AB;
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

# 사이드바 문장 색상 커스터마이징
st.markdown("""
    <style>
    .css-1d391kg p.sidebar-highlight {
        color: #B8D8FF !important;
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# 사이드바 내용
with st.sidebar:
    st.markdown("<h2 style='color:#4F8BF9;'>🧑‍🏫 Curator AI: FR</h2>", unsafe_allow_html=True)
    st.markdown("""
    - 프랑스어 문법/회화/고급표현
    - 프랑스 박물관 큐레이션
    - 프랑스 문화/역사/예술
    - 프랑스에 대한 모든 것
    """)
    if st.button("💬 대화 초기화"):
        st.session_state.messages = []


# --- 초기화 실패 시 중단 ---
if not all([OPENAI_API_KEY, COHERE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY]):
    st.warning("❗ 모든 API 키와 주소를 입력해주세요.")
    st.stop()

# --- 리소스 초기화 ---
@st.cache_resource(show_spinner=False)
def init():
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    embeddings = CohereEmbeddings(model="embed-v4.0", cohere_api_key=COHERE_API_KEY)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, api_key=OPENAI_API_KEY)

    text_resp = client.table("text_embeddings").select("content,metadata").limit(5000).execute()
    text_docs = [Document(page_content=it["content"], metadata=it.get("metadata", {})) for it in text_resp.data]
    texts = [d.page_content for d in text_docs]
    bm25 = BM25Retriever.from_texts(texts=texts, metadatas=[d.metadata for d in text_docs], k=5)

    class SupabaseRetriever:
        def __init__(self, table_name="text_embeddings", query_name="match_text_embeddings", k=5):
            self.table = table_name
            self.query = query_name
            self.k = k
        def invoke(self, query, page_filter=None):
            try:
                q_emb = embeddings.embed_query(query)
                matches = client.rpc(self.query, {
                    "query_embedding": q_emb,
                    "match_threshold": 0.5,
                    "match_count": self.k
                }).execute()
                docs = []
                for m in matches.data:
                    meta = m.get("metadata", {}) or {}
                    meta["similarity"] = float(m.get("similarity", 0))
                    meta["source"] = "Vector"
                    if page_filter and str(meta.get("page", "")) != str(page_filter):
                        continue
                    docs.append(Document(page_content=m["content"], metadata=meta))
                return docs
            except Exception as e:
                print("Vector search error:", e)
                return []

        def get_relevant_documents(self, query):
            return self.invoke(query)

    vector_retriever = SupabaseRetriever()

from pydantic import Field
from langchain_core.retrievers import BaseRetriever

class EnsembleRetriever(BaseRetriever):
    retrievers: List[Any] = Field(...)
    weights: List[float] = Field(...)
    
    retriever_names = ["BM25", "Vector"]

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query):
        all_docs = []
        for i, r in enumerate(self.retrievers):
            try:
                docs = r.get_relevant_documents(query)
                for j, d in enumerate(docs):
                    d.metadata = d.metadata or {}
                    d.metadata.update({
                        "source": self.retriever_names[i],
                        "rank": j,
                        "score": 1 / (1 + j) * self.weights[i]
                    })
                    all_docs.append(d)
            except:
                pass
        seen, final = set(), []
        for d in sorted(all_docs, key=lambda x: x.metadata.get("score", 0), reverse=True):
            h = hash(d.page_content)
            if h not in seen:
                seen.add(h)
                final.append(d)
            if len(final) >= 5:
                break
        return final
        
# 아래는 반드시 함수로 감싸야 함!
def init():
    # 이 안에서 필요한 설정들 진행
    client = ...
    embeddings = ...
    llm = ...
    bm25 = ...
    vector_retriever = SupabaseRetriever()

    hybrid = EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.3, 0.7])

    return client, embeddings, llm, hybrid

client, embeddings, llm, hybrid_retriever = init()

# --- 이미지 관련 함수 ---
def analyze_image_relevance(image_url: str, query: str) -> float:
    try:
        q_emb = embeddings.embed_query(query)
        resp = client.table("image_embeddings").select("embedding").eq("metadata->>image_url", image_url).execute()
        if not resp.data: return 0.5
        emb = ast.literal_eval(resp.data[0]["embedding"])
        sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        return (sim + 1) / 2
    except:
        return 0.5

def get_best_images(pages: List[str], query: str, k=3):
    imgs = []
    for p in pages[:3]:
        resp = client.table("image_embeddings").select("metadata").eq("metadata->>page", str(p)).execute()
        for it in resp.data:
            url = it["metadata"].get("image_url")
            if not url: continue
            score = analyze_image_relevance(url, query)
            imgs.append({"url": url, "page": p, "score": score})
    imgs.sort(key=lambda x: x["score"], reverse=True)
    return imgs[:k]

# --- 프롬프트 ---
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
"""다음 대화를 참고해 독립적인 프랑스어 관련 질문으로 바꿔줘:
{chat_history}
새 질문: {question}
⇒""")

QA_PROMPT = PromptTemplate.from_template(
"""당신은 프랑스어와 프랑스 문화에 대한 깊은 전문성을 가진 큐레이터입니다.

당신의 주요 역할은 다음과 같습니다:

1. 프랑스어 학습 지원
- 고급 문법과 어휘에 대한 정확한 설명
- 프랑스어의 뉘앙스와 문화적 맥락을 포함한 설명
- 실용적인 예문과 사용법 제시
- 프랑스어(🇫🇷)로 먼저, 한국어(🇰🇷) 번역은 뒤에 보여 주세요.

2. 프랑스 문화 안내
- 박물관 및 미술관 작품, 사조, 예술사에 대한 전문성 있는 큐레이션
- 예술, 문학, 음식, 역사 등 다양한 문화 주제에 대한 전문적인 설명
- 현대 프랑스 사회와 전통의 조화에 대한 통찰
- 문화적 맥락을 고려한 상세한 설명

3. 멀티모달 요청 처리
- 이미지와 텍스트를 결합한 종합적인 설명
- 시각적 요소와 문화적 맥락의 연계
- 다양한 매체를 활용한 풍부한 설명

응답 작성 시 다음 사항을 준수하세요:
1. 대화의 맥락을 유지하며 자연스러운 대화를 이어가세요.
2. 정확하고 전문적인 정보를 제공하되, 친근하고 이해하기 쉽게 설명하세요.
3. 프랑스어와 한국어를 적절히 혼용하여 설명하세요.
4. 문화적 맥락과 역사적 배경을 포함하여 설명하세요.
5. 사용자의 수준과 관심사에 맞춰 설명을 조정하세요.


참고 문서:
{context}
질문: {question}
답변:
""")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=hybrid_retriever,
    memory=memory,
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
)

# --- UI ---
st.markdown("## 🇫🇷 Curator AI French Edition")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("프랑스에 대해 무엇이든 물어보세요.")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            result = qa_chain.invoke({"question": prompt})
            answer = result["answer"]
            pages = [str(d.metadata.get("page", "")) for d in result.get("source_documents", [])]
            pages = list(dict.fromkeys(pages))
            images = get_best_images(pages, prompt)

            if images:
                answer += "\n\n### 📸 관련 이미지"
                for img in images:
                    answer += f"\n![img]({img['url']})\n(페이지 {img['page']}, 유사도 {img['score']:.2f})"

            placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            placeholder.error("오류 발생: " + str(e))
