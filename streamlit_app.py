import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import Document
import tempfile
import os

global filename_cache

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    filename_cache=''

    # Set the API key for LangChain components
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md or .pdf)", type=("txt", "md","pdf")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        if(filename_cache!=uploaded_file.name):
            # 파일 확장자로 타입 확인
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == "pdf":
                # PDF 파일을 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # PyPDFLoader로 PDF 로드
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                finally:
                    # 임시 파일 삭제
                    os.unlink(tmp_file_path)
            else:
                document = uploaded_file.read().decode()
                # Document 객체로 변환
                docs = [Document(page_content=document)]

            # 문서를 청크로 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(docs)
            
            # 임베딩 및 벡터 저장소 생성
            with st.spinner("문서를 처리하고 있습니다..."):
                embeddings = OpenAIEmbeddings()
                vectordb = FAISS.from_documents(split_docs, embeddings)
            
                # 질의응답 체인 구성
                try:
                    # LangChain Hub에서 RAG 프롬프트 가져오기
                    prompt = hub.pull("rlm/rag-prompt")
                except:
                    # Hub에서 가져오기 실패시 기본 프롬프트 사용
                    from langchain_core.prompts import ChatPromptTemplate
                    prompt = ChatPromptTemplate.from_template("""
                    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

                    Question: {question} 

                    Context: {context} 

                    Answer:
                    """)
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                # 검색기 설정
                retriever = vectordb.as_retriever(search_kwargs={"k": 10})
        else:
            filename_cache=uploaded_file.name
        # 질의응답 체인 구성
        qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | ChatOpenAI(model='gpt-4o-mini')
            | StrOutputParser()
        )

        qa_chain_with_context = (
            RunnableParallel(
                {
                    "context": retriever,
                    "question": RunnablePassthrough(),
                    "answer": (  # LLM의 결과를 "answer" 키로 저장
                        {
                            "context": retriever | format_docs,
                            "question": RunnablePassthrough(),
                        }
                        | prompt
                        | ChatOpenAI(model='gpt-4o-mini')
                        | StrOutputParser()
                    ),
                }
            )
        )
        
        # 질문 실행
        with st.spinner("답변을 생성하고 있습니다..."):
            result = qa_chain_with_context.invoke(question)
        
        # 답변 표시
        st.subheader("답변:")
        st.write(result['answer'])

        
        # 소스 문서 표시
        st.subheader("참조된 문서 청크:")
        for i, doc in enumerate(result["context"]):
            with st.expander(f"청크 {i+1}"):
                st.write(doc.page_content)
                # 메타데이터가 있는 경우 표시
                if doc.metadata:
                    st.write("**메타데이터:**")
                    st.json(doc.metadata)
