import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import tempfile
import os

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

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

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
                docs = loader.load_and_split()
                
            finally:
                # 임시 파일 삭제
                os.unlink(tmp_file_path)
        else:
            document = uploaded_file.read().decode()
            # Document 객체로 변환
            docs = [Document(page_content=document)]

        # 문서를 청크로 분할 (PDF가 아닌 경우에도 적용)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(docs)
        
        # 임베딩 및 벡터 저장소 생성
        with st.spinner("문서를 처리하고 있습니다..."):
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectordb = FAISS.from_documents(split_docs, embeddings)
        
        # 질의응답 체인 구성 (return_source_documents=True)
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=openai_api_key, temperature=0),
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),  # 상위 3개 청크 검색
            return_source_documents=True
        )
        
        # 질문 실행
        with st.spinner("답변을 생성하고 있습니다..."):
            result = qa.invoke({"query": question})
        
        # 답변 표시
        st.subheader("답변:")
        st.write(result["result"])
        
        # 소스 문서 표시
        st.subheader("참조된 문서 청크:")
        for i, doc in enumerate(result["source_documents"]):
            with st.expander(f"청크 {i+1}"):
                st.write(doc.page_content)
                # 메타데이터가 있는 경우 표시
                if doc.metadata:
                    st.write("**메타데이터:**")
                    st.json(doc.metadata)
