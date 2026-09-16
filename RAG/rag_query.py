import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA  # 这个需要 pip install langchain

# ============ 配置 ============
CHROMA_DB_PATH = "./chroma_db"
OLLAMA_MODEL = "qwen3.5:9b"  # 改成你装的千问版本

# ============ 1. 加载向量库 ============
print("📂 加载向量库...")
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings,
    collection_name="xenoblade_wiki"
)

# ============ 2. 连接 Ollama ============
print(f"🤖 连接 Ollama 模型: {OLLAMA_MODEL}")
llm = Ollama(
    model=OLLAMA_MODEL,
    base_url="http://localhost:11434",
    temperature=0.1,
)

# ============ 3. 构建 RAG 问答链 ============
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}
    ),
    return_source_documents=True,
)

# ============ 4. 交互式问答 ============
print("\n✅ RAG 系统已就绪！输入问题开始问答（输入 q 退出）\n")

while True:
    query = input("🤔 你: ")
    if query.lower() in ["q", "quit", "exit"]:
        break
    
    print("⏳ 思考中...")
    try:
        result = qa_chain.invoke({"query": query})
        
        print(f"\n🤖 AI: {result['result']}")
        print(f"\n📖 参考来源:")
        for i, doc in enumerate(result["source_documents"]):
            title = doc.metadata.get("title", "未知来源")
            print(f"  [{i+1}] {title}")
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"❌ 出错: {e}")
        print("请确保 Ollama 服务已启动 (ollama serve)")