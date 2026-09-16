import os
import json
import glob

# 使用国内镜像（加速下载）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 修正后的导入
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ============ 配置 ============
PAGES_DIR = "xenoblade_wiki/pages"  # 你的文件目录
CHROMA_DB_PATH = "./chroma_db"

# ============ 1. 读取并处理所有 JSON 文件 ============
def load_json_files(directory):
    """读取目录下所有 JSON 文件，提取有效内容"""
    documents = []
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    print(f"📁 找到 {len(json_files)} 个 JSON 文件")
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 提取元数据
            metadata = {
                "source": data.get("url", ""),
                "title": data.get("title", ""),
                "file": os.path.basename(file_path)
            }
            
            # ---- 核心内容：按段落处理 ----
            paragraphs = data.get("paragraphs", [])
            
            # 过滤掉空段落和纯公式（太短或全是特殊字符）
            valid_paragraphs = [
                p for p in paragraphs 
                if p and len(p) > 10 and not all(c in "{}()+-×÷= " for c in p)
            ]
            
            # 把有效段落组合成文档
            if valid_paragraphs:
                # 方式1：整段作为独立文档（保留上下文）
                full_text = "\n".join(valid_paragraphs)
                doc = Document(
                    page_content=full_text,
                    metadata={
                        **metadata,
                        "type": "full_content",
                        "chunk_type": "paragraph_combined"
                    }
                )
                documents.append(doc)
                
                # 方式2：每个段落单独作为文档（提高召回精度）
                for i, para in enumerate(valid_paragraphs[:20]):  # 限制单文件最大20段
                    doc = Document(
                        page_content=para,
                        metadata={
                            **metadata,
                            "type": "single_paragraph",
                            "paragraph_index": i,
                            "chunk_type": "single_paragraph"
                        }
                    )
                    documents.append(doc)
            
            # ---- 额外：标题作为独立文档（增强标题检索） ----
            headings = data.get("headings", {})
            all_headings = []
            for level in ["h1", "h2", "h3"]:
                if headings.get(level):
                    all_headings.extend(headings[level])
            
            if all_headings:
                heading_text = " | ".join(all_headings)
                doc = Document(
                    page_content=f"标题索引: {heading_text}",
                    metadata={
                        **metadata,
                        "type": "headings",
                        "chunk_type": "headings"
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            print(f"⚠️ 处理 {file_path} 时出错: {e}")
    
    return documents


# ============ 2. 加载文档 ============
print("🚀 开始加载文档...")
docs = load_json_files(PAGES_DIR)
print(f"✅ 共生成 {len(docs)} 个文档片段")

# ============ 3. 切分文档（进一步细化） ============
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,           # 减小块大小，提高召回精度
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    length_function=len,
)

# 只对长文档进行切分（短文档保留原样）
chunks = []
for doc in docs:
    if len(doc.page_content) > 500:
        # 长文档切分
        split_docs = splitter.split_documents([doc])
        chunks.extend(split_docs)
    else:
        # 短文档直接保留
        chunks.append(doc)

print(f"✂️ 切分后共 {len(chunks)} 个块")

# ============ 4. 初始化 Embedding 模型 ============
print("📥 加载 embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",  # 中文支持更好
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ============ 5. 存入向量库 ============
print("💾 构建向量库...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DB_PATH,
    collection_name="xenoblade_wiki"
)
vectorstore.persist()

print(f"✅ 成功存入 {len(chunks)} 个文档片段到向量库")
print(f"📂 向量库保存在: {CHROMA_DB_PATH}")

# ============ 6. 测试查询 ============
print("\n🧪 测试检索...")
test_queries = ["仇恨值怎么计算", "什么是二类仇恨", "衔尾蛇变身"]
for query in test_queries:
    results = vectorstore.similarity_search(query, k=2)
    print(f"\n🔍 查询: {query}")
    for i, doc in enumerate(results):
        print(f"  结果{i+1}: {doc.metadata.get('title', '未知')[:30]}...")
        print(f"  内容预览: {doc.page_content[:100]}...")