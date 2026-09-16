import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
print("正在使用的 Python 路径:", sys.executable)
print("Gradio 版本:", __import__('gradio').__version__)
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr

# --- 1. 初始化 RAG 组件（这部分和之前类似）---
print("📂 加载向量库...")
# 如果之前用了镜像，这里也加上
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="xenoblade_wiki"
)

llm = Ollama(
    model="qwen3.5:9b",  # 确认你的模型名
    base_url="http://localhost:11434",
    temperature=0.1,
)

# --- 2. 核心查询函数（返回答案、来源和思考过程）---
import time  # 确保在文件开头导入了 time 模块

def query_rag_with_process(question):
    """执行RAG查询，并返回答案、来源文档和思考过程"""
    
    # --- 步骤1：检索 ---
    docs = vectorstore.similarity_search(question, k=5)
    
    thought_process = f"🔍 **1. 检索阶段**\n"
    thought_process += f"收到问题：\"{question}\"\n"
    thought_process += f"从向量库中找到 {len(docs)} 个相关文档片段：\n\n"
    
    sources = []
    context_parts = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "未知章节")
        source_info = f"参考资料 {i+1}：{title}"
        sources.append(source_info)
        
        content_preview = doc.page_content[:150].replace('\n', ' ') + "..."
        thought_process += f"- **{source_info}**\n  `{content_preview}`\n\n"
        
        context_parts.append(f"【{source_info}】\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    # --- 步骤2：构建提示词 ---
    prompt = f"""你是一个专业的游戏攻略助手。请基于以下参考信息回答问题。
如果参考信息中完全没有相关内容，请说"根据现有资料无法回答"。

=== 参考信息 ===
{context}

=== 问题 ===
{question}

=== 回答 ===
"""
    
    thought_process += f"📝 **2. 生成阶段**\n"
    thought_process += f"已构建包含 {len(docs)} 份参考资料的提示词。\n"
    # 新增：显示完整的提示词（调试关键！）
    thought_process += f"\n--- 完整提示词 (供调试) ---\n```\n{prompt[:1000]}{'...' if len(prompt)>1000 else ''}\n```\n"
    thought_process += f"⏳ 正在调用AI模型，请稍候...\n"
    
    # --- 步骤3：生成（记录耗时）---
    start_time = time.time()  # 记录开始时间
    try:
        response = llm.invoke(prompt)
        elapsed = time.time() - start_time  # 计算耗时
        answer = response.strip()
        # 新增：显示原始回答和耗时
        thought_process += f"✅ AI 生成完成，耗时 {elapsed:.2f} 秒。\n"
        thought_process += f"\n--- AI 原始回答 ---\n```\n{answer}\n```\n"
    except Exception as e:
        elapsed = time.time() - start_time
        answer = f"❌ 调用 Ollama 时出错: {e}"
        thought_process += f"❌ 生成失败 (耗时 {elapsed:.2f} 秒): {e}"
    
    # 格式化来源列表
    sources_text = "\n".join([f"- {s}" for s in sources])
    
    return answer, sources_text, thought_process

# --- 3. 创建并启动 Web 界面 ---
# ... 前面的代码保持不变 ...

def create_interface():
    with gr.Blocks(title="RAG 问答系统 - 异度神剑3攻略") as demo:  # 也移除了 theme 参数
        gr.Markdown("""
        # 🎮 异度神剑3 RAG 问答系统
        输入你的问题，AI 会基于游戏攻略资料进行回答。下方会展示完整的检索与思考过程。
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="你的问题", 
                    placeholder="例如：仇恨值怎么计算？",
                    lines=2
                )
                ask_button = gr.Button("🚀 提问", variant="primary")
            
            with gr.Column(scale=1):
                pass
        
        with gr.Row():
            with gr.Column():
                answer_output = gr.Textbox(
                    label="🤖 AI 回答", 
                    lines=8,
                    interactive=False
                )
            with gr.Column():
                sources_output = gr.Textbox(
                    label="📖 参考来源", 
                    lines=8,
                    interactive=False
                )
        
        # 可折叠的思考过程（移除了 show_copy_button）
        with gr.Accordion("🧠 思考与检索过程（点击展开）", open=False):
            thought_output = gr.Textbox(
                label="详细过程",
                lines=15,
                interactive=False
                # show_copy_button 参数已移除
            )
        
        # 绑定按钮事件
        ask_button.click(
            fn=query_rag_with_process,
            inputs=question_input,
            outputs=[answer_output, sources_output, thought_output]
        )
        
        question_input.submit(
            fn=query_rag_with_process,
            inputs=question_input,
            outputs=[answer_output, sources_output, thought_output]
        )
    
    return demo

# ... 后面的启动代码保持不变 ...

# --- 4. 启动服务 ---
if __name__ == "__main__":
    print("🚀 正在启动 Web 界面...")
    print("💡 提示：请确保 Ollama 服务已运行 (ollama serve)")
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False  # 设为 True 可生成公网链接
    )