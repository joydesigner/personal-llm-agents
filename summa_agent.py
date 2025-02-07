from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import OpenAI
import networkx as nx
import os

# 加载环境变量
load_dotenv()

# 调试信息：检查 OPENAI_API_KEY 是否被正确加载
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    print("OPENAI_API_KEY 未从 .env 文件中加载，请检查 .env 文件。")
else:
    print("OPENAI_API_KEY 已成功加载。")

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = api_key
# 设置 USER_AGENT 环境变量
os.environ["USER_AGENT"] = "siliconflow"

# 步骤 1: 定义从网页加载数据的函数
def load_data_from_web(urls):
    loaders = [WebBaseLoader(url) for url in urls]
    documents = []
    for loader in loaders:
        docs = loader.load()
        documents.extend(docs)
    return documents

# 步骤 2: 定义文本分割函数
def split_text(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

# 步骤 3: 定义生成摘要的函数
def generate_summary(texts):
    llm = OpenAI(temperature=0.7, base_url="https://api.siliconflow.cn/v1", model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke({"input_documents": texts})
    return summary["output_text"]

# 步骤 4: 使用 LangGraph 构建信息整合流程
def information_integration(urls):
    # 创建 networkx 图
    G = nx.DiGraph()

    # 定义节点
    G.add_node("load_data")
    G.add_node("split_text")
    G.add_node("generate_summary")

    # 定义边（节点之间的依赖关系）
    G.add_edge("load_data", "split_text")
    G.add_edge("split_text", "generate_summary")

    # 执行图对应的操作
    documents = load_data_from_web(urls)
    split_docs = split_text(documents)
    summary = generate_summary(split_docs)

    return summary

# 示例使用
urls = [
    "https://daybreak.hashnode.dev/machine-learning-supervised-learning-2",
    "https://daybreak.hashnode.dev/machine-learning-supervised-learning-1"
]

# 替换为实际的网页 URL
if __name__ == "__main__":
    final_summary = information_integration(urls)
    print(final_summary)