from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import OpenAI
import networkx as nx
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests import Session
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Debug information: Check if the OPENAI_API_KEY is loaded correctly.
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    print("OPENAI_API_KEY was not loaded from the.env file. Please check the.env file.")
else:
    print("OPENAI_API_KEY has been successfully loaded。")

# Set up the OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key
# Set the USER_AGENT environment variable
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

# Create a `requests` session with custom request headers
session = Session()
session.headers.update({"User-Agent": user_agent})

# Step 1: Define a function to load data from a web page
def load_data_from_web(urls):
    loaders = [WebBaseLoader(url, session=session) for url in urls]
    documents = []
    for loader in loaders:
        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading data from {loader.url}: {e}")
    return documents

# Step 2: Define a text splitting function
def split_text(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

# Step 3: Define a function to generate summaries
def generate_summary(texts):
    llm = OpenAI(temperature=0.7, base_url="https://api.siliconflow.cn/v1", model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    try:  # 捕获异常并打印错误信息
        summary = chain.invoke({"input_documents": texts})
        return summary["output_text"]
    except Exception as e:
        print(f"Error generating summary: {e.__class__.__name__}: {str(e)}")
        return None

# Step 4: Use LangGraph to build an information integration process
def information_integration(urls):
    # Create a NetworkX graph
    G = nx.DiGraph()

    # Define nodes
    G.add_node("load_data")
    G.add_node("split_text")
    G.add_node("generate_summary")

    # Define edges (dependency relationships between nodes)
    G.add_edge("load_data", "split_text")
    G.add_edge("split_text", "generate_summary")

    # Drawing
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=10, font_weight='bold', edge_color='gray')
    plt.title("Information Integration Process")
    plt.show()

    # Execute the operations corresponding to the graph
    documents = load_data_from_web(urls)
    split_docs = split_text(documents)
    summary = generate_summary(split_docs)

    return summary

# get lobste.rs top 10 news
def get_hacker_news_urls():
    base_url = "https://lobste.rs/"
    headers = {"User-Agent": user_agent}
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.select('.story_liner a')[:10]
    urls = []
    for link in links:
        href = link['href']
        full_url = urljoin(base_url, href)
        urls.append(full_url)
    return urls


# Replace with the actual web page URL
if __name__ == "__main__":
    hacker_news_urls = get_hacker_news_urls()
    for index, url in enumerate(hacker_news_urls, start=1):
        print(f"Processing news {index}: {url}")
        try:
            summary = information_integration([url])
            if summary:
                print(f"Summary of news {index}:")
                print(summary)
            else:
                print(f"Failed to generate summary for news {index}")
        except Exception as e:
            print(f"Error processing news {index} from {url}: {e.__class__.__name__}: {str(e)}")
        print("-" * 80)