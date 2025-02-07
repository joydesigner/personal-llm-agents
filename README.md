# Technology News Information Integration and Summary Generation Tool

## Overview

This project is an information integration tool written in Python, leveraging the LangChain and NetworkX libraries to scrape technology news from web pages and generate summaries of the news content. The project achieves effective integration and extraction of news content through steps such as loading data from specified web pages, splitting text, and generating summaries using a large language model.

## Install Dependencies
Before running this project, you need to install the necessary Python libraries. You can use the following command to install them:

```bash
pip install -r requirements.txt
```
## Configure Environment Variables
Create a .env file in the project root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage
Ensure that you have completed the dependency installation and environment variable configuration, then run the following command in the terminal:

```bash
python summa_agent.py
```
Here, summa_agent.py is the name of the Python file where you saved the code.

## Customize News Sources
If you want to scrape news content from other web pages, you can modify the URLs in the urls list and replace them with the web page addresses you want to scrape.

## Code Structure
- load_data_from_web(urls): Load data from the specified web page URLs.
- split_text(documents): Split the loaded documents into smaller text chunks.
- generate_summary(texts): Generate a summary of the text using the OpenAI model.
- information_integration(urls): Build the information integration process, including loading data, splitting text, and generating summaries.
## Notes
- Make sure your .env file contains a valid OpenAI API key; otherwise, the program will not be able to use the OpenAI model properly.
- Since network requests and large language model calls may be affected by network conditions and API limitations, there may be delays or errors during the running process.

## Contribution
If you find any issues or have improvement suggestions, please feel free to submit an issue or a pull request.

## License
This project is licensed under the MIT License.