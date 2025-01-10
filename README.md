# Scheme-Research-Application

**Overview**
The Scheme Research Tool is an innovative web application designed to streamline the research process for government schemes and programs. Built with cutting-edge technologies like Streamlit, OpenAI GPT-4, LangChain, and FAISS, this tool allows users to extract, summarize, and search key details (such as benefits, eligibility, application process, and required documents) from government scheme URLs and PDFs.

This tool is especially useful for researchers, policymakers, and anyone involved in social impact work. It saves time by providing instant insights into various government schemes and allows easy querying for specific details.

**Key Features**
*OpenAI GPT-4:* Powers query-based responses, offering accurate, specific, and context-aware answers based on the extracted data.
*LangChain:* Efficient document processing for unstructured data from URLs and PDFs.
*FAISS:* Fast, high-performance similarity search to match queries with the most relevant documents.
*Streamlit:* User-friendly interface to interact with the tool and easily navigate through inputs, processing, and results.
**How It Works**

*Input Options:*

Users can either enter individual URLs or upload a text file containing a list of URLs.
The tool supports URLs that point to both regular web pages and PDFs.

*Processing:*

If the URL points to a PDF, the tool extracts the text from the document.
The tool processes the content of the URLs or PDFs to extract essential details about the scheme, including:
Scheme Benefits
Application Process
Eligibility Criteria
Documents Required

*Querying:*

Users can enter specific queries about schemes (e.g., "What are the benefits of Scheme XYZ?").
The system performs a fast similarity search and provides the most relevant answer based on the processed data.



*Prerequisites*
Python 3.9 or higher
An OpenAI API key

![Screenshot of the app](/Screenshot 2025-01-10 195403.png)



