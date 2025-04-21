from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain_ollama.llms import OllamaLLM
import requests

MODEL_NAME = 'dunzhang/stella_en_1.5B_v5'
TEXT_FILE_PATH = './book.txt'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Directly use the locally deployed Mistral model via Ollama
class LocalMistral:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def generate(self, prompt):
        response = requests.post(
            f"{self.base_url}/generate",
            json={"model": "mistral", "prompt": prompt}
        )
        response.raise_for_status()
        return response.json()["content"]

class RunnableLocalMistral(Runnable):
    def __init__(self, base_url="http://localhost:11434"):
        self.llm = LocalMistral(base_url=base_url)

    def invoke(self, input_str: str, config=None):  # Add 'config' as an optional parameter
        # Simply ignore the config if not needed
        return self.llm.generate(input_str)


REDUCE_SYSTEM_PROMPT = """
You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.

Generate a response of the target length and format that responds to the user's question, summarizing all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided are ranked in descending order of importance.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should:
1. Remove all irrelevant information from the analysts' reports
2. Merge the cleaned information into a comprehensive answer
3. Provide explanations of all key points and implications appropriate for the response length and format
4. Add sections and commentary as appropriate for the length and format
5. Style the response in markdown

Preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Preserve all data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

Example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"
where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.
Do not include information where supporting evidence is not provided.

---Target response length and format---

multiple paragraphs

---Analyst Reports---

{report_data}
"""

reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", REDUCE_SYSTEM_PROMPT),
    ("human", "{question}"),
])

#llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
#llm = RunnableLocalMistral()
llm = OllamaLLM(model="mistral")
reduce_chain = reduce_prompt | llm | StrOutputParser()