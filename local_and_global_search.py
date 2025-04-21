import os
from typing import Dict, List
import pandas as pd
from neo4j import Result
from langchain_community.graphs import Neo4jGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
#from langchain_community.vectorstores import Neo4jVector
#from langchain.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import Runnable
from neo4j import GraphDatabase
import asyncio
from microsoft_to_neo4j import DB_CONFIG, db_query
import ollama
#from langchain.embeddings import NomicEmbedText
#from langchain_nomic import NomicEmbeddings
#from langchain.chat_models import Mistral
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
import requests

# Configuration
TOP_CHUNKS = 3
TOP_COMMUNITIES = 3
TOP_OUTSIDE_RELS = 10
TOP_INSIDE_RELS = 10
TOP_ENTITIES = 10
RESPONSE_TYPE = "multiple paragraphs"

url = DB_CONFIG["url"]
username = DB_CONFIG["username"]
password = DB_CONFIG["password"]
index_name = DB_CONFIG["index_name"]
#lc_retrieval_query = "MATCH (n) RETURN n"

# Vector store setup
lc_retrieval_query = """
WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
collect {
    UNWIND nodes as n
    MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)
    WITH c, count(distinct n) as freq
    RETURN c.text AS chunkText
    ORDER BY freq DESC
    LIMIT $topChunks
} AS text_mapping,
// Entity - Report Mapping
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WITH c, c.rank as rank, c.weight AS weight
    RETURN c.summary 
    ORDER BY rank, weight DESC
    LIMIT $topCommunities
} AS report_mapping,
// Outside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m) 
    WHERE NOT m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC 
    LIMIT $topOutsideRels
} as outsideRels,
// Inside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m) 
    WHERE m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC 
    LIMIT $topInsideRels
} as insideRels,
// Entities description
collect {
    UNWIND nodes as n
    RETURN n.description AS descriptionText
} as entities
// We don't have covariates or claims here
RETURN {Chunks: text_mapping, Reports: report_mapping, 
       Relationships: outsideRels + insideRels, 
       Entities: entities} AS text, 1.0 AS score, {} AS metadata
"""

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

# Wrap LocalMistral in a Runnable-compatible class
class RunnableLocalMistral(Runnable):
    def __init__(self, base_url="http://localhost:11434"):
        self.llm = LocalMistral(base_url=base_url)

    def invoke(self, input_str: str, config=None):  # Add 'config' as an optional parameter
        # Simply ignore the config if not needed
        return self.llm.generate(input_str)


# Function to directly use the locally deployed Nomic-Embed-Text from Ollama
'''class LocalNomicEmbedText:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def embed_query(self, text):
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": "nomic-embed-text", "text": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]'''


class LocalNomicEmbedText:
    def __init__(self, base_url="http://localhost:11434", embedding_dim=1536):
        self.base_url = base_url
        self.embedding_dim = embedding_dim  # Define the expected dimension

    def embed_query(self, text):
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": "nomic-embed-text", "text": text}
        )
        response.raise_for_status()
        
        # Log and validate the response
        response_json = response.json()
        if "embedding" not in response_json:
            raise ValueError(f"Response does not contain 'embedding': {response_json}")
        
        embedding = response_json["embedding"]
        if not isinstance(embedding, list) or len(embedding) != self.embedding_dim:
            raise ValueError(
                f"The embedding dimension {len(embedding) if isinstance(embedding, list) else 0} "
                f"does not match the expected dimension {self.embedding_dim}. "
                f"Response: {response_json}"
            )
        
        return embedding


# Load environment variables
def load_env_vars():
    with open("./ragtest/.env", "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            os.environ[key] = value

#load_env_vars()


#embedding_vector = embeddings.embed_query(text)
'''
lc_vector = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(model="text-embedding-3-small"),
    url=url,
    username=username,
    password=password,
    index_name=index_name,
    retrieval_query=lc_retrieval_query
)

# LLM setup
llm = ChatOpenAI(model="gpt-4o")
'''

lc_vector = Neo4jVector.from_existing_index(
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    url=url,
    username=username,
    password=password,
    index_name=index_name,
    retrieval_query=lc_retrieval_query
)

# Instantiate the Mistral model from local Ollama
#llm = RunnableLocalMistral()
llm = OllamaLLM(model="mistral")



# Prompts
MAP_SYSTEM_PROMPT = """
You are a helpful assistant responding to questions about data in the provided tables.

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

Use the data provided in the data tables as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following elements:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
        {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
    ]
}}

Preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

Example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

Do not include information where supporting evidence is not provided.

---Data tables---

{context_data}
"""

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

{response_type}

---Analyst Reports---

{report_data}
"""

map_prompt = ChatPromptTemplate.from_messages([
    ("system", MAP_SYSTEM_PROMPT),
    ("human", "{question}"),
])

reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", REDUCE_SYSTEM_PROMPT),
    ("human", "{question}"),
])

map_chain = map_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()

graph = Neo4jGraph(
    url=url,
    username=username,
    password=password,
    refresh_schema=False,
)



async def global_retriever(query: str, level: int, response_type: str = RESPONSE_TYPE) -> str:
    community_data = graph.query(
        """
        MATCH (c:__Community__)
        WHERE c.level IN [0,$level]
        RETURN c.full_content AS output
        """,
        params={"level": level},
    )
    if len(community_data)==0:
        print("ERROR: No Community Data Found in The Graph!!!")
        return []

    async def process_community(community):
        return await asyncio.to_thread(map_chain.invoke, {
            "question": query,
            "context_data": community["output"]
        })

    intermediate_results = await asyncio.gather(
        *[process_community(community) for community in community_data]
    )

    final_response = reduce_chain.invoke({
        "report_data": intermediate_results,
        "question": query,
        "response_type": response_type,
    })
    return final_response

def local_search(query: str, k: int = TOP_ENTITIES) -> List[Dict]:
    return lc_vector.similarity_search(
        query,
        k=k,
        params={
            "topChunks": TOP_CHUNKS,
            "topCommunities": TOP_COMMUNITIES,
            "topOutsideRels": TOP_OUTSIDE_RELS,
            "topInsideRels": TOP_INSIDE_RELS,
        },
    )

