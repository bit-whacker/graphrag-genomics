import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import utils as g_utils
import time


#import streamlit as st
from microsoft_to_neo4j import DB_CONFIG
import microsoft_to_neo4j as mtn
from neo4j import GraphDatabase
#from chroma_db import GenericVectorStore, reduce_chain
from prompipe import reduce_chain
from local_and_global_search import local_search, global_retriever
#import os
import asyncio
#import time

import ollama
# Set Streamlit to wide mode
st.set_page_config(layout="wide")

# Load environment variables
def load_env_vars():
    with open("./ragtest/.env", "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            os.environ[key] = value

#load_env_vars()
os.environ["GRAPHRAG_API_KEY"] = "<API_KEY>"
# Neo4j connection
url = DB_CONFIG["url"]
username = DB_CONFIG["username"]
password = DB_CONFIG["password"]

############

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_graph_data():
    driver = GraphDatabase.driver(url, auth=(username, password))
    with driver.session() as session:
        result = session.run("""
        MATCH (n)-[r]->(m)
        RETURN n.name AS source, m.name AS target, type(r) AS relationship
        LIMIT 100
        """)
        return [(record["source"], record["target"], record["relationship"]) for record in result]

st.title("Chat with Graph + RAG + Local LLM + Local Documents")

level = 1

async def timed_local_search(query):
    start_time = time.time()
    graph_results = local_search(query)
    graph_report_data = "\n\n".join([f"Result {i+1}: {result}" for i, result in enumerate(graph_results)])
    processed_graph_results = reduce_chain.invoke({
        "report_data": graph_report_data,
        "question": query
    })
    end_time = time.time()
    return processed_graph_results, end_time - start_time

async def timed_vector_search(query):
    start_time = time.time()
    #vector_results = vector_store.query(query)
    vector_results = query
    end_time = time.time()
    return vector_results, end_time - start_time

async def timed_global_search(query, level):
    start_time = time.time()
    graph_result = await global_retriever(query, level)
    end_time = time.time()
    return graph_result, end_time - start_time

############

#Constants
PRO_ROOT_DIR = os.getcwd()

#st.set_page_config(layout="wide")

st.session_state["projects"] = g_utils.load_project_names(PRO_ROOT_DIR)
if "selected_project" not in st.session_state:
    if len(st.session_state["projects"]) > 0:
        st.session_state["selected_project"] = st.session_state["projects"][0]

# 2. horizontal menu
selected_menu = option_menu(None, ["GraphIndex", "Query"], 
    icons=['cloud-upload', 'home'], 
    menu_icon="cast", default_index=1, orientation="horizontal")


side_menu_selected = None
# 1. as sidebar menu
with st.sidebar:
    if selected_menu == "Query":
        st.markdown("""
                    ### ⚠️ Before You Start

                    Please make sure you have **generated the knowledge** for this project using the **Indexing** menu in this app **before** using the chat interface.
                    """)


    if selected_menu == 'GraphIndex':
        st.markdown("""
                    ### ⚠️ Before You Start

                    Please make sure you have **indexed your documents** using the provided commandline script with this app.
                    """)
        
    if selected_menu == 'Settings':
        st.info("Settings  here!")
# Body
with st.container():

    if selected_menu == 'GraphIndex':
        st.header(body="GraphIndex", anchor="GraphIndex", divider="red")

        selected_project = st.selectbox("Select Project", options=st.session_state["projects"])
        st.session_state["selected_project"] = selected_project
        st.subheader(f"Project - {selected_project}")

        recent_indexed_folder = g_utils.get_latest_folder(f"{st.session_state['selected_project']}\output")

        if recent_indexed_folder == None:
            st.error("Unable to locate indexed folder!")
        
        st.info(f"recent indexed folder: {recent_indexed_folder}")

        if st.button("Generate Graph"):
            #st.stop()
            st.info(f'{st.session_state["selected_project"]} - Graph Creation Started...')
            graph_dir = f"{st.session_state['selected_project']}/output/{recent_indexed_folder}"
            st.write(f"index dir: {graph_dir}...")
            mtn.import_microsoft_graph(graph_dir)
            st.info("Neo4j Graph Creation Completed...")

    if selected_menu == 'Query':
        st.header(body="Chat with me", anchor="Query", divider="red")
        # Display chat messages from history on app rerun

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                async def run_searches():
                    local_task = asyncio.create_task(timed_local_search(prompt))
                    #vector_task = asyncio.create_task(timed_vector_search(query))
                    global_task = asyncio.create_task(timed_global_search(prompt, level))
                    
                    #return await asyncio.gather(local_task, vector_task, global_task)
                    return await asyncio.gather(local_task, global_task)
            
                results = asyncio.run(run_searches())
                local_results, local_time = results[0]
                global_results, global_time = results[1]
                st.markdown(local_results)
                st.write("Global Results")
                st.markdown(global_results)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": local_results})
    