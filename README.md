
# 📊 # GraphRAG-genomics

**GraphRAG-Omics** is an extension of Microsoft's [GraphRAG](https://github.com/microsoft/graphrag) library and TheAiSingularity/graphrag-local-ollama library that enables users to convert unstructured documents into **knowledge graphs** and interact with them using **natural language queries**.

This is just an expremental repository where the prompts were tailored for **genomics** and **clinical documents**.

---

## 🚀 Features

- 📄 **Document Indexing**: Convert raw `.txt` documents into `.parquet` files.
- 🧠 **Knowledge Graph Generation**: Transform indexed documents into a structured knowledge graph stored in a **Neo4j** server.
- 💬 **Natural Language Querying**: Interact with your knowledge graph through an intuitive **Streamlit** web interface — ask questions, get insights.

---

## 🗂️ Project Structure

```
graphrag-omics/
│
├── graphrag_workflow.bat      # Command-line script to index documents
├── app.py                     # Streamlit app for graph creation and querying
├── input/                     # Directory to place raw .txt documents
└── proj_<project_name>/       # Generated output for each project
```

### Components

1. **Command-Line Indexing Script**
   - Takes input `.txt` documents
   - Outputs `.parquet` files into a project-specific folder

2. **Streamlit Web App**
   - **Indexing Tab**: Load `.parquet` files and generate a knowledge graph in Neo4j
   - **Query Tab**: Use natural language to query your knowledge graph (GraphRAG interface)

---

## 🧪 How to Run
### Prerequisits

1. install all necessary required libraries
2. install neo4j-desktop
3. install the graphrag, by executing the following command inside the root directory of the project.

```bash
pip install -e .
```

### Step 1: Index Your Documents

1. Place your `.txt` documents inside the `input/` folder (located in the root of the project).
2. Run the following command:

```bash
bash graphrag_workflow.bat proj_<project_name>
```

> 🔒 The project name **must start with `proj_`**  
> ✅ Example: For a project named "med", use:

```bash
bash graphrag_workflow.bat proj_med
```

This will create a folder `proj_med/` and generate the `.parquet` files inside it.

---

### Step 2: Generate the Knowledge Graph & Query

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Navigate to your browser where the app opens automatically.
3. Use the following tabs inside the app:
   - **Indexing**: Select a project (e.g., `proj_med`) and generate the knowledge graph in Neo4j.
   - **Query**: Ask questions using natural language — powered by the generated knowledge graph (GraphRAG style).

---

## 📌 Notes

- Only `.txt` documents are currently supported.
- Ensure that the Neo4j server is running before using the **Indexing** or **Query** functionality in the app.
- This project is under active development — feedback and contributions are welcome!

---

## 🧬 Use Cases

- Genomics research papers
- Clinical documents & patient summaries
- Biomedical literature mining
- Interactive Q&A from specialized unstructured data
