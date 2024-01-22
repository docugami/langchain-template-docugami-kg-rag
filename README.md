
# docugami-kg-rag

This template contains a reference architecture for Retrieval Augmented Generation against a set of documents using Docugami's XML Knowledge Graph (KG-RAG).

## Video Walkthrough

[![Docugami KG-RAG Walkthrough](https://img.youtube.com/vi/xOHOmL1NFMg/0.jpg)](https://www.youtube.com/watch?v=xOHOmL1NFMg)

## Setup

### Environment Setup

You need to set some required environment variables before using your new app based on this template. These are used to index as well as run the application, and exceptions are raised if the following required environment variables are not set:

1. `OPENAI_API_KEY`: from the OpenAI platform.
2. `DOCUGAMI_API_KEY`: from the [Docugami Developer Playground](https://help.docugami.com/home/docugami-api)

```shell
export OPENAI_API_KEY=...
export DOCUGAMI_API_KEY=...
```

Finally, make sure that you run `poetry install --all-extras` (or select a specific set of extras, see pyproject.toml) to install dependencies.

### Process Documents in Docugami (before you use this template)

Before you use this template, you must have some documents already processed in Docugami. Here's what you need to get started:

1. Create a [Docugami workspace](https://app.docugami.com/) (free trials available)
2. Create an access token via the Developer Playground for your workspace. [Detailed instructions](https://help.docugami.com/home/docugami-api).
3. Add your documents to Docugami for processing. There are two ways to do this:
    - Upload via the simple Docugami web experience. [Detailed instructions](https://help.docugami.com/home/adding-documents).
    - Upload via the Docugami API, specifically the [documents](https://api-docs.docugami.com/#tag/documents/operation/upload-document) endpoint. Code samples are available for python and JavaScript or you can use the [docugami](https://pypi.org/project/docugami/) python library.

Once your documents are in Docugami, they are processed and organized into sets of similar documents, e.g. NDAs, Lease Agreements, and Service Agreements. Docugami is not limited to any particular types of documents, and the clusters created depend on your particular documents. You can [change the docset assignments](https://help.docugami.com/home/working-with-the-doc-sets-view) later if you wish. You can monitor file status in the simple Docugami webapp, or use a [webhook](https://api-docs.docugami.com/#tag/webhooks) to be informed when your documents are done processing. The [Docugami RAG over XML Knowledge Graphs (KG-RAG) Cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/docugami_xml_kg_rag.ipynb) has end to end code to upload docs and wait for them to be processed, if you are interested. 

Once your documents are finished processing, you can index them in the following step.

## Usage

### Indexing

Before you can run your app, you need to build your vector index. See [index.py](./index.py) which you can run via `poetry run python index.py` after setting the environment variables as specified above. The CLI will query docsets in the workspace corresponding to your `DOCUGAMI_API_KEY` and let you pick which one(s) you want to index.

Indexing in this template uses the Docugami Loader for LangChain to create semantic chunks out of your documents. Refer to this [documentation](https://python.langchain.com/docs/integrations/document_loaders/docugami) for details.

Note that if you previously ran indexing for the same docset, the index will not be recreated. If you want to force recreate the index (e.g. if you have new docs in the docset or changed your chunking config parameters) please specify `poetry run index.py --overwrite`

### Creating app
To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package git+https://github.com/docugami/langchain-template-docugami-kg-rag.git
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add git+https://github.com/docugami/langchain-template-docugami-kg-rag.git
```

And add the following code to your `server.py` file:
```python
from docugami_kg_rag import chain as docugami_kg_rag_chain

add_routes(app, docugami_kg_rag, path="/docugami-kg-rag")
```

(Optional) Let's now configure LangSmith. 
LangSmith will help us trace, monitor and debug LangChain applications. 
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section


```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

### Running app
If you are inside the app directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/docugami-kg-rag/playground](http://127.0.0.1:8000/docugami-kg-rag/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/docugami-kg-rag")
```
