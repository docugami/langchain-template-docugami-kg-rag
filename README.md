# Docugami KG-RAG Template with Langchain

## About this Project

This project comes with the `docugami-kg-rag` template preinstalled in the packages. This template contains a reference architecture for Retrieval Augmented Generation against a set of documents using Docugami's XML Knowledge Graph (KG-RAG).

## Setup

Please follow the steps in setting up the `docugami-kg-rag` template [here](packages/docugami-kg-rag/README.md).

We heavily recommend creating a [virtual environment](https://github.com/pypa/virtualenv) when working with our project.

After creating your virtual environment, you can install dependencies by running:

```bash
poetry install
```

We also require you to set environment variables in order to use our tools. You will also need your Docugami API key and an OpenAI API key.

```bash
export DOCUGAMI_API_KEY=...
export OPENAI_API_KEY=...
```

## Adding more packages

```bash
# adding packages from 
# https://github.com/langchain-ai/langchain/tree/master/templates
langchain app add $PROJECT_NAME

# adding custom GitHub repo packages
langchain app add --repo $OWNER/$REPO
# or with whole git string (supports other git providers):
# langchain app add git+https://github.com/hwchase17/chain-of-verification

# with a custom api mount point (defaults to `/{package_name}`)
langchain app add $PROJECT_NAME --api_path=/my/custom/path/rag
```

Note: you remove packages by their api path

```bash
langchain app remove my/custom/path/rag
```

## Setup LangSmith (Optional)
LangSmith will help us trace, monitor and debug LangChain applications. 
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section


```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

## Running the project

Before the project can work, you'll have to index your Docugami docsets.

```bash
poetry run index

1: Master Services Agreements (ID: d6aubexzhf5c)
2: Non-Disclosure Agreements (ID: j3ms43eg302p)
3: Earnings Calls (ID: f46vx4fcmxac)

Please enter the number(s) of the docset(s) to index (comma-separated) or 'all' to index all docsets:
```

Note that indexing may take awhile, especially with larger docsets.

Once you're done creating your index, launch your project with:

```bash
poetry run uvicorn app.server:app --reload
```
