# FastAPI Backend
================

This is a FastAPI backend project that uses Qdrant for vector search.

## Table of Contents
-----------------

* [Getting Started](#getting-started)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Environment Variables](#environment-variables)
* [Running the Application](#running-the-application)
* [API Documentation](#api-documentation)

## Getting Started
---------------

This project provides a FastAPI backend for vector search using Qdrant.

### Prerequisites

* Python 3.12+
* FastAPI
* Qdrant
* MkDocs

### Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Environment Variables

The following environment variables are required:

* `OPENAI_API_KEY`: OpenAI API key
* `OPENAI_PROXY`: OpenAI http proxy

```
Rename file /server/.env.example to /server/.env
```

### Running the Application

To run the application, use the following command:

```bash
docker compose up
```

### API Documentation

The API documentation is available at `/docs`.

http://localhost:8080/docs

The Code documentations of main .py

http://localhost:8000

The Google Colab Notebook

https://colab.research.google.com/drive/12Ux-1WQF3kIJrujrd62MqDZeQosmEZ3o?usp=sharing

## Contributing
------------

by Erdeni Tsyrendashiev

## License
-------

This project is licensed under the MIT License.
```

Note: You should update the README.MD file to reflect the specific requirements and details of your project.