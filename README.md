# Sentiment Analysis API

This project is a sentiment analysis API built using FastAPI, Redis and VADER sentiment analysis. It includes a Jupyter Notebook for implementation, API endpoints and a UI page.

## Prerequisites

- Python 3.7+
- Redis server
- Jupyter Notebook

## Jupyter Notebook

### Installation

If you don't already have Jupyter Notebook installed, you can install it using pip:

```sh
pip install notebook
```

### Running the Notebook

Navigate to the directory containing the Jupyter Notebook:

```sh
cd notebooks
```

Launch Jupyter Notebook:
```sh
jupyter notebook
```
A browser window should open automatically. If it doesn't, open a browser and navigate to the URL provided in the terminal (typically http://localhost:8888/).

Open the 'dataScrapping.ipynb' notebook from the Jupyter interface to view and extract data from amazon and open 'dataCleaning.ipynb' to clean and analyze the data.


## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/your-repo/sentiment-analysis-api.git
    cd sentiment-analysis-api
    ```

2. Install the required Python packages:

    ```sh
    pip install -r requirements.txt
    ```

## Running the Application Locally for Development

1. Start the Redis server:

    ```sh
    redis-server
    ```

2. Run the FastAPI application with Uvicorn:

    ```sh
    uvicorn redisapp:app --reload
    ```

