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
## Using Postman to Access the APIs

### Importing the Postman Collection
- Open Postman.
- Click on the Import button in the top left corner.
- Select the File tab.
- Click on Choose Files and select the exported Postman collection file (IntelAPI.postman_collection.json).

### Using the APIs
- Ensure your FastAPI server is running (see the "Running the Application Locally for Development" section).
- Open the imported collection in Postman.
- You will see the various endpoints available in the collection.
- Select an endpoint and click Send to make a request.

## Usage

### API Endpoints

- **Upload File:** `/upload-file/`
    - Method: POST
    - Description: Uploads a file to the server.
    - Request Body: Multipart/form-data
    - Response: JSON with status and message.

- **Remove Duplicate Rows:** `/remove-duplicate-rows/`
    - Method: POST
    - Description: Remove duplicate rows from the uploaded file.
    - Request Body: JSON with `session_id`.
    - Response: JSON with status and message.

### Example Request

**Perform Sentiment Analysis**

```sh
curl -X POST "http://127.0.0.1:8000/remove-duplicate-rows/" -H "accept: application/json" -H "Content-Type: application/json" -d '{"session_id": "your_session_id"}'
```

You can use other api endpoints in this way.
