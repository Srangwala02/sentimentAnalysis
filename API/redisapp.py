from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
import pandas as pd
from models import SessionIDModel, ColumnsModel, SpecificColumnsModel, SpecificColumnsFileModel, BoxPlotModel, SingleColumnModel, ConditionsModel, RegexModel, ChartModel, MultiChartModel
from io import StringIO, BytesIO
import redis
import uuid
import json
import matplotlib.pyplot as plt 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import spacy
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

app = FastAPI()

# Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# redis_client = redis.StrictRedis(host='gusc1-amazed-longhorn-30415.upstash.io', port=30415, password='2bdfd3514322444a9a6717a7019c9c7d', db='sentimentAnalysis')
# redis_client = redis.Redis(
#   host='gusc1-amazed-longhorn-30415.upstash.io',
#   port=30415,
#   password='********',
#   ssl=True
# )

nlp = spacy.load('en_core_web_sm')

app.mount("/static", StaticFiles(directory="static"), name="static")

# Function to generate unique session IDs
def generate_session_id():
    return str(uuid.uuid4())

@app.get("/", response_class=HTMLResponse)
async def get_html_page():
    file_path = "index.html"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)
    else:
        return HTMLResponse(content="<h1>File not found</h1>", status_code=404)

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    session_id = generate_session_id()  # Generate unique session ID
    try:
        # Read the CSV file
        content = await file.read()
        # Decode the content to string
        s = str(content, 'utf-8')
        # Convert the string to StringIO object
        data = StringIO(s)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(data)
        # Store the DataFrame in Redis with the session ID as key
        redis_client.set(session_id, df.to_json())
        return JSONResponse({'session_id': session_id, 'message': 'File uploaded successfully'})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)

@app.get("/get-column-counts/")
async def get_column_counts(session_id: str = Query(...)):
    try:
        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded for this session")
        # Decode the bytes object to string
        df_json = df_json.decode('utf-8')
        # Wrap the JSON string in a StringIO object
        data = StringIO(df_json)
        # Read the JSON string into a DataFrame
        df = pd.read_json(data)
        
        # Identify categorical columns
        cat_col = [col for col in df.columns if df[col].dtype == 'object']
        # Get the count of unique values for each categorical column
        result = df[cat_col].nunique()
        
        return JSONResponse({'result': result.to_dict()})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)

@app.get("/get-first-rows/")
async def get_first_rows(session_id: str = Query(...)):
    try:
        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded for this session")
        # Decode the bytes object to string
        df_json = df_json.decode('utf-8')
        # Wrap the JSON string in a StringIO object
        data = StringIO(df_json)
        # Read the JSON string into a DataFrame
        df = pd.read_json(data)
        result = df.head()
        
        # Convert Timestamp objects to strings
        for col in result.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']):
            result[col] = result[col].astype(str)

        return JSONResponse({'result': result.to_dict()})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)

@app.post("/remove-duplicate-rows/")
async def remove_duplicate_rows(body: SessionIDModel):
    try:
        # Retrieve the DataFrame from Redis using the session ID
        session_id = body.session_id
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded for this session")
        # Decode the bytes object to string
        df_json = df_json.decode('utf-8')
        # Wrap the JSON string in a StringIO object
        data = StringIO(df_json)
        # Read the JSON string into a DataFrame
        df = pd.read_json(data)
        df.drop_duplicates(inplace=True)
        redis_client.set(session_id, df.to_json())
        output_file_path = "no_duplicates_file.csv"
        df.to_csv(output_file_path, index=False)
        # Return the CSV file as a file response
        return FileResponse(path=output_file_path, media_type='text/csv', filename=output_file_path)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    
@app.post("/get-info/")
async def get_info(body: SessionIDModel):
    try:
        session_id = body.session_id
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")
        # Decode the bytes object to string
        df_json = df_json.decode('utf-8')
        # Wrap the JSON string in a StringIO object
        data = StringIO(df_json)
        # Read the JSON string into a DataFrame
        df = pd.read_json(data)

        duplicate_rows = df.duplicated().sum()
        
        # Column info: constructing manually
        cols_info = {col: str(df[col].dtype) for col in df.columns}

        # Categorical columns
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        # Numerical columns
        num_cols = [col for col in df.columns if df[col].dtype != 'object']
        # Null counts
        null_counts = df.isnull().sum()
        # Null percentage
        null_percentage = round((df.isnull().sum() / df.shape[0]) * 100, 2)
        # DataFrame shape
        shape = df.shape

        return JSONResponse({
            'duplicate_rows': int(duplicate_rows),
            'cols_info': cols_info,
            'categorical_columns': cat_cols,
            'numerical_cols': num_cols,
            'null_counts': null_counts.astype(int).to_dict(),
            'null_percentage': null_percentage.astype(float).to_dict(),
            'total_rows': int(shape[0]),
            'total_cols': int(shape[1])
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)

@app.post("/get-unique-values-count/")
async def get_unique_values_count(body: SessionIDModel):
    try:
        # Retrieve the DataFrame from Redis using the session ID
        session_id = body.session_id
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded for this session")
        # Decode the bytes object to string
        df_json = df_json.decode('utf-8')
        # Wrap the JSON string in a StringIO object
        data = StringIO(df_json)
        # Read the JSON string into a DataFrame
        df = pd.read_json(data)
        # Identify categorical columns
        cat_col = [col for col in df.columns if df[col].dtype == 'object']
        # Get the count of unique values for each categorical column
        result = df[cat_col].nunique()
        
        return JSONResponse({'result': result.to_dict()})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    
@app.post("/get-unique-values-column/")
async def get_unique_values_column(body: ColumnsModel):
    try:
        # Retrieve the DataFrame from Redis using the session ID
        session_id = body.session_id
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded for this session")
        # Decode the bytes object to string
        df_json = df_json.decode('utf-8')
        # Wrap the JSON string in a StringIO object
        data = StringIO(df_json)
        # Read the JSON string into a DataFrame
        df = pd.read_json(data)
        result = {col: df[col].unique().tolist() for col in body.columns if col in df.columns}
        
        return JSONResponse({'result': result})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    
@app.post("/drop-columns/")
async def drop_columns(body: ColumnsModel):
    try:
        session_id = body.session_id
        columns_to_drop = body.columns

        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)
        # Drop the specified columns
        df1 = df.drop(columns=columns_to_drop)

        # Update the DataFrame in Redis
        redis_client.set(session_id, df1.to_json())

        return {"message": "Columns dropped successfully", "remaining_columns": df1.columns.tolist()}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/drop-null-value-rows/")
async def drop_null_value_rows(body: ColumnsModel):
    try:
        session_id = body.session_id
        columns = body.columns

        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        df.dropna(subset=columns, axis=0, inplace=True)

        # Update the DataFrame in Redis
        redis_client.set(session_id, df.to_json())

        return {"message": "Null value rows of columns are dropped successfully"}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/get-null-counts/")
async def get_null_counts(body: ColumnsModel):
    try:
        session_id = body.session_id
        columns = body.columns

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Calculate null counts for specified columns
        null_counts = df[columns].isnull().sum().astype(int).to_dict()

        return JSONResponse({'null_counts': null_counts})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    
@app.post("/get-column-stats/")
async def get_column_stats(body: ColumnsModel):
    try:
        session_id = body.session_id
        columns = body.columns

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Calculate statistics for specified columns
        stats = {}
        errors = {}
        for col in columns:
            if col not in df.columns:
                errors[col] = 'Column does not exist in the DataFrame'
            elif df[col].dtype == 'object':
                errors[col] = 'Column is not numerical'
            else:
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std_dev': df[col].std()
                }

        return JSONResponse({'stats': stats, 'errors': errors})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    

@app.post("/get-unique-value-counts/")
async def get_unique_value_counts(body: ColumnsModel):
    try:
        session_id = body.session_id
        columns = body.columns

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Calculate value counts for specified columns
        value_counts = {}
        errors = {}
        for col in columns:
            if col not in df.columns:
                errors[col] = 'Column does not exist in the DataFrame'
            else:
                value_counts[col] = df[col].value_counts().to_dict()

        return JSONResponse({'value_counts': value_counts, 'errors': errors})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    
@app.post("/get-specific-value-counts/")
async def get_specific_value_counts(body: SpecificColumnsModel):
    try:
        session_id = body.session_id
        column_value_pairs = body.column_value_pairs

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Calculate value counts for specified column-value pairs
        counts = {}
        errors = {}
        for pair in column_value_pairs:
            column = pair.column
            value = pair.value
            if column not in df.columns:
                errors[column] = 'Column does not exist in the DataFrame'
            else:
                # Convert the value to the appropriate type based on the column's dtype
                if df[column].dtype == 'float64' or df[column].dtype == 'float32':
                    value = float(value)
                elif df[column].dtype == 'int64' or df[column].dtype == 'int32':
                    value = int(value)

                count = df[df[column] == value].shape[0]
                counts[f'{column}_{value}'] = count
                # count = df[df[column] == value].shape[0]
                # counts[f'{column}_{value}'] = count

        return JSONResponse({'counts': counts, 'errors': errors})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    
@app.post("/get-specific-value-rows/")
async def get_specific_value_rows(body: SpecificColumnsModel):
    try:
        session_id = body.session_id
        column_value_pairs = body.column_value_pairs

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Filter rows for specified column-value pairs
        filtered_rows = {}
        errors = {}
        for pair in column_value_pairs:
            column = pair.column
            value = pair.value
            if column not in df.columns:
                errors[column] = 'Column does not exist in the DataFrame'
            else:
                # Convert the value to the appropriate type based on the column's dtype
                if df[column].dtype == 'float64' or df[column].dtype == 'float32':
                    value = float(value)
                elif df[column].dtype == 'int64' or df[column].dtype == 'int32':
                    value = int(value)

                filtered_df = df[df[column] == value]
                if not filtered_df.empty:
                    filtered_rows[f'{column}_{value}'] = filtered_df.to_dict(orient='records')
                else:
                    errors[column] = f'No rows found with {column} = {value}'

        return JSONResponse({'filtered_rows': filtered_rows, 'errors': errors})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    
# @app.post("/create-csv-without-duplicates/")
# async def create_csv_without_duplicates(body: SpecificColumnsFileModel):
#     try:
#         # Get the filtered rows from the previous API
#         response = await get_specific_value_rows(body)

#         if 'filtered_rows' not in response:
#             return JSONResponse({'error': 'No filtered rows found'})

#         # Combine all filtered rows into a single DataFrame
#         combined_df = pd.concat([pd.DataFrame(rows) for rows in response['filtered_rows'].values()], ignore_index=True)

#         # Remove duplicate rows
#         combined_df.drop_duplicates(inplace=True)

#         # Save the DataFrame to a new CSV file
#         output_file_path = body.file_name
#         combined_df.to_csv(output_file_path, index=False)

#         return JSONResponse({'message': 'CSV file created without duplicates', 'file_path': output_file_path})
#     except Exception as e:
#         return JSONResponse({'error': str(e)}, status_code=400)
    
@app.post("/create-boxplot-image/")
async def create_boxplot_image(data: BoxPlotModel):
    try:
        session_id = data.session_id
        column_name = data.column_name
        title = data.title

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Check if the specified column exists in the DataFrame
        if column_name not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' does not exist in the DataFrame")

        # Check if the column is numeric or not
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' is not numeric and cannot be used for a box plot")
        
        # Generate the box plot
        plt.boxplot(df[column_name], vert=False)
        plt.ylabel('Variable')
        plt.xlabel(column_name)
        if title:
            plt.title(title)
        else:
            plt.title('Box Plot')

        # Convert the plot to an image
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        
        # Close the plot to avoid memory leaks
        plt.close()

        # Return the image as a StreamingResponse
        return StreamingResponse(content=image_stream, media_type="image/png")
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    
@app.post("/get-lower-upper-bound/")
async def get_lower_upper_bound(data: SingleColumnModel):
    try:
        session_id = data.session_id
        column_name = data.column_name

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Check if the specified column exists in the DataFrame
        if column_name not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' does not exist in the DataFrame")

        # Check if the column is numeric
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' is not numerical")

        # Remove outliers using the IQR method
        # Q1 = df[column_name].quantile(0.25)
        # Q3 = df[column_name].quantile(0.75)
        # IQR = Q3 - Q1
        # lower_bound = Q1 - 1.5 * IQR
        # upper_bound = Q3 + 1.5 * IQR
        mean = df[column_name].mean() 
        std = df[column_name].std() 

        # Calculate the lower and upper bounds 
        lower_bound = mean - std*2
        upper_bound = mean + std*2

        return {"lower_bound": lower_bound, "upper_bound": upper_bound}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove-outliers/")
async def remove_outliers(data: SingleColumnModel):
    try:
        session_id = data.session_id
        column_name = data.column_name

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Check if the specified column exists in the DataFrame
        if column_name not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' does not exist in the DataFrame")

        # Check if the column is numeric
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' is not numerical")

        # Remove outliers using the IQR method
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

        # Convert DataFrame back to JSON and store it in Redis
        df_json = df.to_json()
        redis_client.set(session_id, df_json)

        return {"message": f"Outliers removed from column '{column_name}'"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/perform-sentiment-analysis/")
async def perform_sentiment_analysis(data: SingleColumnModel):
    try:
        session_id = data.session_id
        column_name = data.column_name

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Check if the specified column exists in the DataFrame
        if column_name not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' does not exist in the DataFrame")

        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[column_name]):
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' is numerical")

        # Function to perform sentiment analysis using VADER
        def get_sentiment(row):
            score = analyzer.polarity_scores(row)
            neg = score['neg']
            neu = score['neu']
            pos = score['pos']
            if neg > pos:
                return "Negative"
            elif pos > neg:
                return "Positive"
            else:
                return "Neutral"

        # Normalize text data
        def clean_text(text):
            text = text.lower()  # Convert to lowercase
            text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
            text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
            return text

        # Apply text normalization and sentiment analysis to the specified column
        analyzer = SentimentIntensityAnalyzer()
        df[column_name] = df[column_name].apply(clean_text)
        df['sentiment'] = df[column_name].apply(get_sentiment)

        # Convert DataFrame back to JSON and store it in Redis
        df_json = df.to_json()
        redis_client.set(session_id, df_json)

        return { "status_code":200, "message": f"Sentiment analysis performed on column '{column_name}' and added as a new column 'sentiment' "}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/get-sentiment-counts/")
async def get_sentiment_counts(data: SingleColumnModel):
    try:
        session_id = data.session_id
        column_name = data.column_name

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Check if the specified columns exist in the DataFrame
        if column_name not in df.columns: 
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' does not exist in the DataFrame")
        elif 'sentiment' not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column 'sentiment' does not exist in the DataFrame")

        # Group by 'product_name' and 'sentiment', then count the occurrences of each sentiment category
        sentiment_counts = df.groupby([column_name, 'sentiment'])['sentiment'].count()

        # Unstack the grouped DataFrame to pivot 'sentiment' into columns
        sentiment_counts = sentiment_counts.unstack(level=-1, fill_value=0)

        # Rename the columns for clarity
        sentiment_counts.columns = ['Negative', 'Neutral', 'Positive']

        return {"sentiment_counts": sentiment_counts.to_dict()}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Function to extract aspects from review text
def extract_aspects(review_text):
    doc = nlp(review_text)
    aspects = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] or token.dep_ in ['amod', 'compound']]
    return aspects
analyzer = SentimentIntensityAnalyzer()

# Function to classify sentiment of an aspect
def classify_sentiment(aspect):
    score = analyzer.polarity_scores(aspect)
    sentiment_score = score['compound']
    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

@app.post("/extract-keyfeature/")
async def extract_keyfeature(data: SessionIDModel):
    try:
        session_id = data.session_id

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Function to process review text and add aspect column to DataFrame
        def process_reviews(df):
            df = df.copy()  # Make a copy of the DataFrame to avoid modifying the original DataFrame
            df['aspect'] = df['review_text'].apply(extract_aspects)
            df['keyfeature'] = df['aspect'].apply(lambda aspects: max(aspects, key=aspects.count) if aspects else None)
            df.drop(columns=['aspect'], inplace=True)  # Drop the 'aspect' column
            return df

        # Apply aspect extraction and sentiment analysis
        df_aspects = process_reviews(df)

        # Convert DataFrame back to JSON and store it in Redis
        df_json = df_aspects.to_json()
        redis_client.set(session_id, df_json)

        return {"status_code":200, "message": "Aspects extracted and sentiment analysis performed successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/keyfeature-improvements/")
async def keyfeature_improvements(data: SessionIDModel):
    try:
        session_id = data.session_id

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        negative_keyfeatures = df[df['sentiment'] == 'Negative']['keyfeature']

        keyfeature_counts = negative_keyfeatures.value_counts()

        filtered_keyfeature_counts = keyfeature_counts[keyfeature_counts > 10]

        return {"improvements": filtered_keyfeature_counts.to_dict()}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Function to categorize products
def categorize_product(name, conditions):
    name = name.lower()
    for category, rules in conditions.items():
        for rule in rules:
            if all(word in name for word in rule):
                return category
    return 'Other'

@app.post("/apply-conditions/")
async def apply_conditions(data: ConditionsModel):
    try:
        session_id = data.session_id
        resultant_col = data.resultant_col
        base_col = data.base_col
        conditions = data.conditions

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        if base_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column {base_col} does not exist in the DataFrame")
        
        # Apply categorization
        df[resultant_col] = df[base_col].apply(lambda x: categorize_product(x, conditions))

        # Convert DataFrame back to JSON and store it in Redis
        df_json = df.to_json()
        redis_client.set(session_id, df_json)

        return {"message": base_col+" categorized successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/drop-specific-rows/")
async def drop_specific_rows(data: SpecificColumnsModel):
    try:
        session_id = data.session_id
        column_value_pairs = data.column_value_pairs

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        data = StringIO(df_json)
        df = pd.read_json(data)

        # Drop rows where the specified column-value pairs match
        for pair in column_value_pairs:
            column = pair.column
            value = pair.value
            
            if column not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{column}' does not exist in the DataFrame")
            
            df = df[df[column] != value]

        # Convert DataFrame back to JSON and store it in Redis
        df_json = df.to_json()
        redis_client.set(session_id, df_json)

        return {"message": "Rows dropped successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/apply-regex/")
async def apply_regex(data: RegexModel):
    try:
        session_id = data.session_id
        extraction_patterns = data.extraction_patterns

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        df = pd.read_json(StringIO(df_json))

        # Apply the extraction patterns
        for pattern in extraction_patterns:
            df[pattern.new_column_name] = df[pattern.column_name].str.extract(pattern.pattern, expand=False)

        # Convert extracted date to datetime if any column is named 'date'
        for pattern in extraction_patterns:
            if 'date' in pattern.new_column_name.lower():
                df[pattern.new_column_name] = pd.to_datetime(df[pattern.new_column_name], errors='coerce')

        # Convert DataFrame back to JSON and store it in Redis
        df_json = df.to_json()
        redis_client.set(session_id, df_json)

        return {"message": "Regular expression applied successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/extract-year/")
async def extract_year(data: SingleColumnModel):
    
    try:
        session_id = data.session_id
        column_name = data.column_name

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No file uploaded with this session")

        # Decode the bytes object to string and read into DataFrame
        df_json = df_json.decode('utf-8')
        df = pd.read_json(StringIO(df_json))

        if column_name not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' does not exist in the DataFrame")
        
        if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' is not of the date type")

        df.loc[:, 'year'] = pd.to_datetime(df[column_name]).dt.year

        # Convert DataFrame back to JSON and store it in Redis
        df_json = df.to_json()
        redis_client.set(session_id, df_json)

        return {"message": "Year extracted successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/plot-chart/")
# async def plot_chart(request: ChartModel):
#     try:
#         session_id = request.session_id
#         chart_type = request.chart_type.lower()
#         x_column = request.x_column
#         y_column = request.y_column
#         category_column = request.category_column
#         group_by_columns = request.group_by_columns
#         stacked = request.stacked
#         x_label = request.x_label
#         y_label = request.y_label
#         title = request.title
        
#         # Retrieve the DataFrame from Redis using the session ID
#         df_json = redis_client.get(session_id)
#         if df_json is None:
#             raise HTTPException(status_code=400, detail="No data available for this session")

#         # Decode the bytes object to string and read into DataFrame
#         df_str = df_json.decode('utf-8')
#         df = pd.read_json(io.StringIO(df_str))

#         if category_column and group_by_columns!=[]:
#             sentiment_counts = df.groupby(group_by_columns)[category_column].count()
#         elif group_by_columns!=[]:
#             sentiment_counts = df.groupby(group_by_columns).size()
#         else:
#             sentiment_counts = df 

#         sentiment_counts = sentiment_counts.unstack(level=-1, fill_value=0)
#         custom_colors = ['#33DDFF','#F3F353','#F45288','#D06FE1']
        
#         fig, ax = plt.subplots()
#         # figsize=(14, 8)
#         ax.set_title(title)
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)
#         ax.tick_params(axis='x', rotation=45)
#         plt.legend(title=category_column, loc='upper right')
#         plt.tight_layout()

#         if chart_type == 'bar':
#             sentiment_counts.plot(kind='bar', stacked=stacked, ax=ax, color=custom_colors)
#         elif chart_type == 'heatmap':
#             heatmap_data = sentiment_counts.unstack().fillna(0)
#             sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='g', ax=ax)
#         elif chart_type == 'line':
#             if x_column and y_column:
#                 ax.plot(df[x_column], df[y_column])
#             else:
#                 raise HTTPException(status_code=400, detail="x_column and y_column are required for line chart")
#         elif chart_type == 'histogram':
#             if y_column:
#                 ax.hist(df[y_column], bins=30)
#             else:
#                 raise HTTPException(status_code=400, detail="y_column is required for histogram")
#         elif chart_type == 'scatter':
#             if x_column and y_column:
#                 ax.scatter(df[x_column], df[y_column])
#             else:
#                 raise HTTPException(status_code=400, detail="x_column and y_column are required for scatter plot")
#         elif chart_type == 'pie':
#             if category_column:
#                 sentiment_counts = df.groupby(category_column).size()
#                 ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
#             else:
#                 raise HTTPException(status_code=400, detail="category_column is required for pie chart")
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported chart type")

#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
#         plt.close(fig)

#         return StreamingResponse(buf, media_type='image/png')
#     except Exception as e:
#         return {"error": str(e)}

@app.post("/plot-chart/")
async def plot_chart(request: ChartModel):
    try:
        session_id = request.session_id
        chart_type = request.chart_type.lower()
        x_column = request.x_column
        y_column = request.y_column
        category_column = request.category_column
        group_by_columns = request.group_by_columns
        stacked = request.stacked
        x_label = request.x_label
        y_label = request.y_label
        title = request.title
        
        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No data available for this session")

        # Decode the bytes object to string and read into DataFrame
        df_str = df_json.decode('utf-8')
        df = pd.read_json(io.StringIO(df_str))

        if category_column and group_by_columns != []:
            sentiment_counts = df.groupby(group_by_columns)[category_column].count()
        elif group_by_columns != []:
            sentiment_counts = df.groupby(group_by_columns).size()
        else:
            sentiment_counts = df

        sentiment_counts = sentiment_counts.unstack(level=-1, fill_value=0)
        custom_colors = ['#33DDFF', '#F3F353', '#F45288', '#D06FE1']
        
        fig, ax = plt.subplots(figsize=(14, 8))  # Increase the figure size
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.tick_params(axis='x', rotation=45, labelsize=10)  # Rotate and set label size
        plt.legend(title=category_column, loc='upper right')

        if chart_type == 'bar':
            sentiment_counts.plot(kind='bar', stacked=stacked, ax=ax, color=custom_colors)
        elif chart_type == 'heatmap':
            heatmap_data = sentiment_counts.unstack().fillna(0)
            sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='g', ax=ax)
        elif chart_type == 'line':
            if x_column and y_column:
                ax.plot(df[x_column], df[y_column])
            else:
                raise HTTPException(status_code=400, detail="x_column and y_column are required for line chart")
        elif chart_type == 'histogram':
            if y_column:
                ax.hist(df[y_column], bins=30)
            else:
                raise HTTPException(status_code=400, detail="y_column is required for histogram")
        elif chart_type == 'scatter':
            if x_column and y_column:
                ax.scatter(df[x_column], df[y_column])
            else:
                raise HTTPException(status_code=400, detail="x_column and y_column are required for scatter plot")
        elif chart_type == 'pie':
            if category_column:
                sentiment_counts = df.groupby(category_column).size()
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
            else:
                raise HTTPException(status_code=400, detail="category_column is required for pie chart")
        else:
            raise HTTPException(status_code=400, detail="Unsupported chart type")

        plt.tight_layout()  # Adjust layout to fit everything
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return StreamingResponse(buf, media_type='image/png')
    except Exception as e:
        return {"error": str(e)}


# @app.post("/plot-chart/")
# async def plot_chart(request: ChartModel):
#     try:
#         print(request)
#         session_id = request.session_id
#         chart_type = request.chart_type.lower()
#         x_column = request.x_column
#         y_column = request.y_column
#         category_column = request.category_column
#         group_by_columns = request.group_by_columns
#         stacked = request.stacked
#         x_label = request.x_label
#         y_label = request.y_label
#         title = request.title
        
#         # Retrieve the DataFrame from Redis using the session ID
#         df_json = redis_client.get(session_id)
#         if df_json is None:
#             raise HTTPException(status_code=400, detail="No data available for this session")

#         # Decode the bytes object to string and read into DataFrame
#         df_str = df_json.decode('utf-8')
#         df = pd.read_json(io.StringIO(df_str))

#         if category_column:
#             sentiment_counts = df.groupby(group_by_columns)[category_column].count()
#         else:
#             sentiment_counts = df.groupby(group_by_columns).size()

#         # Unstack the grouped DataFrame to pivot 'sentiment' into columns
#         sentiment_counts = sentiment_counts.unstack(level=-1, fill_value=0)
#         custom_colors = ['#33DDFF','#F3F353','#F45288','#D06FE1']
        
#         # plt.figure()
#         plt.figure(figsize=(14, 8))
#         plt.title(title)
#         plt.xlabel(x_label)
#         plt.ylabel(y_label)
#         plt.xticks(rotation=45, ha='right')
#         plt.legend(title=category_column, loc='upper right')
#         plt.tight_layout()

#         if chart_type == 'bar':
#             sentiment_counts.plot(kind='bar', stacked=stacked, figsize=(14, 8), width=0.8, color=custom_colors)

#         elif chart_type == 'heatmap':
#             heatmap_data = sentiment_counts.unstack().fillna(0)

#             plt.figure(figsize=(14, 8))
#             sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='g')

#         elif chart_type == 'line':
#             if x_column and y_column:
#                 plt.plot(df[x_column], df[y_column])
#                 plt.xticks(rotation=20, ha='right') 
#             else:
#                 raise HTTPException(status_code=400, detail="x_column and y_column are required for line chart")

#         elif chart_type == 'histogram':
#             if y_column:
#                 plt.hist(df[y_column], bins=30)
#                 plt.xticks(rotation=20, ha='right') 
#             else:
#                 raise HTTPException(status_code=400, detail="y_column is required for histogram")

#         elif chart_type == 'scatter':
#             if x_column and y_column:
#                 plt.scatter(df[x_column], df[y_column])
#                 plt.xticks(rotation=20, ha='right') 
#             else:
#                 raise HTTPException(status_code=400, detail="x_column and y_column are required for scatter plot")

#         elif chart_type == 'pie':
#             if category_column:
#                 sentiment_counts = df.groupby(category_column).size()
#                 plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
#             else:
#                 raise HTTPException(status_code=400, detail="y_column and category_column are required for pie chart")
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported chart type")

#         # Save the plot to a bytes buffer
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
#         plt.close()

#         return StreamingResponse(buf, media_type='image/png')
#     except Exception as e:
#         return {"error": str(e)}




@app.post("/plot-multi-chart/")
async def plot_multi_chart(request: MultiChartModel):
    try:
        session_id = request.session_id
        chart_type = request.chart_type.lower()
        x_column = request.x_column
        y_column = request.y_column
        category_column = request.category_column
        group_by_columns = request.group_by_columns
        stacked = request.stacked
        multi_column = request.multi_column
        x_label = request.x_label
        y_label = request.y_label 
        title = request.title

        # Retrieve the DataFrame from Redis using the session ID
        df_json = redis_client.get(session_id)
        if df_json is None:
            raise HTTPException(status_code=400, detail="No data available for this session")

        # Decode the bytes object to string and read into DataFrame
        df_str = df_json.decode('utf-8')
        df = pd.read_json(io.StringIO(df_str))
        
        custom_colors = ['#33DDFF', '#F3F353', '#F45288', '#D06FE1']

        unique_values = df[multi_column].unique()
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
            for value in unique_values:
                data_subset = df[df[multi_column] == value]
                
                if chart_type == 'bar':
                    if category_column:
                        sentiment_counts = data_subset.groupby(group_by_columns)[category_column].count().unstack(fill_value=0)
                    else:
                        sentiment_counts = data_subset.groupby(group_by_columns).size().unstack(fill_value=0)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sentiment_counts.plot(kind='bar', stacked=stacked, ax=ax, figsize=(14, 8), width=0.8, color=custom_colors)

                    plt.title(title)
                    plt.xlabel(x_label)
                    plt.ylabel(y_label)
                    plt.xticks(rotation=45, ha='right')
                    plt.legend(title=category_column, loc='upper right')
                    plt.tight_layout()

                elif chart_type == 'heatmap':
                    sentiment_counts = data_subset.groupby(group_by_columns).size().unstack(fill_value=0)

                    plt.figure(figsize=(14, 8))
                    sns.heatmap(sentiment_counts, annot=True, cmap='YlGnBu', fmt='g')

                    plt.title(title)
                    plt.xlabel(x_label)
                    plt.ylabel(y_label)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                elif chart_type == 'line':
                    if x_column and y_column:
                        plt.plot(data_subset[x_column], data_subset[y_column])
                        plt.xticks(rotation=20, ha='right')
                        plt.title(title)
                        plt.xlabel(x_label)
                        plt.ylabel(y_label)
                        plt.tight_layout()
                    else:
                        raise HTTPException(status_code=400, detail="x_column and y_column are required for line chart")

                elif chart_type == 'histogram':
                    if y_column:
                        plt.hist(data_subset[y_column], bins=30)
                        plt.xticks(rotation=20, ha='right')
                        plt.title(title)
                        plt.xlabel(y_label)
                        plt.ylabel('Frequency')
                        plt.tight_layout()
                    else:
                        raise HTTPException(status_code=400, detail="y_column is required for histogram")

                elif chart_type == 'scatter':
                    if x_column and y_column:
                        plt.scatter(data_subset[x_column], data_subset[y_column])
                        plt.xticks(rotation=20, ha='right')
                        plt.title(title)
                        plt.xlabel(x_label)
                        plt.ylabel(y_label)
                        plt.tight_layout()
                    else:
                        raise HTTPException(status_code=400, detail="x_column and y_column are required for scatter plot")

                elif chart_type == 'pie':
                    if category_column:
                        sentiment_counts = data_subset.groupby(group_by_columns).size().unstack(fill_value=0)
                        
                        for cat in sentiment_counts.index:
                            plt.figure(figsize=(6, 6))
                            sentiment_counts.loc[cat].plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=custom_colors)
                            plt.title(title)
                            plt.ylabel('')  # Hide y-label for better appearance
                            plt.tight_layout()

                            # Save the plot to a bytes buffer
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png')
                            buf.seek(0)
                            plt.close()

                            # Add the buffer to the zip file
                            zip_file.writestr(f'{value}_{cat}.png', buf.getvalue())

                        continue  # Skip the final save and add below for pie chart

                # Save the plot to a bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close()

                # Add the buffer to the zip file
                zip_file.writestr(f'{value}.png', buf.getvalue())

        zip_buffer.seek(0)
        return StreamingResponse(zip_buffer, media_type='application/zip', headers={"Content-Disposition": "attachment;filename=charts.zip"})
    except Exception as e:
        return {"error": str(e)}

# @app.post("/plot-multi-chart/")
# async def plot_multi_chart(request: MultiChartModel):
#     try:
#         session_id = request.session_id
#         chart_type = request.chart_type.lower()
#         x_column = request.x_column
#         y_column = request.y_column
#         category_column = request.category_column
#         group_by_columns = request.group_by_columns
#         stacked = request.stacked
#         multi_column = request.multi_column
        
#         # Retrieve the DataFrame from Redis using the session ID
#         df_json = redis_client.get(session_id)
#         if df_json is None:
#             raise HTTPException(status_code=400, detail="No data available for this session")

#         # Decode the bytes object to string and read into DataFrame
#         df_str = df_json.decode('utf-8')
#         df = pd.read_json(io.StringIO(df_str))
        
#         custom_colors = ['#33DDFF','#F3F353','#F45288','#D06FE1']

#         unique_values = df[multi_column].unique()
#         zip_buffer = io.BytesIO()

#         with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
#             for value in unique_values:
#                 data_subset = df[df[multi_column] == value]
                
#                 if chart_type == 'bar':
#                     if category_column:
#                         sentiment_counts = data_subset.groupby(group_by_columns)[category_column].count().unstack(fill_value=0)
#                     else:
#                         sentiment_counts = data_subset.groupby(group_by_columns).size().unstack(fill_value=0)
                    
#                     fig, ax = plt.subplots(figsize=(10, 6))
#                     sentiment_counts.plot(kind='bar', stacked=stacked, ax=ax, figsize=(14, 8), width=0.8, color=custom_colors)

#                     plt.title(f'{chart_type.capitalize()} Chart for {multi_column.capitalize()} = {value}')
#                     plt.xlabel(x_column.capitalize())
#                     plt.ylabel('Count of Sentiments')
#                     plt.xticks(rotation=45, ha='right')
#                     plt.legend(title='Sentiment', loc='upper right')
#                     plt.tight_layout()

#                 elif chart_type == 'heatmap':
#                     sentiment_counts = data_subset.groupby(group_by_columns).size().unstack(fill_value=0)

#                     plt.figure(figsize=(14, 8))
#                     sns.heatmap(sentiment_counts, annot=True, cmap='YlGnBu', fmt='g')

#                     plt.title(f'Heatmap for {multi_column.capitalize()} = {value}')
#                     plt.xlabel(x_column.capitalize())
#                     plt.ylabel('Sentiment')
#                     plt.xticks(rotation=45)
#                     plt.tight_layout()
                    
#                 elif chart_type == 'line':
#                     if x_column and y_column:
#                         plt.plot(data_subset[x_column], data_subset[y_column])
#                         plt.xticks(rotation=20, ha='right')
#                         plt.title(f'{chart_type.capitalize()} Chart for {multi_column.capitalize()} = {value}')
#                         plt.xlabel(x_column.capitalize())
#                         plt.ylabel(y_column.capitalize())
#                         plt.tight_layout()
#                     else:
#                         raise HTTPException(status_code=400, detail="x_column and y_column are required for line chart")

#                 elif chart_type == 'histogram':
#                     if y_column:
#                         plt.hist(data_subset[y_column], bins=30)
#                         plt.xticks(rotation=20, ha='right')
#                         plt.title(f'{chart_type.capitalize()} Chart for {multi_column.capitalize()} = {value}')
#                         plt.xlabel(y_column.capitalize())
#                         plt.ylabel('Frequency')
#                         plt.tight_layout()
#                     else:
#                         raise HTTPException(status_code=400, detail="y_column is required for histogram")

#                 elif chart_type == 'scatter':
#                     if x_column and y_column:
#                         plt.scatter(data_subset[x_column], data_subset[y_column])
#                         plt.xticks(rotation=20, ha='right')
#                         plt.title(f'{chart_type.capitalize()} Chart for {multi_column.capitalize()} = {value}')
#                         plt.xlabel(x_column.capitalize())
#                         plt.ylabel(y_column.capitalize())
#                         plt.tight_layout()
#                     else:
#                         raise HTTPException(status_code=400, detail="x_column and y_column are required for scatter plot")

#                 elif chart_type == 'pie':
#                     if category_column and y_column:
#                         sentiment_counts = df.groupby(group_by_columns)[category_column].count()
#                         plt.figure(figsize=(6, 6))
#                         sentiment_counts.loc[category_column].plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#33DDFF','#F3F353','#D06FE1'])
#                         plt.title(f'Sentiment Distribution for {category_column}')
#                         plt.ylabel('')  # Hide y-label for better appearance
#                         # sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=custom_colors)
#                         # plt.title(f'Sentiment Distribution for {category_column.capitalize()} = {value}')
#                         # plt.ylabel('')  # Hide y-label for better appearance
#                         plt.tight_layout()
#                     else:
#                         raise HTTPException(status_code=400, detail="category_column and y_column are required for pie chart")

#                 # Save the plot to a bytes buffer
#                 buf = io.BytesIO()
#                 plt.savefig(buf, format='png')
#                 buf.seek(0)
#                 plt.close()

#                 # Add the buffer to the zip file
#                 zip_file.writestr(f'{value}.png', buf.getvalue())

#         zip_buffer.seek(0)
#         return StreamingResponse(zip_buffer, media_type='application/zip', headers={"Content-Disposition": "attachment;filename=charts.zip"})
#     except Exception as e:
#         return {"error": str(e)}


# @app.post("/plot-multi-chart/")
# async def plot_multi_chart(request: MultiChartModel):
#     try:
#         session_id = request.session_id
#         chart_type = request.chart_type.lower()
#         x_column = request.x_column
#         y_column = request.y_column
#         category_column = request.category_column
#         group_by_columns = request.group_by_columns
#         stacked = request.stacked
#         multi_column = request.multi_column
        
#         # Retrieve the DataFrame from Redis using the session ID
#         df_json = redis_client.get(session_id)
#         if df_json is None:
#             raise HTTPException(status_code=400, detail="No data available for this session")

#         # Decode the bytes object to string and read into DataFrame
#         df_str = df_json.decode('utf-8')
#         df = pd.read_json(io.StringIO(df_str))
        
#         custom_colors = ['#33DDFF', '#F3F353', '#F45288', '#D06FE1']

#         unique_values = df[multi_column].unique()
#         zip_buffer = io.BytesIO()

#         with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
#             for value in unique_values:
#                 data_subset = df[df[multi_column] == value]

#                 if chart_type == 'bar':
#                     if category_column:
#                         sentiment_counts = data_subset.groupby(group_by_columns)[category_column].count().unstack(fill_value=0)
#                     else:
#                         sentiment_counts = data_subset.groupby(group_by_columns).size().unstack(fill_value=0)
                    
#                     fig, ax = plt.subplots(figsize=(10, 6))
#                     sentiment_counts.plot(kind='bar', stacked=stacked, ax=ax, figsize=(14, 8), width=0.8, color=custom_colors)

#                     plt.title(f'{chart_type.capitalize()} Chart for {multi_column.capitalize()} = {value}')
#                     plt.xlabel(x_column.capitalize())
#                     plt.ylabel('Count of Sentiments')
#                     plt.xticks(rotation=45, ha='right')
#                     plt.legend(title='Sentiment', loc='upper right')
#                     plt.tight_layout()

#                 elif chart_type == 'heatmap':
#                     sentiment_counts = data_subset.groupby(group_by_columns).size().unstack(fill_value=0)

#                     plt.figure(figsize=(14, 8))
#                     sns.heatmap(sentiment_counts, annot=True, cmap='YlGnBu', fmt='g')

#                     plt.title(f'Heatmap for {multi_column.capitalize()} = {value}')
#                     plt.xlabel(x_column.capitalize())
#                     plt.ylabel('Sentiment')
#                     plt.xticks(rotation=45)
#                     plt.tight_layout()

#                 elif chart_type == 'line':
#                     if x_column and y_column:
#                         plt.figure(figsize=(14, 8))
#                         plt.plot(data_subset[x_column], data_subset[y_column])
#                         plt.title(f'Line Chart for {multi_column.capitalize()} = {value}')
#                         plt.xlabel(x_column.capitalize())
#                         plt.ylabel(y_column.capitalize())
#                         plt.xticks(rotation=20, ha='right')
#                         plt.tight_layout()

#                 elif chart_type == 'histogram':
#                     if y_column:
#                         plt.figure(figsize=(14, 8))
#                         plt.hist(data_subset[y_column], bins=30, color=custom_colors[0])
#                         plt.title(f'Histogram for {multi_column.capitalize()} = {value}')
#                         plt.xlabel(y_column.capitalize())
#                         plt.ylabel('Frequency')
#                         plt.xticks(rotation=20, ha='right')
#                         plt.tight_layout()

#                 elif chart_type == 'scatter':
#                     if x_column and y_column:
#                         plt.figure(figsize=(14, 8))
#                         plt.scatter(data_subset[x_column], data_subset[y_column], color=custom_colors[0])
#                         plt.title(f'Scatter Plot for {multi_column.capitalize()} = {value}')
#                         plt.xlabel(x_column.capitalize())
#                         plt.ylabel(y_column.capitalize())
#                         plt.xticks(rotation=20, ha='right')
#                         plt.tight_layout()

#                 elif chart_type == 'pie':
#                     if category_column and y_column:
#                         # Group by 'category' and 'sentiment', then count the occurrences of each sentiment category
#                         sentiment_counts = data_subset.groupby([category_column, 'sentiment'])[y_column].count()

#                         # Unstack the grouped DataFrame to pivot 'sentiment' into columns
#                         sentiment_counts = sentiment_counts.unstack(level=-1, fill_value=0)

#                         for category in sentiment_counts.index:
#                             plt.figure(figsize=(3, 3))
#                             sentiment_counts.loc[category].plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=custom_colors)
#                             plt.title(f'Sentiment Distribution for {category}')
#                             plt.ylabel('')  # Hide y-label for better appearance
#                             plt.tight_layout()

#                             # Save the plot to a bytes buffer
#                             buf = io.BytesIO()
#                             plt.savefig(buf, format='png')
#                             buf.seek(0)
#                             plt.close()

#                             # Add the buffer to the zip file
#                             zip_file.writestr(f'{value}_{category}.png', buf.getvalue())
#                         continue  # Skip the final save step for pie charts as it is done inside the loop

#                 # Save the plot to a bytes buffer for other chart types
#                 buf = io.BytesIO()
#                 plt.savefig(buf, format='png')
#                 buf.seek(0)
#                 plt.close()

#                 # Add the buffer to the zip file
#                 zip_file.writestr(f'{value}.png', buf.getvalue())

#         zip_buffer.seek(0)
#         return StreamingResponse(zip_buffer, media_type='application/zip', headers={"Content-Disposition": "attachment;filename=charts.zip"})
#     except Exception as e:
#         return {"error": str(e)}


# @app.post("/plot-multi-chart/")
# async def plot_multi_chart(request: MultiChartModel):
#     try:
#         session_id = request.session_id
#         chart_type = request.chart_type.lower()
#         x_column = request.x_column
#         y_column = request.y_column
#         category_column = request.category_column
#         group_by_columns = request.group_by_columns
#         stacked = request.stacked
#         multi_column = request.multi_column
        
#         # Retrieve the DataFrame from Redis using the session ID
#         df_json = redis_client.get(session_id)
#         if df_json is None:
#             raise HTTPException(status_code=400, detail="No data available for this session")

#         # Decode the bytes object to string and read into DataFrame
#         df_str = df_json.decode('utf-8')
#         df = pd.read_json(io.StringIO(df_str))
        
#         custom_colors = ['#33DDFF','#F3F353','#F45288','#D06FE1']

#         unique_values = df[multi_column].unique()
#         zip_buffer = io.BytesIO()

#         with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
#             for value in unique_values:
#                 data_subset = df[df[multi_column] == value]
                
#                 if chart_type == 'bar':
#                     if category_column:
#                         sentiment_counts = data_subset.groupby(group_by_columns)[category_column].count().unstack(fill_value=0)
#                     else:
#                         sentiment_counts = data_subset.groupby(group_by_columns).size().unstack(fill_value=0)
                    
#                     fig, ax = plt.subplots(figsize=(10, 6))
#                     sentiment_counts.plot(kind='bar', stacked=stacked, ax=ax, figsize=(14, 8), width=0.8, color=custom_colors)

#                     plt.title(f'{chart_type.capitalize()} Chart for {multi_column.capitalize()} = {value}')
#                     plt.xlabel(x_column.capitalize())
#                     plt.ylabel('Count of Sentiments')
#                     plt.xticks(rotation=45, ha='right')
#                     plt.legend(title='Sentiment', loc='upper right')
#                     plt.tight_layout()

#                 elif chart_type == 'heatmap':
#                     sentiment_counts = data_subset.groupby(group_by_columns).size().unstack(fill_value=0)

#                     plt.figure(figsize=(14, 8))
#                     sns.heatmap(sentiment_counts, annot=True, cmap='YlGnBu', fmt='g')

#                     plt.title(f'Heatmap for {multi_column.capitalize()} = {value}')
#                     plt.xlabel(x_column.capitalize())
#                     plt.ylabel('Sentiment')
#                     plt.xticks(rotation=45)
#                     plt.tight_layout()

#                 # Save the plot to a bytes buffer
#                 buf = io.BytesIO()
#                 plt.savefig(buf, format='png')
#                 buf.seek(0)
#                 plt.close()

#                 # Add the buffer to the zip file
#                 zip_file.writestr(f'{value}.png', buf.getvalue())

#         zip_buffer.seek(0)
#         return StreamingResponse(zip_buffer, media_type='application/zip', headers={"Content-Disposition": "attachment;filename=charts.zip"})
#     except Exception as e:
#         return {"error": str(e)}

# yet to test in postman
@app.delete("/clear-redis/")
async def clear_redis():
    try:
        redis_client.flushdb()  # This clears the current database
        return {"message": "Redis database cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# yet to test in postman
@app.delete("/delete-key/{key}")
async def delete_key(key: str):
    try:
        result = redis_client.delete(key)
        if result == 1:
            return {"status_code":200, "message": f"Key '{key}' deleted"}
        else:
            return {"message": f"Key '{key}' not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))