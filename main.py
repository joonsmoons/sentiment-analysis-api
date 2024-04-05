from fastapi import FastAPI, HTTPException
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

sentiment_pipeline = pipeline(
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/sentiment")
async def analyze_sentiment(q: str):
    """
    Analyzes sentiment of the given text.

    Parameters:
    - q (str): Text to analyze sentiment for.

    Returns:
    - dict: Sentiment analysis result.
    """
    try:
        return sentiment_pipeline(q)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
