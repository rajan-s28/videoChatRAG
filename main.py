import re
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import logging
from pydantic import BaseModel
from embedding import check_simpler, embeddings_with_timestamps, search_question_in_faiss, chunk_text_by_tokens, create_openai_embeddings, DEFAULT_EMBEDDING_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YouTube Caption Extractor",
    description="An API to extract captions from YouTube videos.",
    version="1.0.0"
)

# Mount static files (optional, if you have separate CSS/JS)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates directory
templates = Jinja2Templates(directory="templates")

class VideoURL(BaseModel):
    url: str

def extract_video_id(url: str):
    """
    Extracts the YouTube video ID from a URL.
    Supports various YouTube URL formats.
    """
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/extract_captions", response_class=JSONResponse)
async def get_captions_api(video_data: VideoURL):
    """
    API endpoint to extract captions from a YouTube video URL.
    Expects a JSON payload with a "url" field.
    """
    video_url = video_data.url
    if not video_url:
        raise HTTPException(status_code=400, detail="Video URL is required.")

    video_id = extract_video_id(video_url)
    if not video_id:
        logger.error(f"Invalid YouTube URL or could not extract video ID: {video_url}")
        raise HTTPException(status_code=400, detail="Invalid YouTube URL or could not extract video ID.")

    logger.info(f"Extracted video ID: {video_id} for URL: {video_url}")

    try:
        # Attempt to fetch English transcript first, then any available
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        transcript = None
        try:
            # Try for manually created English transcript
            transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
            logger.info(f"Found manually created English transcript for {video_id}")
        except NoTranscriptFound:
            logger.info(f"No manual English transcript for {video_id}. Trying generated English transcript.")
            try:
                # Try for generated English transcript
                transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
                logger.info(f"Found generated English transcript for {video_id}")
            except NoTranscriptFound:
                logger.info(f"No generated English transcript for {video_id}. Trying any available transcript.")
                # If English not found, try to get any available transcript
                # This part iterates through available transcripts and picks the first one.
                # You might want to add logic to prefer certain languages if multiple are available.
                available_langs = [t.language_code for t in transcript_list]
                if not available_langs:
                    raise NoTranscriptFound(video_id, available_langs, "No transcripts available at all.")
                
                logger.info(f"Available languages for {video_id}: {available_langs}. Fetching first one: {available_langs[0]}")
                transcript = transcript_list.find_transcript([available_langs[0]])


        fetched_transcript = transcript.fetch()
        # Iterate through each segment of the fetched transcript to format the output
        detailed_captions = {}
        for segment in fetched_transcript:
            caption_text = segment['text']
            start_timestamp = segment['start']
            # Calculate the end timestamp by adding duration to the start timestamp
            end_timestamp = segment['start'] + segment['duration']
            
            # Store the data in the requested format:
            # key: caption_text
            # value: list [start_timestamp, end_timestamp, video_id, language]
            detailed_captions[caption_text] = [
                start_timestamp,
                end_timestamp,
                video_id,
                transcript.language
            ]
        
        captions_text = "\t".join([caption_text for caption_text in detailed_captions])
        cap_list = detailed_captions.keys()
        
        logger.info(f"Successfully extracted captions for video ID: {video_id}")
        chunks = check_simpler(captions_text)
        embeddings_with_timestamps(cap_list, detailed_captions, video_id)
        return {"captions": detailed_captions, "video_id": video_id, "language": transcript.language}

    except TranscriptsDisabled as e:
        logger.error(f"My error {e}")
        logger.error(f"lafdaa error ")
        logger.error(f"Transcripts are disabled for video ID: {video_id}")
        raise HTTPException(status_code=404, detail="Transcripts are disabled for this video.")
    except NoTranscriptFound:
        logger.error(f"No transcript found for video ID: {video_id}")
        raise HTTPException(status_code=404, detail="No transcript found for this video in the preferred languages or any language.")
    except VideoUnavailable:
        logger.error(f"Video unavailable for video ID: {video_id}")
        raise HTTPException(status_code=404, detail="The video is unavailable (e.g., private, deleted).")
    except Exception as e:
        logger.error(f"An unexpected error occurred for video ID {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

def generate_youtube_clip_links(video_id: str, matches: list[tuple[str, float, float, float]]) -> list[dict]:
    """
    Given a video_id and a list of matched tuples (text, start_time, end_time, score),
    return a list of dicts with YouTube clip URLs, timestamps, and metadata.
    """
    results = []
    for text, start_time, end_time, score in matches:
        if int(start_time)-3 > 0:
            start_seconds = int(start_time)-3
        else:
            start_seconds = int(start_time) 
        url = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}s"
        results.append({
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "score": score,
            "url": url
        })
    return results

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask_question", response_class=JSONResponse)
async def ask_question(payload: QuestionRequest, request: Request):
    """
    Receives a user-entered question from the frontend via JSON.
    """
    question = payload.question
    logger.info(f"Received question ########################: {question}")
    
    # Replace this with your actual retrieval logic
    result = search_question_in_faiss(question)
    logger.info(f"Received Result ########################: {result}")

    video_id = result[0][4] if result else ""
    matches = [(match[0], match[1], match[2], match[3]) for match in result]  # text, start, end, score

    enriched_result = generate_youtube_clip_links(video_id, matches)
    logger.info(f"#######################Results URLS####################### \n: {enriched_result}")
    
    return {"question": question, "results": enriched_result}

if __name__ == "__main__":
    import uvicorn
    # This is for local development. For deployment, use a proper ASGI server like Uvicorn or Hypercorn.
    # Example: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)