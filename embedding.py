# embedding.py
import os
import tiktoken
from openai import OpenAI
from fuzzywuzzy import fuzz
import faiss
import numpy as np
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# It's highly recommended to use environment variables for API keys
# For example: OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# For this example, we'll pass it as an argument, but remind the user.

# Define token limits for common models (it's good practice to leave some buffer)
MODEL_TOKEN_LIMITS = {
    "text-embedding-ada-002": 8191,
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    
}
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002" # Or choose text-embedding-3-small for a good balance

def get_tokenizer(model_name: str):
    """Gets the tokenizer for a given OpenAI model."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"Warning: Model {model_name} not found. Using cl100k_base encoding.")
        return tiktoken.get_encoding("cl100k_base")

def search_question_in_faiss(question: str, top_k: int = 2):
    """
    Embed the question and search the FAISS vector DB.
    Returns list of (text, start_time, end_time, score) tuples.
    """
    # Step 1: Load embedding model and create question embedding
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    question_embedding = embedder.embed_query(question)
    question_embedding = np.array([question_embedding]).astype("float32")
    faiss.normalize_L2(question_embedding)

    # Step 2: Load FAISS index and metadata
    index = faiss.read_index("video_index.faiss")
    with open("video_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Step 3: Search for nearest neighbors
    scores, indices = index.search(question_embedding, top_k)  # shape: (1, top_k)

    # Step 4: Prepare results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        results.append((meta["text"], meta["start"], meta["end"], float(scores[0][i]), meta["video_id"]))

    return results

def store_in_faiss(embedding_data, video_id):
    """
    embedding_data format:
    {
        chunk_text: [embedding_vector, start_time, end_time],
        ...
    }
    """

    # Step 1: Prepare data
    texts = list(embedding_data.keys())
    vectors = [embedding_data[text][0] for text in texts]
    metadata = [
        {"text": text, "start": embedding_data[text][1], "end": embedding_data[text][2], "video_id": video_id}
        for text in texts
    ]

    # Step 2: Convert to numpy array (float32)
    embeddings_np = np.array(vectors).astype("float32")

    # Step 3: Create FAISS index
    dimension = embeddings_np.shape[1]  # e.g., 384 for MiniLM
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_np)     # before adding to FAISS # L2 distance (you can also use cosine)
    index.add(embeddings_np)

    # Step 4: Save index and metadata
    faiss.write_index(index, "video_index.faiss")

    with open("video_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Stored in FAISS and saved metadata.")


def generate_embeddings_with_timestamps(chunk_timestamp_dict):
    # Load the local embedding model
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Extract text chunks in order
    chunks = list(chunk_timestamp_dict.keys())

    # Compute embeddings
    embeddings = embedder.embed_documents(chunks)

    # Build new dictionary with embedding and timestamps
    output = {}
    for i, chunk in enumerate(chunks):
        timestamp = chunk_timestamp_dict[chunk]  # [start, end]
        output[chunk] = [embeddings[i], timestamp[0], timestamp[1]]

    return output

def fuzzy_match_sentences(chunk, detailed_captions) -> dict:
    """
    Compare two sentences using fuzzy logic and return match score and label.
    
    Args:
        s1 (str): First sentence
        s2 (str): Second sentence
        threshold (int): Minimum similarity score to consider it a match
    
    Returns:
        dict: {
            "score": int,
            "label": str,
            "is_match": bool
        }
    """
    best_match = ""
    max_score = 0
    for caption, metadata in detailed_captions.items():
        score = fuzz.token_set_ratio(chunk, caption)
        # Label based on score
        if score> max_score:
            max_score = score
            best_match = caption
            start_time, end_time = metadata[0], metadata[1]


    return best_match, start_time, end_time

def embeddings_with_timestamps(chunks, detailed_captions, video_id):

    detailed_embeddings = {}
    for item in chunks:
        caption, start, end = fuzzy_match_sentences(item, detailed_captions)
        detailed_embeddings[item] = [start, end]
    output = generate_embeddings_with_timestamps(detailed_embeddings)
    store_in_faiss(output, video_id)
    # return detailed_embeddings


def check_simpler(text):
    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=12,
        chunk_overlap=3,
        separators=["\n\n", "\n", " ", ""]
    )

    final_chunks = token_splitter.split_text(text)

    return final_chunks

def chunk_text_by_tokens(
    text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    max_tokens_per_chunk: int = 2000, # A practical limit, well below the model's max
    overlap_tokens: int = 100 # Number of tokens to overlap between chunks
) -> list[str]:
    """
    Chunks text into smaller pieces based on token count with overlap.
    """
    if max_tokens_per_chunk <= overlap_tokens and overlap_tokens > 0:
        raise ValueError("max_tokens_per_chunk must be greater than overlap_tokens.")

    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer.encode(text)
    
    if not tokens:
        return []

    if len(tokens) <= max_tokens_per_chunk:
        return [text]

    chunks = []
    current_pos = 0
    while current_pos < len(tokens):
        end_pos = min(current_pos + max_tokens_per_chunk, len(tokens))
        chunk_tokens = tokens[current_pos:end_pos]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if end_pos == len(tokens): # Reached the end
            break
            
        current_pos += (max_tokens_per_chunk - overlap_tokens)
        if current_pos >= len(tokens): # Ensure we don't go past the end with the overlap logic
             # This can happen if the remaining text is smaller than overlap_tokens
             # Add the last bit if it wasn't fully covered
            last_chunk_tokens = tokens[current_pos - (max_tokens_per_chunk - overlap_tokens) + max_tokens_per_chunk - overlap_tokens:]
            if last_chunk_tokens and tokenizer.decode(last_chunk_tokens) not in chunks[-1]: # Avoid duplicate if last chunk was small
                final_small_chunk = tokens[current_pos - (max_tokens_per_chunk - overlap_tokens) :]
                if final_small_chunk: # Check if there are any tokens left
                    chunks.append(tokenizer.decode(final_small_chunk))
            break


    # A simple post-processing step to ensure no chunk exceeds the absolute model limit
    # This is a fallback, the primary chunking logic should prevent this.
    model_absolute_max = MODEL_TOKEN_LIMITS.get(model_name, 8191)
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = tokenizer.encode(chunk)
        if len(chunk_tokens) > model_absolute_max:
            # This chunk is too big, needs further splitting (simplified here)
            # For a robust solution, you might recursively call chunk_text or use a more granular splitter
            print(f"Warning: A chunk was too large ({len(chunk_tokens)} tokens) and is being truncated. Consider refining chunking strategy.")
            truncated_tokens = chunk_tokens[:model_absolute_max]
            final_chunks.append(tokenizer.decode(truncated_tokens))
        else:
            final_chunks.append(chunk)
            
    return final_chunks


async def create_openai_embeddings(
    api_key: str,
    text_chunks: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL
) -> list[list[float]] | None:
    """
    Creates embeddings for a list of text chunks using OpenAI API.
    Returns a list of embedding vectors, or None if an error occurs.
    """
    if not api_key:
        print("Error: OpenAI API key is missing.")
        return None
    if not text_chunks:
        print("Warning: No text chunks provided for embedding.")
        return []

    try:
        client = OpenAI(api_key=api_key)
        embeddings_response = client.embeddings.create(
            input=text_chunks,
            model=model
        )
        
        embeddings = [item.embedding for item in embeddings_response.data]
        return embeddings
    except Exception as e:
        print(f"Error creating OpenAI embeddings: {e}")
        return None

if __name__ == '__main__':
    # Example Usage (requires OPENAI_API_KEY environment variable or direct assignment)
    sample_api_key = os.environ.get("OPENAI_API_KEY") # Replace with your actual key if not using env var
    if not sample_api_key:
        print("Please set the OPENAI_API_KEY environment variable or provide it directly in the code for this example.")
    else:
        sample_text = (
            "This is the first paragraph. It contains some interesting information. " * 50 +
            "This is the second paragraph, slightly longer and with more details. " * 70 +
            "A third paragraph to ensure we have enough text to warrant chunking. " * 60 +
            "And a final short sentence."
        )
        print(f"Original text length (chars): {len(sample_text)}")

        tokenizer = get_tokenizer(DEFAULT_EMBEDDING_MODEL)
        original_tokens = tokenizer.encode(sample_text)
        print(f"Original text length (tokens): {len(original_tokens)}")
        
        # Using a smaller max_tokens_per_chunk for demonstration to ensure chunking happens
        chunks = chunk_text_by_tokens(sample_text, model_name=DEFAULT_EMBEDDING_MODEL, max_tokens_per_chunk=100, overlap_tokens=20)
        print(f"\nNumber of chunks created: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            chunk_tok_count = len(tokenizer.encode(chunk))
            print(f"Chunk {i+1} (tokens: {chunk_tok_count}):\n'{chunk[:100]}...'\n")

        if chunks:
            print("Attempting to create embeddings...")
            # In a real app, you'd get the API key securely
            embeddings_result = asyncio.run(create_openai_embeddings(api_key=sample_api_key, text_chunks=chunks))
            
            if embeddings_result:
                print(f"\nSuccessfully created {len(embeddings_result)} embeddings.")
                print(f"Dimension of first embedding: {len(embeddings_result[0])}")
                # print(f"First embedding vector (first 5 values): {embeddings_result[0][:5]}")
            else:
                print("\nFailed to create embeddings.")
