import os
import torch
import numpy as np
import pandas as pd
import cv2
from torchvision import transforms
import torch.nn as nn
import math
import glob
import timm
from collections import Counter
import tempfile
import shutil
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================================
#   1. ARCHITECTURE DEFINITIONS
# =====================================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerCaptioning(nn.Module):
    def __init__(self, feature_dim, vocab_size, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1, max_len=100):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, video_features, captions):
        video_embed = self.pos_encoder(self.feature_proj(video_features))
        memory = self.encoder(video_embed)
        caption_embed = self.pos_encoder(self.embedding(captions))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(caption_embed.size(1)).to(caption_embed.device)
        output = self.decoder(tgt=caption_embed, memory=memory, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

# =====================================================================================
#   2. RESPONSE MODELS
# =====================================================================================

class PredictionResponse(BaseModel):
    success: bool
    predicted_text: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    session_id: Optional[str] = None  # Added session_id field

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

# =====================================================================================
#   3. SIGN LANGUAGE RECOGNIZER SERVICE
# =====================================================================================

class SignLanguageService:
    def __init__(self):
        self.MODEL_PATH = "best_transformer_model.pth"
        self.MAX_SEQ_LENGTH = 80
        self.FEATURE_DIM = 512
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.DEVICE}")
        
        # Load vocabulary and model
        self.caption_to_index, self.index_to_caption = self.load_vocabulary()
        vocab_size = len(self.caption_to_index)
        
        logger.info(f"Loading model from {self.MODEL_PATH}...")
        self.model = self.load_inference_model(self.MODEL_PATH, vocab_size, self.DEVICE)
        
        # Initialize feature extractor
        logger.info("Initializing feature extractor (ResNet18)...")
        self.feature_extractor = timm.create_model('resnet18', pretrained=True, num_classes=0)
        self.feature_extractor = self.feature_extractor.to(self.DEVICE)
        self.feature_extractor.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Service initialized successfully")
    
    def load_vocabulary(self, vocabulary_csv_path="vocabulary.csv", dataset_csv_dir="file_csv"):
        """Load vocabulary from CSV files"""
        logger.info(f"Building vocabulary from source CSVs...")
        all_combined_tokens = []

        # Process main vocabulary file
        try:
            df_vocab = pd.read_csv(vocabulary_csv_path)
            if 'Sentence' in df_vocab.columns:
                vocab_sentences = df_vocab['Sentence'].dropna().astype(str).tolist()
                for sentence in vocab_sentences:
                    all_combined_tokens.extend(sentence.strip().split())
        except FileNotFoundError:
            logger.warning(f"Warning: {vocabulary_csv_path} not found. Skipping.")

        # Process dataset caption files
        dataset_csv_files = glob.glob(os.path.join(dataset_csv_dir, "*.csv"))
        if not dataset_csv_files:
            logger.warning(f"Warning: No caption CSV files found in '{dataset_csv_dir}'.")
        
        for csv_file in dataset_csv_files:
            try:
                df_dataset = pd.read_csv(csv_file)
                if '7' in df_dataset.columns:
                    valid_captions = df_dataset['7'].dropna().astype(str).tolist()
                    for caption in valid_captions:
                        all_combined_tokens.extend(caption.strip().split())
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")

        token_freq = Counter(all_combined_tokens)
        SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        unique_words = sorted([word for word in token_freq.keys() if word])
        vocab = SPECIAL_TOKENS + unique_words
        
        caption_to_index = {token: idx for idx, token in enumerate(vocab)}
        index_to_caption = {idx: token for token, idx in caption_to_index.items()}
        
        logger.info(f"Vocabulary loaded. Total tokens: {len(vocab)}")
        return caption_to_index, index_to_caption
    
    def load_inference_model(self, model_path, vocab_size, device):
        """Load the trained model"""
        model = TransformerCaptioning(
            feature_dim=512,
            vocab_size=vocab_size,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            dropout=0.2,
            max_len=100
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    
    def extract_features_from_video(self, video_path):
        """Extract features from video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        
        cap.release()
        
        if not frames:
            raise Exception("No frames extracted from video")

        # Apply transformations
        transformed_frames = [self.transform(frame) for frame in frames]
        frames_tensor = torch.stack(transformed_frames).to(self.DEVICE)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(frames_tensor)
        
        # Pad or truncate features
        num_frames = features.shape[0]
        if num_frames < self.MAX_SEQ_LENGTH:
            padding = torch.zeros(self.MAX_SEQ_LENGTH - num_frames, self.FEATURE_DIM).to(self.DEVICE)
            features = torch.cat([features, padding], dim=0)
        elif num_frames > self.MAX_SEQ_LENGTH:
            features = features[:self.MAX_SEQ_LENGTH, :]
            
        return features.unsqueeze(0).cpu()
    
    def generate_caption_from_features(self, video_features, max_len=20):
        """Generate caption from video features"""
        self.model.eval()
        BOS_IDX = self.caption_to_index["<BOS>"]
        EOS_IDX = self.caption_to_index["<EOS>"]
        
        video_features = video_features.to(self.DEVICE)
        generated_indices = [BOS_IDX]
        
        with torch.no_grad():
            for _ in range(max_len - 1):
                captions_tensor = torch.tensor([generated_indices], dtype=torch.long).to(self.DEVICE)
                logits = self.model(video_features, captions_tensor)
                last_logits = logits[:, -1, :]
                predicted_idx = torch.argmax(last_logits, dim=-1).item()
                generated_indices.append(predicted_idx)
                if predicted_idx == EOS_IDX:
                    break
                    
        generated_caption = [self.index_to_caption.get(idx, "<UNK>") for idx in generated_indices]
        final_sentence = " ".join(generated_caption).replace("<BOS>", "").replace("<EOS>", "").strip()
        return final_sentence
    
    def predict_sign_from_video(self, video_path):
        """Main prediction function"""
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Extract features
            video_features = self.extract_features_from_video(video_path)
            
            # Generate caption
            predicted_text = self.generate_caption_from_features(video_features)
            
            logger.info(f"Predicted: {predicted_text}")
            return predicted_text
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise e

# =====================================================================================
#   4. FASTAPI APPLICATION
# =====================================================================================

# Initialize the service
service = SignLanguageService()

# Create FastAPI app
app = FastAPI(
    title="Sign Language Recognition API",
    description="API for recognizing sign language from video files",
    version="1.0.0"
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=service.model is not None,
        device=str(service.DEVICE)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_sign_language(file: UploadFile = File(...), session_id: Optional[str] = None):
    """
    Predict sign language from uploaded video file
    
    Args:
        file: Video file (mp4, avi, mov formats supported)
        session_id: Optional session identifier
    
    Returns:
        PredictionResponse with predicted text and session_id
    """
    import time
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be a video file (mp4, avi, mov)"
        )
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, f"temp_video_{file.filename}")
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process video
        predicted_text = service.predict_sign_from_video(temp_file_path)
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            success=True,
            predicted_text=predicted_text,
            processing_time=processing_time,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return PredictionResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time,
            session_id=session_id
        )
    
    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sign Language Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST with video file)",
            "docs": "/docs"
        }
    }

# =====================================================================================
#   5. MAIN FUNCTION
# =====================================================================================

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    