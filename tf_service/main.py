import os

import boto3
from fastapi import FastAPI, HTTPException, Body
import uvicorn
import chess
import tensorflow as tf

from heuristics import HeuristicsEngine

app = FastAPI()
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "sidzinski-model")
MODEL_S3_KEY   = os.environ.get("MODEL_S3_KEY", "my_engine_eval_model_100GB_of_parsed_games_pure_argmax_cpu.keras")
LOCAL_MODEL_PATH = "/tmp/model.keras"
print(f"Downloading model from s3://{S3_BUCKET_NAME}/{MODEL_S3_KEY} to {LOCAL_MODEL_PATH}")
s3 = boto3.client("s3")
s3.download_file(S3_BUCKET_NAME, MODEL_S3_KEY, LOCAL_MODEL_PATH)
model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
@app.post("/bestmove")
def bestmove_endpoint(payload: dict):
    try:
        fen = payload["fen"]
        board = chess.Board(fen)
        if board.is_game_over():
            return {"best_move_uci": None, "reason": "Game is over."}
        engine = HeuristicsEngine(model, 0.8)
        best_move =  engine.get_best_move(board)
        return {"best_move_uci": best_move.uci()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok"}