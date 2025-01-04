import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chess
from typing import List, Dict

# Define the path to your trained Keras model
MODEL_PATH = "best_trained_model_small_cnn_l2.keras"

# Load the model once when the service starts
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model. Ensure the MODEL_PATH is correct. Error: {e}")

# Initialize move encoding mappings
# Replace with your actual move encoding logic

class MoveEncoder:
    def __init__(self):
        self.uci_to_index = {}
        self.index_to_uci = {}
        self.NUM_MOVES = 20480  # Must match the model's num_moves

    def initialize_encoding(self):
        # Initialize encoding mappings
        # Replace this with your actual move list
        predefined_uci_moves = self.generate_predefined_moves()
        for i, uci in enumerate(predefined_uci_moves):
            if i >= self.NUM_MOVES:
                break
            self.uci_to_index[uci] = i
            self.index_to_uci[i] = uci

    def generate_predefined_moves(self) -> List[str]:
        # Placeholder for generating or loading the list of 20480 UCI moves
        # Replace this with your actual logic
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['1','2','3','4','5','6','7','8']
        moves = []
        for from_file in files:
            for from_rank in ranks:
                for to_file in files:
                    for to_rank in ranks:
                        move = f"{from_file}{from_rank}{to_file}{to_rank}"
                        moves.append(move)
                        if len(moves) >= self.NUM_MOVES:
                            break
                    if len(moves) >= self.NUM_MOVES:
                        break
                if len(moves) >= self.NUM_MOVES:
                    break
            if len(moves) >= self.NUM_MOVES:
                break
        return moves

    def encode_move(self, move: chess.Move) -> int:
        return self.uci_to_index.get(move.uci(), -1)

    def decode_move(self, index: int) -> str:
        return self.index_to_uci.get(index, None)

# Initialize the encoder
move_encoder = MoveEncoder()
move_encoder.initialize_encoding()

# Define number of moves
NUM_MOVES = move_encoder.NUM_MOVES

# Define the request model
class MoveRequest(BaseModel):
    fen: str

# Define the response model
class MoveResponse(BaseModel):
    best_move: str

# Initialize FastAPI app
app = FastAPI(title="Chess AI Move Predictor")

def board_to_features(board: chess.Board) -> np.ndarray:
    """
    Convert a chess board to a feature tensor with shape (8, 8, 14).
    This should match the encoding used during model training.
    """
    planes = 14  # Must match the model's board_shape channels
    board_planes = np.zeros((8, 8, planes), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type  # 1 to 6
            color = int(piece.color)       # 0 for white, 1 for black
            # Example encoding: each piece type and color have their own plane
            # Adjust plane index based on your encoding during training
            # For example, plane 0-5 for white pieces and 6-11 for black pieces
            # Additional planes can be used for castling rights, en passant, etc.
            if piece_type <= 6 and color <=1:
                plane_idx = (piece_type - 1) + color * 6
                if plane_idx < planes:
                    row = 7 - chess.square_rank(square)  # Convert to 0-indexed row
                    col = chess.square_file(square)      # 0-indexed column
                    board_planes[row, col, plane_idx] = 1.0

    # Additional features can be added here (e.g., castling rights, en passant)
    # Example: Plane 12 for castling rights, Plane 13 for en passant
    # Plane 12: 1 if white can castle kingside, else 0
    # Plane 13: 1 if black can castle queenside, else 0
    # Adjust based on your model's expectations

    # Example: Adding castling rights
    board_planes[:, :, 12] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    board_planes[:, :, 13] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0

    return board_planes  # Shape: (8, 8, 14)

def get_legal_moves_mask(board: chess.Board) -> np.ndarray:
    """
    Create a mask for legal moves based on the encoding.
    The mask length should match NUM_MOVES (20480).
    """
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    for move in board.legal_moves:
        index = move_encoder.encode_move(move)
        if index != -1:
            mask[index] = 1.0
    return mask  # Shape: (20480,)

@app.post("/predict-move", response_model=MoveResponse)
def predict_move(move_request: MoveRequest):
    fen = move_request.fen

    try:
        board = chess.Board(fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string.")

    # Encode the board state as (8, 8, 14)
    board_tensor = board_to_features(board)  # Shape: (8, 8, 14)

    # Expand dimensions to add batch size
    input_board = np.expand_dims(board_tensor, axis=0)  # Shape: (1, 8, 8, 14)

    # Generate the legal moves mask
    mask = get_legal_moves_mask(board)  # Shape: (20480,)

    # Expand dimensions to add batch size
    mask_input = np.expand_dims(mask, axis=0)  # Shape: (1, 20480)

    # Create a dictionary of inputs matching the model's expected input layer names
    inputs: Dict[str, np.ndarray] = {
        'input_board': input_board,   # Shape: (1, 8, 8, 14)
        'mask_input': mask_input      # Shape: (1, 20480)
    }

    # Get model prediction
    try:
        # The model outputs [policy_output, value_output]
        policy_output, value_output = model.predict(inputs, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    # Flatten policy_output to (20480,)
    output_probs = policy_output.flatten()

    # Check if any valid moves are present
    if not np.any(output_probs):
        raise HTTPException(status_code=400, detail="AI could not determine a valid move.")

    # Select the move with the highest probability among legal moves
    best_move_index = np.argmax(output_probs)
    best_move_uci = move_encoder.decode_move(best_move_index)

    if not best_move_uci:
        raise HTTPException(status_code=400, detail="AI could not determine a valid move.")

    return MoveResponse(best_move=best_move_uci)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)