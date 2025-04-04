import chess
import numpy as np


def encode_single_board(board: chess.Board) -> np.ndarray:
    """
    Returns a float32 array of shape (64, 17).

    Plane layout (plane indices):
      0  : White Pawn
      1  : White Knight
      2  : White Bishop
      3  : White Rook
      4  : White Queen
      5  : White King
      6  : Black Pawn
      7  : Black Knight
      8  : Black Bishop
      9  : Black Rook
      10 : Black Queen
      11 : Black King
      12 : Castling rights (integer 0..15) scaled to [0..1], replicated
      13 : Side to move (1.0 = White, 0.0 = Black), replicated
      14 : En-passant square (1.0 where the EP capture square is, else 0.0)
      15 : Halfmove clock (scaled/clamped to [0..1]), replicated
      16 : Fullmove number (scaled/clamped to [0..1]), replicated
    """
    enc = np.zeros((64, 17), dtype=np.float32)

    # 1) Fill piece planes
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            piece_type = piece.piece_type  # 1..6
            color_offset = 0 if piece.color == chess.WHITE else 6
            plane_idx = color_offset + (piece_type - 1)
            enc[sq, plane_idx] = 1.0

    # 2) Castling rights in plane 12
    castling_code = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_code |= 1  # bit 0
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_code |= 2  # bit 1
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_code |= 4  # bit 2
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_code |= 8  # bit 3
    enc[:, 12] = castling_code / 15.0

    # 3) Side to move in plane 13
    enc[:, 13] = float(board.turn == chess.WHITE)

    # 4) En-passant square in plane 14
    if board.ep_square is not None:
        enc[board.ep_square, 14] = 1.0

    # 5) Halfmove clock in plane 15 (clamp at 100 for scaling)
    halfmove_val = min(board.halfmove_clock, 100) / 100.0
    enc[:, 15] = halfmove_val

    # 6) Fullmove number in plane 16 (clamp at 1000)
    fullmove_val = min(board.fullmove_number, 1000) / 1000.0
    enc[:, 16] = fullmove_val

    return enc


def move_to_index(move: chess.Move) -> int:
    """
    Convert chess.Move to an integer in [0, 20480).
    from_sq * 64 * 5 + to_sq * 5 + promo_code
    """
    promo_code = 0
    if move.promotion:
        if move.promotion == chess.KNIGHT:
            promo_code = 1
        elif move.promotion == chess.BISHOP:
            promo_code = 2
        elif move.promotion == chess.ROOK:
            promo_code = 3
        elif move.promotion == chess.QUEEN:
            promo_code = 4

    return (
        move.from_square * 64 * 5
        + move.to_square * 5
        + promo_code
    )