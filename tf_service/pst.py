import chess

PIECE_VALUES_OUR = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

PST_KING_MIDGAME = [
    [ -30, -40, -40, -50, -50, -40, -40, -30 ],
    [ -30, -40, -40, -50, -50, -40, -40, -30 ],
    [ -30, -40, -40, -50, -50, -40, -40, -30 ],
    [ -30, -40, -40, -50, -50, -40, -40, -30 ],
    [ -20, -30, -30, -40, -40, -30, -30, -20 ],
    [ -10, -20, -20, -20, -20, -20, -20, -10 ],
    [  20,  20,   0,   0,   0,   0,  20,  20 ],
    [  20,  30,  10,   0,   0,  10,  30,  20 ],
]

# A typical king endgame PST encourages the king to move more centrally:
PST_KING_ENDGAME = [
    [ -50, -30, -30, -30, -30, -30, -30, -50 ],
    [ -30, -20,   0,   0,   0,   0, -20, -30 ],
    [ -30,  10,  20,  30,  30,  20,  10, -30 ],
    [ -30,  10,  30,  40,  40,  30,  10, -30 ],
    [ -30,  10,  30,  40,  40,  30,  10, -30 ],
    [ -30,  10,  20,  30,  30,  20,  10, -30 ],
    [ -30, -20,   0,   0,   0,   0, -20, -30 ],
    [ -50, -30, -30, -30, -30, -30, -30, -50 ],
]

# Keep the other piece tables from your production set:
PST_PAWN = [
    [   0,   0,   0,   0,   0,   0,   0,   0 ],
    [  50,  50,  50,  50,  50,  50,  50,  50 ],
    [  10,  10,  20,  30,  30,  20,  10,  10 ],
    [   5,   5,  10,  25,  25,  10,   5,   5 ],
    [   0,   0,   0,  20,  20,   0,   0,   0 ],
    [   5,  -5, -10,   0,   0, -10,  -5,   5 ],
    [   5,  10,  10, -20, -20,  10,  10,   5 ],
    [   0,   0,   0,   0,   0,   0,   0,   0 ],
]
PST_KNIGHT = [
    [ -50, -40, -30, -30, -30, -30, -40, -50 ],
    [ -40, -20,   0,   0,   0,   0, -20, -40 ],
    [ -30,   0,  10,  15,  15,  10,   0, -30 ],
    [ -30,   5,  15,  20,  20,  15,   5, -30 ],
    [ -30,   0,  15,  20,  20,  15,   0, -30 ],
    [ -30,   5,  10,  15,  15,  10,   5, -30 ],
    [ -40, -20,   0,   5,   5,   0, -20, -40 ],
    [ -50, -40, -30, -30, -30, -30, -40, -50 ],
]
PST_BISHOP = [
    [ -20, -10, -10, -10, -10, -10, -10, -20 ],
    [ -10,   5,   0,   0,   0,   0,   5, -10 ],
    [ -10,  10,  10,  10,  10,  10,  10, -10 ],
    [ -10,   0,  10,  10,  10,  10,   0, -10 ],
    [ -10,   5,   5,  10,  10,   5,   5, -10 ],
    [ -10,   0,   5,  10,  10,   5,   0, -10 ],
    [ -10,   0,   0,   0,   0,   0,   0, -10 ],
    [ -20, -10, -10, -10, -10, -10, -10, -20 ],
]
PST_ROOK = [
    [   0,   0,   0,   5,   5,   0,   0,   0 ],
    [  -5,   0,   0,   0,   0,   0,   0,  -5 ],
    [  -5,   0,   0,   0,   0,   0,   0,  -5 ],
    [  -5,   0,   0,   0,   0,   0,   0,  -5 ],
    [  -5,   0,   0,   0,   0,   0,   0,  -5 ],
    [  -5,   0,   0,   0,   0,   0,   0,  -5 ],
    [   5,  10,  10,  10,  10,  10,  10,   5 ],
    [   0,   0,   0,   5,   5,   0,   0,   0 ],
]
PST_QUEEN = [
    [ -20, -10, -10,  -5,  -5, -10, -10, -20 ],
    [ -10,   0,   5,   0,   0,   0,   0, -10 ],
    [ -10,   5,   5,   5,   5,   5,   0, -10 ],
    [   0,   0,   5,   5,   5,   5,   0,  -5 ],
    [  -5,   0,   5,   5,   5,   5,   0,  -5 ],
    [ -10,   0,   5,   5,   5,   5,   0, -10 ],
    [ -10,   0,   0,   0,   0,   0,   0, -10 ],
    [ -20, -10, -10,  -5,  -5, -10, -10, -20 ],
]

# We’ll keep the rest of the dictionary for other pieces the same:
PIECE_TO_PST = {
    chess.PAWN:   PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK:   PST_ROOK,
    chess.QUEEN:  PST_QUEEN,
    # For the king, we will dynamically choose midgame vs endgame below.
    # So no static entry for the King here (or you can store PST_KING_MIDGAME as a default if you prefer).
}

def is_endgame(board: chess.Board) -> bool:
    val = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type != chess.PAWN:
            # PIECE_VALUES_OUR is your dict {Pawn:1, Knight:3, ...}
            val += PIECE_VALUES_OUR.get(piece.piece_type, 0)
    return val <= 13

def piece_square_table_score_white(board: chess.Board) -> float:
    score = 0.0
    endgame = is_endgame(board)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue
        piece_type = piece.piece_type

        # Choose which PST to use:
        if piece_type == chess.KING:
            # If it's a king, pick midgame or endgame PST
            if endgame:
                pst_table = PST_KING_ENDGAME
            else:
                pst_table = PST_KING_MIDGAME
        else:
            # For other piece types, just use the normal dictionary
            if piece_type not in PIECE_TO_PST:
                continue
            pst_table = PIECE_TO_PST[piece_type]

        rank = chess.square_rank(sq)  # 0..7, 0=rank1 (White's back rank)
        file = chess.square_file(sq)  # 0..7, 0=file a
        # White uses table as-is, Black mirrors rank (7-rank) and flips sign
        if piece.color == chess.WHITE:
            value = pst_table[rank][file]
        else:
            value = - pst_table[7 - rank][file]

        score += value

    return score