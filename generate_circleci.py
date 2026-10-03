"""
CircleCI Chess Dataset Generator (generate_circleci.py)
------------------------------------------------------
Runs on CircleCI Linux Container (2 vCPU, 4GB RAM).
- Range: Games 530,000 -> 580,000 (50,000 games)
- Output: /tmp/v5_dataset_530k_580k.bin
"""

import os
import io
import sys
import time
import struct
import urllib.request
import subprocess
import multiprocessing
import threading
from typing import Tuple, Generator

GAMES_TO_SKIP = 530000
MAX_GAMES = 50000
DEPTH = 10
NUM_WORKERS = 2  # 2 vCPU on CircleCI medium
HASH_PER_WORKER = 64
OUTPUT_FILE = "/tmp/v5_dataset_530k_580k.bin"
STOCKFISH_BIN = "/usr/games/stockfish"
LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2015-10.pgn.zst"
LOCAL_PGN = "/tmp/lichess_db_standard_rated_2015-10.pgn.zst"

def setup_stockfish() -> str:
    import shutil
    which = shutil.which("stockfish")
    if which:
        print(f"[Stockfish] Found at {which}", flush=True)
        return which
    candidates = ["/usr/games/stockfish", "/usr/bin/stockfish"]
    for c in candidates:
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c
    subprocess.run(["sudo", "apt-get", "update", "-qq"], check=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "stockfish"], check=True)
    return "/usr/games/stockfish"

def setup_pgn() -> str:
    if os.path.exists(LOCAL_PGN) and os.path.getsize(LOCAL_PGN) > 1000000:
        return LOCAL_PGN
    print(f"[PGN] Downloading from Lichess CDN ({LICHESS_URL})...", flush=True)
    t0 = time.time()
    subprocess.run(["curl", "-s", "-L", "-o", LOCAL_PGN, LICHESS_URL], check=True)
    dt = time.time() - t0
    mb = os.path.getsize(LOCAL_PGN) / (1024 ** 2)
    print(f"[PGN] Downloaded {mb:.1f} MB in {dt:.1f}s ({mb/dt:.1f} MB/s)!", flush=True)
    return LOCAL_PGN

def encode_board_and_metadata(board) -> Tuple[bytes, int]:
    import chess
    board_arr = [0] * 64
    for pt, bb in enumerate([board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings], 1):
        white_bb = bb & board.occupied_co[chess.WHITE]
        black_bb = bb & board.occupied_co[chess.BLACK]
        while white_bb:
            lsb = white_bb & -white_bb
            sq = lsb.bit_length() - 1
            board_arr[sq] = pt
            white_bb &= white_bb - 1
        while black_bb:
            lsb = black_bb & -black_bb
            sq = lsb.bit_length() - 1
            board_arr[sq] = pt + 6
            black_bb &= black_bb - 1
            
    packed_board = bytearray(32)
    for i in range(32):
        packed_board[i] = (board_arr[i * 2] << 4) | board_arr[i * 2 + 1]
        
    stm = 0 if board.turn else 1
    cr = board.castling_rights
    castling = (1 if cr & 128 else 0) | (2 if cr & 1 else 0) | (4 if cr & 9223372036854775808 else 0) | (8 if cr & 72057594037927936 else 0)
    ep = board.ep_square
    ep_file = ep & 7 if ep else 8
    halfmove = min(board.halfmove_clock, 127)
    
    metadata = (stm) | (castling << 1) | (ep_file << 5) | (halfmove << 9)
    return bytes(packed_board), metadata

global_engine = None
global_depth = 10

def worker_init(engine_path: str, hash_size: int, depth: int):
    global global_engine, global_depth
    import chess.engine
    global_depth = depth
    global_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    global_engine.configure({"Hash": hash_size, "Threads": 1})

def worker_task(pgn_string: str) -> bytes:
    global global_engine, global_depth
    import chess
    import chess.pgn
    import chess.engine
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_string))
        if not game:
            return b""
        result_str = game.headers.get("Result", "*")
        res_val = 2
        if result_str == "1-0": res_val = 1
        elif result_str == "0-1": res_val = -1
        elif result_str == "1/2-1/2": res_val = 0
        
        board = game.board()
        packed_positions = bytearray()
        for move in game.mainline_moves():
            info = global_engine.analyse(board, chess.engine.Limit(depth=global_depth))
            raw_score = info["score"].white().score(mate_score=32000)
            score = raw_score if raw_score is not None else 0
            score = max(-32767, min(32767, score))
            packed_board, metadata = encode_board_and_metadata(board)
            packed_pos = struct.pack('<32s H h b b', packed_board, metadata, score, res_val, global_depth)
            packed_positions.extend(packed_pos)
            board.push(move)
        return bytes(packed_positions)
    except Exception:
        return b""

def open_pgn_stream(file_path: str):
    if file_path.endswith('.zst'):
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        f_raw = open(file_path, 'rb')
        reader = dctx.stream_reader(f_raw)
        return io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
    else:
        return open(file_path, 'r', encoding='utf-8', errors='replace')

def game_generator(stream_file, games_to_skip: int, max_games: int) -> Generator[str, None, None]:
    current_lines = []
    games_scanned = 0
    games_yielded = 0
    print(f"[Stream] Fast-forwarding past first {games_to_skip:,} games...", flush=True)
    t0 = time.time()
    for line in stream_file:
        if line.startswith("[Event ") and current_lines:
            games_scanned += 1
            if games_scanned <= games_to_skip:
                if games_scanned % 50000 == 0:
                    print(f"  -> Skipped {games_scanned:,}/{games_to_skip:,} games ({time.time()-t0:.1f}s)...", flush=True)
            else:
                yield "".join(current_lines)
                games_yielded += 1
                if games_yielded >= max_games:
                    return
            current_lines = [line]
        else:
            current_lines.append(line)
            
    if current_lines:
        games_scanned += 1
        if games_scanned > games_to_skip and games_yielded < max_games:
            yield "".join(current_lines)

def main():
    print("=" * 65, flush=True)
    print(f"CIRCLECI CHESS DATASET GENERATOR (530k -> 580k)", flush=True)
    print("=" * 65, flush=True)

    engine_bin = setup_stockfish()
    pgn_path = setup_pgn()

    print(f"\nConfiguration:")
    print(f"  Games to Skip:    {GAMES_TO_SKIP:,} (Game {GAMES_TO_SKIP+1:,} -> {GAMES_TO_SKIP+MAX_GAMES:,})")
    print(f"  Target Games:     {MAX_GAMES:,}")
    print(f"  Engine Depth:     {DEPTH}")
    print(f"  Engine Binary:    {engine_bin}")
    print(f"  Workers:          {NUM_WORKERS} parallel Stockfish instances")
    print(f"  Output Path:      {OUTPUT_FILE}")
    print("=" * 65, flush=True)

    stream = open_pgn_stream(pgn_path)
    games_iter = game_generator(stream, GAMES_TO_SKIP, MAX_GAMES)

    print(f"[Pool] Spawning {NUM_WORKERS} Stockfish processes...", flush=True)
    pool = multiprocessing.Pool(
        processes=NUM_WORKERS,
        initializer=worker_init,
        initargs=(engine_bin, HASH_PER_WORKER, DEPTH)
    )

    MAX_QUEUE = NUM_WORKERS * 4
    semaphore = threading.Semaphore(MAX_QUEUE)
    write_lock = threading.Lock()

    total_games_done = 0
    total_positions_packed = 0
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    bin_file = open(OUTPUT_FILE, "wb")
    start_time = time.time()

    def result_callback(packed_bytes: bytes):
        nonlocal total_games_done, total_positions_packed
        if packed_bytes:
            with write_lock:
                bin_file.write(packed_bytes)
                total_positions_packed += len(packed_bytes) // 38
        with write_lock:
            total_games_done += 1
            if total_games_done % 200 == 0 or total_games_done == MAX_GAMES:
                elapsed = time.time() - start_time
                speed_games = total_games_done / max(elapsed, 0.1)
                speed_pos = total_positions_packed / max(elapsed, 0.1)
                eta_m = (MAX_GAMES - total_games_done) / speed_games / 60 if speed_games > 0 else 0
                file_mb = os.path.getsize(OUTPUT_FILE) / (1024 ** 2)
                print(
                    f"[{total_games_done:,}/{MAX_GAMES:,} games] "
                    f"Packed: {total_positions_packed:,} pos ({file_mb:.1f} MB) "
                    f"| Speed: {speed_games:.1f} g/s ({speed_pos:.0f} pos/s) "
                    f"| Elapsed: {elapsed/60:.1f}m | ETA: {eta_m:.1f}m",
                    flush=True
                )
        semaphore.release()

    def error_callback(err):
        print(f"[Worker Error] {err}", flush=True)
        semaphore.release()

    print("[Pipeline] Dispatching games to worker pool...", flush=True)
    for game_pgn in games_iter:
        semaphore.acquire()
        pool.apply_async(worker_task, (game_pgn,), callback=result_callback, error_callback=error_callback)

    while total_games_done < MAX_GAMES:
        time.sleep(0.5)

    pool.terminate()
    pool.join()
    bin_file.flush()
    bin_file.close()
    stream.close()

    if os.path.exists(LOCAL_PGN):
        try: os.remove(LOCAL_PGN)
        except Exception: pass

    total_time = time.time() - start_time
    final_mb = os.path.getsize(OUTPUT_FILE) / (1024 ** 2)

    print("\n" + "=" * 65, flush=True)
    print("CIRCLECI DATASET GENERATION COMPLETED!", flush=True)
    print(f"  Total Games Processed:   {total_games_done:,}")
    print(f"  Total Positions Packed:  {total_positions_packed:,}")
    print(f"  Output File Size:        {final_mb:.2f} MB")
    print(f"  Total Time:              {total_time/60:.2f} minutes")
    print(f"  Saved At:                {OUTPUT_FILE}")
    print("=" * 65, flush=True)

if __name__ == '__main__':
    main()
