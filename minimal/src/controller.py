# controller.py
from __future__ import annotations

import json
from typing import Any, Dict
from clip_encoder import Clip
from env import VideoEnv


def minimal_video_rlm(video_path: str, query: str):
    clip = Clip()
    env = VideoEnv(video_path, clip, seg_len_s=60.0, base_fps=1.0)

    print("Building index (1 FPS CLIP + motion)...")
    env.build_index()

    # -------------------------
    # Step 1: coarse search (TOOL)
    # -------------------------
    top_k = 3
    cs1 = env.act({"tool": "search_segments", "query": query, "top_k": top_k})

    print("\n[ContextSlice] search_segments")
    print(env.context_to_dict(cs1))  # JSON-serializable

    # pick best seg from evidence note (or parse from your SearchResult if you prefer)
    # easiest: call search_segments directly only to get seg_idx OR change act() to also return seg_idx list.
    # BUT since you already have env.trace + evidence notes, we can parse seg idx from the first evidence note:
    if not cs1.evidence:
        print("\nNo segments returned. Exiting.")
        return None, None

    # note looks like: "seg=3, window=[60.00,120.00]"
    best_note = cs1.evidence[0].note
    best_seg_idx = int(best_note.split("seg=")[1].split(",")[0])

    # -------------------------
    # Step 2: refine best segment (TOOL)
    # -------------------------
    cs2 = env.act({
        "tool": "refine_in_segment",
        "query": query,
        "seg_idx": best_seg_idx,
        "dense_fps": 8.0,
        "window_s": 2.0,
    })

    print("\n[ContextSlice] refine_in_segment")
    print(env.context_to_dict(cs2))


    # Step 3: hypothesis-driven S1 inspection (adaptive FPS + zoom)
    # -------------------------
    if cs2.evidence:
        t0 = cs2.window.get("t0")
        t1 = cs2.window.get("t1")

        if t0 is not None and t1 is not None:
            # Guard: avoid empty/degenerate windows
            if t1 <= t0:
                # expand to a small valid window
                t0 = max(0.0, float(t0) - 1.0)
                t1 = min(env.duration, float(t0) + 2.0)

            hypothesis = "S1"                 # microevent regime
            fps_schedule = [4.0, 16.0, 24.0]  # escalate if uncertain
            sharpness_tau = 0.03              # tune: max-mean threshold
            zoom_s = 1.0                      # zoom window size around best evidence

            def sharpness(stats: Dict[str, Any]) -> float:
                ss = stats.get("score_stats")
                if not ss:
                    return -1e9
                return float(ss["max"] - ss["mean"])

            best = None
            best_sharp = -1e9

            for fps in fps_schedule:
                cs3 = env.act({
                    "tool": "inspect_window",
                    "query": query,
                    "t0": t0,
                    "t1": t1,
                    "fps": fps,
                    "top_m": 5,
                })
                d3 = env.context_to_dict(cs3)

                print(f"\n[ContextSlice] inspect_window @ fps={fps}")
                print(d3)

                sh = sharpness(d3["stats"])
                if sh > best_sharp:
                    best_sharp = sh
                    best = d3

                # Stop early if we have a sharp peak (microevent-like evidence)
                if hypothesis == "S1" and sh >= sharpness_tau and d3["stats"]["n_samples"] > 0:
                    # Zoom around the top evidence time and re-inspect once
                    if d3["evidence"]:
                        t_star = float(d3["evidence"][0]["t"])
                        z0 = max(0.0, t_star - zoom_s / 2)
                        z1 = min(env.duration, t_star + zoom_s / 2)

                        cs4 = env.act({
                            "tool": "inspect_window",
                            "query": query,
                            "t0": z0,
                            "t1": z1,
                            "fps": max(16.0, fps),
                            "top_m": 5,
                        })
                        print(f"\n[ContextSlice] zoom_inspect @ [{z0:.2f},{z1:.2f}] fps={max(16.0,fps)}")
                        print(env.context_to_dict(cs4))
                    break

    # -------------------------
    # Verify tool usage: env.trace
    # -------------------------
    print("\n=== TOOL TRACE ===")
    print(json.dumps(env.trace, indent=2))

    return cs1, cs2


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()
    minimal_video_rlm(args.video, args.query)