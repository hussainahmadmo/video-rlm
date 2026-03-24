# notes/model_map.yaml
tasks:
  coarse_indexing:
    goal: "cheap signals for routing"
    candidates:
      - name: "CLIP ViT-L/14 (open_clip)"
        cost: "cheap"
        strengths: ["semantic similarity", "works at ~1 FPS"]
        weaknesses: ["not descriptive", "misses subtle actions"]
  cheap_caption:
    goal: "short text tags per segment"
    candidates:
      - name: "Qwen2-VL (small)"
        cost: "medium"
        strengths: ["basic actions/objects", "better than CLIP scores alone"]
        weaknesses: ["can hallucinate", "still misses subtle cues"]
  final_vlm_caption:
    goal: "descriptive evidence after routing"
    candidates:
      - name: "Qwen2.5-VL (larger)"
        cost: "expensive"
        strengths: ["richer captions", "better action phrasing"]
        weaknesses: ["cost"]
  judge_controller_llm:
    goal: "decide next tool / backtrack"
    candidates:
      - name: "Qwen2.5-7B-Instruct"
        cost: "cheap"
        strengths: ["good tool selection", "fast"]
        weaknesses: ["needs descriptive evidence to be confident"]