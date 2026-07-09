import os
import json
from pathlib import Path
from conductor.profiler.llm_profiler import profile_query_llm

BASE_URL = "http://localhost:9000/v1"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

DATASETS = {
    "egoschema": {
        "input": "/dataheart/hussainahmad/video-rlm/conductor/experiments/egoschema/results/egoschema_budget_all.jsonl",
        "output": "egoschema_profiler.jsonl",
    },

    "lvbench": {
        "input": "/dataheart/hussainahmad/video-rlm/conductor/experiments/lvb/results/lvb_knob_sensitivity_unique.jsonl",
        "output": "lvbench_profiler.jsonl",
    },

    "vrbench": {
        "input": "/dataheart/hussainahmad/video-rlm/conductor/experiments/vrbench/dataset/vrbench_synthesis_133questions_unique.jsonl",
        "output": "vrbench_profiler.jsonl",
    },
}


import cv2

_DURATION_CACHE = {}

def get_duration(example):
    value = example.get("duration_s")
    if value is not None:
        try:
            value = float(value)
            if value > 0:
                return value
        except Exception:
            pass

    video = example["video"]

    if video in _DURATION_CACHE:
        return _DURATION_CACHE[video]

    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    duration = frames / fps

    _DURATION_CACHE[video] = duration

    return duration


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_question(example):

    for key in [
        "question",
        "query",
        "prompt",
    ]:
        if key in example:
            return example[key]

    raise RuntimeError("Question field not found")


def get_id(example):

    for key in [
        "qid",
        "question_id",
        "id",
    ]:
        if key in example:
            return example[key]

    return None



def run_dataset(name, input_file, output_file):

    print("=" * 100)
    print(name)
    print("=" * 100)

    examples = load_jsonl(input_file)

    print(f"Loaded {len(examples)} questions")

    Path(output_file).unlink(missing_ok=True)

    with open(output_file, "w") as fout:
        print("Writing to:", os.path.abspath(output_file))

        for idx, example in enumerate(examples):

            if idx % 25 == 0:
                print(f"{idx}/{len(examples)}")

            question = get_question(example)
            try:
                duration = get_duration(example)
            except Exception as e:
                print(f"Skipping {get_id(example)}: {e}")
                continue

            result = profile_query_llm(
                query=question,
                duration_s=duration,
                base_url=BASE_URL,
                model=MODEL,
            )

            row = {
                "dataset": name,
                "qid": get_id(example),
                "duration_s": duration,
                "question": question,

                "temporal_requirement":
                    result.analysis["temporal_requirement"],

                "candidate_requirement":
                    result.analysis["candidate_requirement"],

                "context_requirement":
                    result.analysis["context_requirement"],

                "precision_requirement":
                    result.analysis["precision_requirement"],

                "probe_fps":
                    result.execution_policy["probe_fps"],

                "probe_topk":
                    result.execution_policy["probe_topk"],

                "action_topk":
                    result.execution_policy["action_topk"],

                "window_len_s":
                    result.execution_policy["window_len_s"],

                "vlm_budget":
                    result.execution_policy["answer_max_images_total"],
            }

            fout.write(json.dumps(row) + "\n")
            fout.flush()                  # flush Python buffer
            os.fsync(fout.fileno())       # force write to disk

            

            print(
                f"[{name}] "
                f"{idx+1}/{len(examples)} "
                f"({100*(idx+1)/len(examples):.1f}%) "
                f"qid={get_id(example)}"
            )

    print(f"Saved {output_file}")
    print()


def main():

    for dataset_name, config in DATASETS.items():

        run_dataset(
            dataset_name,
            config["input"],
            config["output"],
        )

    print()
    print("=" * 100)
    print("Finished all datasets.")
    print("=" * 100)


if __name__ == "__main__":
    main()