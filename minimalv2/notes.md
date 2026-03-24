 # Core objective

Optimize jointly:
- answer accuracy
- video encoding cost.

- answer accuracy can be improved if we choose the right windows(where the event is happening).
  -- choosing the right window for visual queries is finding the right event.
     -- CLIP identifies events(similar to query), but suffers when the query 
        asks for smaller detail.(for smaller detail)

  -- 


  -- choosing the right window to answer a question for ASR is non-trivial.
     we can find the right visual for an event if its specified in the 
     query, but if its not specified in the query then it wont work.



(NOTE video encoding cost becomes important because the TTFT for 200 image is fast)

Most existing video QA systems optimize only for answer accuracy and often apply a fixed or heavy video-processing pipeline regardless what question actually requires, even when the query only needs one modality or a cheaper configuration. Our contrib is to decide which tools are necessary for a given question and avoid paying the cost of unnecessary video processing, choosing a cheaper configuration that jointly optimizes answer accuracy and video encoding cost. Reducing the cost of video encoding and right model strength to answer it accurately while minimizing unneccesary.

Our goal is to move toward a system that selects only the necessary video tools and the necessary model strength for each question, so that we preserve answer quality while reducing unnecessary video processing cost.

In video QA, each query has an associated video execution config, that specifies which modalaties to invoke, and how much video content to encode such as frame rate, resolution, number of frames, OCR/ASR usage and model strength. Most prior pipelines optimize for accuracy only and apply a fixed pipeline to every query, even when query only needs one modality or a cheaper configuration.

Prior work reduces latency mainly by shrinking the visual input. Our goal is broader, for each query, decide which modality to invoke, how much of the video to invoke, and which model to strength to use, so that latency and encoding cost are reduced without sacrificing the answer accuracy.

# Prior works
(Prior work papers are based on frame selection - or token reduction)

- our work is closer to VideoAgent on online tool use/adaptive inference.
- different from VideoAgent because yours is bounded, cost aware, and execution-plan-driven.
- different from AdaVideoRAG because yours is less about retreival over offline knowledge stores and more about how to process the raw video now.

# Key assumptions

- Different queries require different tools and each tool incurs computation cost.
    - Not all queries require same modality
    - Not all queries require the same amount of compute.

    Examples
    - text in video type of queries/questions require --> OCR

    - speech questions type of queries/questions ---> ASR

    - visual appearance / event questions --> CLIP or VLM-based visual retrieval.
    - - use a vision model itselfs to decide which frames/windows are relevant to the question, instead of using only a CLIP-style embedding similarity score.


This decomposes into two decisions.

1. Tool selection
- CLIP for frame filtering / coarse visual retrieval
  - use CLIP to quickly score which frames or short video windows are most relevant to the question, then
    only keep those for later expensive processing.    
    "What color was the car that appeared near the end?"
        
    1. Sample frames across the video.
    2. use CLIP to compare each frame to the question text.
    3. assign a similarity score.
    4. keep the top frames/top windows
    5. only run expensive inspection on those.

- OCR for text extraction from frames.
    
    1.
    2.


- ASR for speech extraction from audio

    1.
    2.

- LLM for reasoning over extracted evidence.

2. Model/config selection
- cheap vs stronger OCR
- cheap vs strong VLM
- frame sampling rate / resolution / number of frames
- whether higher-resolution inspection is needed.


The problem becomes recognizing the best tool for a query and then using the best model for that query which can jointly reduce the accuracy and the video encoding cost.
    - final answering with a downstream model.
    - CLIP is best for frame filtering
    - OCR text extraction in video.
    - ASR speech extraction.


# Limitations

Main Limitations of the current approach

1. Retreival bottleneck: CLIP is the single point of failure.

A major weakness is that if the CLIP retreived the wrong video, downstream tools such as ASR
and OCR fail.

This is specially harmful when:
- OCR is run only on CLIP-selected windows.
- ASR is run only on CLIP-selected windows.

Key takeaway - even if OCR or ASR are the correct modalities for the question, they may never see the relevant region if the CLIP misses it.

Why this matters :

- For spoken-answer questions, the relevant evidence may come from audio, not from visual similarity.
- For text-questions, the relevant frame may contain small text that CLIP does not rank highly.

---- How can 


IN RAG we decide how much 
























- We are faced with many design choices.
    Primary Design Knob 1 - Tool selection.

    The first design knob, asks which tool can best answer a particular video question.

    TOOLS required for a particular video question.

    OCR for Optical Character Recognition.(OCR)

    - e.g for questions that require describing text in video(e.g a sign, or subtitiles) 
        the best method is using Optional Character recognition OCR.

        Do all OCR model answer harder text questions ?
        - Some OCR models on text screen questions that are harder to recognize due either the OCR model 
            being low quality or the frame sampler missing important information.
        
        Does frame sampling impact the output of OCR ?
        - Yes, if the frame sampling is bad, OCR will suffer despite the quality of the model.

        Can other tools such as CLIP, ASR replace OCR ? 
            No, other tools cannot replace OCR because,
            - CLIP is used for frame filtering, and semantic retreival of frames but it cannot read text on screen.
            - ASR is used for speech recognition.
            - OCR is used for text on screen.


    - OCR text extraction -
        - for some questions OCR text extraction requires a better model because the text extraction is hard.
        - e.g CLIP cannot do bounding boxes.


    CLIP for 
    - semantic visual retrieval
    - match an image/frame to a text description
    - coarse relevance score.





Note whenever you want to use LLM to decide, tool usage for higher resolution, or completely give the control of the tool usage to an LLM then your system becomes 
similar to videoagent.

Only difference 

 to encode or not or use higher resolutions or not.  

- similarity

Current LLM preprocessing steps on CPU :

- 


Could a simple classifier do the same thing ? 

- Dont know

Why do we need LLM based routing ?

QWEN MODEL is great for OCR stuff(cannot perform JSON as well)
- ocr_model = "Qwen/Qwen2.5-VL-3B-Instruct"

MODELS that are ideal for the JSON outputting -
- "Qwen/Qwen2.5-VL-3B-Instruct"

(Hussain) check what model can do best ASR. 

Currently for ASR questions e.g 
c6fuIEzOZ2E.mp4, the wrong model is chosen. even though the model that should have been chosen is ASR currently it is not an ASR model.

Do a comparision on whether CLIP model can be used or not.
- e.g if we use CLIP + ASR + OCR is it strictly better.


problem for the question that is called the jJGbXCCU5yc_1, the CLIP model cannot recognize
the camera, and misclassider it as an umbrella.

- for small items clip can miss lower level details if the sampling is bad, 
    now we want to make a slide for how CLIP misses the lower level details and its better to use higher level of frames to get those details.)



for spoken-answer questions which is the case for the videos athabNMGceo_0 like this, CLIP sometimes misses the regions where the 
speech is talking about a thing. 

e.g In the athabNMGceo_0 is talking about the 

"question": "Against a blue background, a man wearing black-framed glasses and a white short-sleeve shirt with a small bird pattern is explaining. Which of the following animals evolved hindgut fermentation?",


What we need to do is ASR-first search 

Also make some slides on why ASR search can be bad.
- e.g why doing ASR search only is bad 

In our pipeline, is encoding actually dominant over reasoning cost ? 


Big Problem - if CLIP retreival is wrong, then any downstream stage on the retreived windows can fail. This is speciallay damaging when OCR or ASR are run only on CLIP-selected regions.

- The problem with CLIP is true for the question where a man works with a log:

    "On the golden ground with black and gray caves and scattered wood, what did the man, who appeared for the first time when a shirtless man with a cloth strip tied around his waist and a beard appeared, do?",lift a tree trunk with both hands,The man appears to be performing a ritual or ceremony involving the wooden logs and the cave.,0.8,clip,(VISUAL ONLY)

# Reviewers Questions

How it differs from VideoRAG ?

VideoRAG is mainly about:
- building a knowledge graph
- maintaining relationships across many videos.
- offline multimodal indexing 

videos. e.g if we have many videos I want to maintain all the relationships between them.

Your system is more about:
- query-time decision making
- which modality to invoke
- whether to encode more
- how much visual compute to spend.

So VideoRAG is closer to:
- knowledge organization and retrieval.

Your system is closer to:
    - adaptive execution planning.


How it differs from VideoAgent

VideoAgent is more open-ended:
- the LLM acts like a general tool-using agent.

Our system is more structured:
- limited tools
- explicit knobs
- cost-aware decisions
- bounded recursion/escalation


There are some models that have multimodality in it. e.g QWEN-OMNI why should 
one not use that model, is the complexity of the pipeline.



Picking the right modality:

# Direct spoken-answer queries.
These are questions where the answer is in the speech/transcript.
- visual localization from CLIP does not help.
- the right modality is ASR only

e.g What number does he say.

# Speech reference query
These are questions where the speech only helps only in locating the answer, 
but vision tells you what the answer is.
- so ASR is used as a reference cue, not as final answer source.

e.g What object is he holding when he says "look here" ?

# Execution Policy
The execution policy represents :
    "the execution plan for a one query"

In our system the query comes in and is changed into an execution policy.

Why do we need an execution policy, why not treat all queries as same.
(have the same execution policy for all queries instead of changing policies.)

- The problem with treating all queries the same is that different queries have different requirements for compute.


# Joint optimization idea.


Some videos e.g very small videos can take very low time, but some videos e.g long videos can take a lot of time. The ASR transcripts or windows can take a lot of time for 

# Related Work 

