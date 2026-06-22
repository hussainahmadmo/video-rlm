# Why is it an important yet unsolved problem ?
Multimodal queries may require multiple sources of evidence, such as on-screen text from OCR, spoken content from ASR, 
and visual scene/action evidence from video captioning. However, many queries require only a subset of these modalities.

(The unsolved problem is that the scheduling of these modalities is inefficient)

Current systems schedules every stages of the multimodal workflows even if a query requires only 
a single modality, therefore increasing end to end delay.

(Mention an example of how its inefficiently handled by current systems.)

A common scheme of mu

 take the example of a narrated video where the narrator talks about the
travel destinations. For a query like "Describe the major travel destinations mentioned in the video"
the 

 that the video about Sydneys 

Take an example of the following queries whihc
However exisiting system typically rely on fixed execution pipelines 
that process these modalities in a predetermined order or uniformly across all queries
which is a bad idea because one 

# Introduction 
how is a workflow of a LLM different than a multimodal model, also say these serving systems 
optimize request level scheduling through prefill and decode.

(A workflow is a full sequence of stages the system executes to answer a multimodal question)

- user question + video/input to final answer

Multimodal systems built on large LLMs are AI systems that can understand and reason over multiple types of data(modalities). Not just text, but also images,audio, video and sometimes structured data. Due to the increase in complexity in tasks, current multimodal systems are increasingly becoming agentic.

The steps involved in reasoning are planning(user question + available tools) and synthesis(OCR text + ASR transcript + visual notes), which are later combinedinto one answer, the model is integrating evidence and generating a response. The reasoning process consists of two main stages: planning and synthesis.

In the planning stage, the model considers the users question together with available tools and decides what evidence to gather. 
In the synthesis stage, it combines information from OCR text, ASR transcripts, and visual observations to produce a final answer. 
In other words, the model integrates evidence from multiple sources and generates a response.

These systems solves multimodal tasks by executing workflows, these workflows are designed to invoke different models to solve tasks. 
We refer to each step in this workflow as a stage. A stage is single distinct step in a multimodal workflow, where it performs
a specific kind of work using a particular model or tool. For multimodal tasks, stage decomposition means breaking a single video question-answering task into seperate evidence gathering and reasoning stages, where each stage handles one kind of job, splitting an end-to-end task(videoQA) into modular steps for evidence retreival, transformation and reasoning. Within a single workflow, different stages often have a different computational requirements.(have to make more clear what multimodal workflows are and how they are different).

(this is workflow for LLM type systems you copied it from the paper - change it to follow a multimodal system)
For example a routing or summarization step may benefit from a small, fast model whereas a synthesis or code generation
step, requires a large more capable one.

(PLACEHOLDER - workflow for multimodal systems)

Motivated by the need for a new serving system that operates at workflow level to coordinate decisions across stages, we introduce XYZ
a serving system that sits on top existing LLM serving backends.

Our key insight is to seperate request execution from multimodal policy through X components: 

The workflow orchestrator that runs coordinates the stages of the workflow:

1 - 


Instead of treating the task as:
- video + question -> one giant VLM call -> answer. 

We decompose into stages like:
- query planning/understanding
- coarse retreival or localization
- OCR for on-screen text 
- ASR for spoken content
- visual inspection of selected segments
- temporal reasoning or evidence linking
- final synthesis.

For multimodal tasks, stage decomposition refers to breaking the end-to-end question answering process into specialized stages that seperately
handle planning, evidence localization, modality-specific extraction such as OCR or ASR, visual reasoning, and final synthesis. This decomposition 
allows the system to adapt computation to the needs of the query and coordinate heterogenous models and tools more efficiently.



In our setting, a stage is a unit of computation, such as planning to do what next. Running OCR on frames, transcribing speech with ASR, generating video captions, 
selecting relevant clips, synthesizing all evidence into final answer. ()

As multimodal tasks grow in complexity, with some tasks requiring spoken content, 
other tasks requiring OCR, and some requiring video captioning. Serving multimodal workflows requires two closely related challenges, addressing both heterogeneity
across workflow stages(different steps in the workflow are fundamentally different from each other).



different stages may rely on different modalities, models, and hardware backends,
and the system must coordinate their execution, data flow and resource management across the workflow. Inside a single workflow, one step maybe cheap and simple, while another
may be heavy and expensive. e.g ASR can be cheap and simple, on the other hand VLM can be heavy and expensive.



(heteroginity across workflow stages) - different steps in the workflow are fundamentally different from each other.


- run_asr needs an audio model - can be cheaper 
- run_ocr needs an OCR system - can be expensive - have to check all the videos.
- inspect_window may need a stronger VLM - inspecting window may need high quality VLM.
- coarse_search may use a cheap retreiver.

2 - coordination and decision-making across the components that execute them.
- once we have different stages and components, the system must decide.
- - which stage to run first
- - which model/backend should run each stage.
- - what can run in parelell
- - how to pass outputs from one stage to another 
- - when to preserve memory/cache and when to free it.



Instead of applying a fixed pipeline on data such videos, for several reasons.

First the input size for such systems can become too large. Video, audio and high resolution
images can produce a huge number of tokens or features. If a system tries to encode everything
uniformly and send it a large model the context becomes expensive and often infeasible. This 
causes token blowup, high memory usage, slow inference and large attention cost.

Second, not all parts of the input matter equally. In many multimodal tasks, only a small fraction
of frames, audio and regions matter. A fixed retrive-then-reason pipeline assumes relevance is 
obvious from the start, but in practice the model may need to form a hypothesis, then gather more
evidence, then revise that hypothesis.

Third (relevance revealed later)
Fourth (multimodal evidence is heterogenous)
Fifth (temporal and causal reasoning are hard)

fifth(temporal and causal reasoning are hard)
sixth(multimodal systems are deployed under system constraints)


Existing multimodal inference stacks are composed of applic





LLM based applications require multiple LLM requests to complete one task such as summarization, chain-summary, 
chat summary and multiagent coding, such tasks benefit from map reduce like workflow on text, splitting a long document into chunks, summarizing each chunk separately, 
and then combining those summaries into one final summary, even though such map-reduce like workflow can be used LLM workflows do not translate well to multimodal operations which have different compute requirement.
(Parrot)

Show one or two examples of the same workflows can fall short for different queries.

We observe that there exist significant opportunities to improve the end-to-end experience of multimodal applications by exploiting workflows embedded within multimodal requests.


First multimodal requests may depend on making multiple consecutive calls to an LLM, the result of one call could be direct input to another. 
Therefore its not only desirable to colocate these requests on the server side and execute them consecutively.(This )

Second existing systems optimize execution within a single multimodal request but do not exploit reuse across requests. In practice, many
multimodal queries share intermediate computations(e.g)

Second previous works retrieve relevant content per query, but does not optimize repeated multimodal workflows, across queries through
execution-time cache reuse.

Very important contrib(intermediate stage reuse)
- and cross request reuse.

## What is the differnce between the inferenece pipeline of text vs multimodal pipelines.

The key difference is that text-only inference starts with tokens already in text form
but for multimodal inference has to convert other.


for the input
modality of video, the modality extraction 

### Text only inference
A text only inference pipeline is much simpler which is:

text only pipeline = text --> tokens --> LLM --> output.

#### Compute structure


### Multimodal inference

Note - there is a difference between preprocessing such as audio extraction, video turning into frames which are all preprocessing steps that are required to do multimodal inference and on the other hand applying modality specific modelsto these inputs is something else. Dont try to confuse both.



Preprocessing operators - 

Modality operator means the processing steps that are specialized for each kind of input modality.

- e.g for the input modality of audio, the modality operator used is audio extraction, applying the wrong operator
on an input will result in not 

multimodal pipeline = raw media + preprocessing + modality operators

Modality operator = 


### Vision + Text processing

There are generatlly two types of MLLM architecture. 

1.decoder only architecture 
2.encoder-decoder architecture.

1.Decoder only models concatentate vision and text tokens and then feed them to the LLM as one shared sequence.
In terms of processing at each step the vision tokens are participating in each generation step, so they vision
tokens stay in the prefix throughout decoding.

This is why decoder-only models suffer from full image token unrolling, the image is expanded to many tokens, 
and those tokens go through the decoder like ordinary prefix token. Image tokens + visual tokens
both go through the decode self-attention phase as a result:
  - contribute to prefill cost
  - contribute to KV cache size
  - remain part of prefix during generation
(check modserve diagram in goodnotes for better intution)

2. Cross-attention / encoder-decoder-style models

Cross-attention models usually keep image tokens more separate from the decoder text stream. ModServe says that in these models, self-attention operates on text tokens, while the cross-attention layer attends to both text and image tokens. It also says these models improve inference efficiency by avoiding full image token unrolling in the LLM decoder.  ￼

So the main idea is:
	•	the image is first encoded by a vision encoder
	•	the resulting visual features/tokens are kept as separate visual memory
	•	the text-side decoder uses cross-attention to access them

Because of this, cross-attention models do not push all visual cost into one giant decoder prefix the way decoder-only models do. Instead, more of the visual cost shows up in:
	•	image encoding
	•	a smaller number of cross-attention layers  ￼


Typical flow of encoding an image by a VIT:
1. images are split
2. patches are fed into transformer
3. the vision encoder outputs feature vectors
4. the feature vectors are vision tokens
5. then they are passed to multimodal language model.


Vision tokenss are usually
ViT-H/14
	•	SigLIP
	•	InternViT


### self attention 
- in self attention the sequence attends to itself or other text tokens.

e.g what is the red car ? 
Each text token is can attented to the previous token.

### cross attention 
- in cross attention one sequence attends to a different sequence. 

e.g text tokens attends to visual tokens.

In cross attention for multimodal models:
  Q comes from the hidden state
  K, V comes from the image hidden states.

  e.g Suppose the query "what is the man holding"
  - the text token for holding can you use cross-attention to look at features corresponding
  the man holding, nearby objects and hand regions.

###


#### Multimodal bottlenecks

- There are multi-stage latencues in multimodal bottlencks. The stages that incur the most, 

Important



#### Compute structure of mm inference

Important to remember - whether encoding or LLM prefill dominates depends on the architecture. 

e.g In encoder/decoder models where there is cross attention happening.


# Reuse across requests 
## Cross request reuse(across queries)
- cross request reuse means - reusing work done 

## Intrarequest reuse(within workflow)
1 - Intraday-request reuse
  - - within one query
  - - overlapping internal calls


### Single user query
A request can be:
- a user query
- a follow-up query
- or an internal model call inside a workflow

  - e.g The user sees one query.
  e.g A request 1: What does the slide say at 2:10?
      A request 2: What example is shown on that slide ?
      A request 3: What does the professor say right after that ?

    Although they are different requests, but they overlap on:
    - same video
    - same time region
    - same decoded frames
    - maybe some OCR 
    - maybe same ASR chunk

    - OCR from request 1 can help request 2 


### Multiple user query
(multiple user queries can share the same across request think about this later cross users can 
be using the same video evidence so caching can help)

# Contrast with other(whats unique about your approach compared to others ?)
This should be clear cut contrast, previous work all miss X, and I use Y

We have videoRAG which

# Caching - what does it mean for video systems
For our system, caching would mean more than just generic KV cache. It should mean 
that the decoded video segment cache. ASR transcript cache, OCR result cache, visual embedding/caption cache, 
query-conditioned evidence when safe, in-flight deduplication, so duplicate work is not launched twice.
 

# Two level of caching is happening 

## Decode/preprocessing level cache

The cache is input-side intermediate representations derived from raw media.
- extracted full audio from video
- decoded frames 

# Core objective



Optimize jointly:
- answer accuracy
- video encoding cost.

How can answer accuracy be improved ? 

Answer accuracy can be improved if we choose the right windows(where the event is happening). Choosing the right window for visual queries is finding the right event.
- CLIP identifies events(similar to query), but suffers when the query asks for smaller detail.(for smaller detail)
- Choosing the right window to answer a question for ASR is non-trivial, unless the query explicity mentions in the
specified in the query.

Most existing video QA systems optimize only for answer accuracy and apply a fixed or heavy video-processing pipeline, even when a query requires a single modality or a cheaper configuration. Our contrib is to decide which modalities are necessary multimodal query and avoid paying the cost of unnecessary video processing, choosing a cheaper configuration that jointly optimizes answer accuracy and video encoding cost. Reducing the cost of video encoding and right model strength to answer it accurately while minimizing unneccesary.

Our goal is to develop a system that selects only the necessary video modality and strength for every query. 

In video QA, can benefit from dynamic video execution config, that specifies which modalaties to invoke, and how much video content to encode such as frame rate, resolution, number of frames, OCR/ASR usage and model strength. Most prior pipelines optimize for accuracy only and apply a fixed pipeline to every query, even when query only needs one modality or a cheaper configuration.

Prior work reduces latency mainly by shrinking the visual input. Our goal is broader, for each query, decide which modality to invoke, how much of the video to invoke, and which model to strength to use, so that latency and encoding cost are reduced without sacrificing the answer accuracy.

# Prior works
(Prior work papers are based on frame selection - or token reduction)

- our work is closer to VideoAgent on online adaptive inference.
- but we adapt the model config 
- different from VideoAgent because yours is bounded, cost aware, and execution-plan-driven.
- different from AdaVideoRAG because yours is less about retreival over offline knowledge stores and more about how to process the raw video now.


- RPS 
  - category - modality aware scheduling paper. 
  - context - other papers that it is related to is 

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
  - use CLIP to quickly score which frames or short video windows are most relevant to the question.
  (Removed the idea of using CLIP for coarse retreival as it could hurt downstream tasks such as OCR and ASR, so the updated system only uses CLIP if a query requires it) 
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

# Why does the ordering matters 


## Correct ordering
In our system the ordering matters, an is not an implementation detail. There are several reasons why ordering matters.

Cost: assume that we run expensive VLM first always(which is already limited by VLLM at 250 frames), then we waste alot of VLM calls.

- correct order: ASR -> localize -> visual inspect.
- bad order: visual inspect many windows -> maybe later ASR.

Latency: ordering determines the critical path.

- OCR -> cheap answer can 




# Limitations

Main Limitations of the current approach

1. Retreival bottleneck: CLIP is the single point of failure.

A major weakness of current system design is running OCR and ASR models on CLIP-retreived windows.

Key Point - we can select the right modalities ASR or OCR for a question, they may never see the relevant region if CLIP is producing garbage.


## Design Choices

Our system faces several key design choices. A primary design knob is **model selection**: given a video question, which model or tool is most appropriate for answering it accurately and efficiently?

* (Junchen: Industry is moving to one model only(a model that can do OCR+ASR+VLM model), you have to show that these ensemble of models can outperform the models that can do one rule of them all thing)

* (We are developing a better retreival and 




### 1. Model Selection

Different video questions require different capabilities. No single model is equally effective for all question types, so the system should select tools based on the information needed.

#### OCR for On-Screen Text

OCR is the appropriate tool when the question depends on **text visible in the video**, such as signs, subtitles, labels, or captions.

**Examples**
- Reading a street sign
- Extracting subtitles
- Identifying text shown on a screen

**Key observations**
- OCR is necessary for text-reading tasks; other tools do not substitute for it.
- OCR quality varies across models. Harder text questions may require a stronger OCR model.
- OCR performance also depends on frame selection: even a strong OCR model will fail if the sampled frames miss the relevant text or capture it poorly.

**Questions to consider**
- Do all OCR models handle difficult text equally well?
  - No. Some OCR models fail on challenging text because of model limitations, low image quality, or missed frames.
- Does frame sampling affect OCR output?
  - Yes. Poor frame sampling can significantly reduce OCR performance, even with a strong OCR model.

**Why other tools cannot replace OCR**
- **CLIP** is useful for frame filtering and semantic retrieval, but it cannot reliably read on-screen text.
- **ASR** is useful for speech recognition, not visual text extraction.
- **OCR** is specifically needed for text appearing inside frames.

**Additional note**
- Some OCR tasks may require not only text extraction but also localization (e.g., bounding boxes). CLIP cannot provide this functionality.

#### CLIP for Visual Semantic Retrieval

CLIP is useful when the task requires semantic matching between text and visual content.

**CLIP is useful for**
- Semantic visual retrieval
- Matching an image or frame to a text description
- Producing a coarse relevance score over frames

**Limitations of CLIP**
- It cannot reliably read text on screen
- It is not a substitute for OCR
- It does not perform speech recognition

### 2. Tool-Task Matching

The choice of tool should depend on the type of evidence needed by the question:

- **OCR**: on-screen text
- **ASR**: spoken language
- **CLIP**: semantic frame retrieval and coarse visual relevance

This suggests that model selection should be treated as a task-specific routing problem rather than assuming one model can solve all video questions equally well.



## 2.1 Similar Video Systems

When we give control of the tool usage to an LLM then our system becomes similar to VideoAgent.

Note whenever you want to use LLM to decide, tool usage for higher resolution, or completely give the control of the 
tool usage to an LLM then your system becomes similar to videoagent.

## 3.Experiments

Our work is different than request level scheduling. 
### 3.1 Experiment Questions

Q1 Single Request Visual Cost scaling experiment 

### 3.2 Systems Compared

## 3.3 Workloads / Current Experiment Settings

### Video lengths

The experiments use three duration settings for each video:

| Video size | Duration |
|---|---:|
| Small | 10 seconds |
| Medium | 60 seconds |
| Large | 600 seconds |

Each query is run under two policies:

| Policy | Description |
|---|---|
| `baseline_joint_heavy_60f` | Runs ASR on every 5-second chunk across the full video. | Runs VLM visual inspection on every 5-second chunk across the full video. | Full audio + full visual sweep. Highest coverage, highest cost. |

| `encoder_aware_60f` | Runs ASR over the full video first to localize relevant chunks. | Runs VLM visual inspection only on top selected ASR-localized windows. | Full audio sweep + targeted visual inspection. Lower visual cost. |

---

## 3.4 Metrics

For each run, we record:

| Metric | Meaning |
|---|---|
| `workflow_e2e_s` | End-to-end workflow latency. |
| `profile_interval_s` | Time spent in the query profiling stage. |
| `asr_interval_s` | Wall-clock interval covered by ASR stages. For batched ASR, this is measured as the start of the batch to the end of the batch. |
| `visual_interval_s` | Wall-clock interval covered by visual inspection stages. |
| `answer_interval_s` | Time spent in final answer generation. |
| `explained_interval_s` | Sum of non-overlapping stage intervals. |
| `unexplained_e2e_s` | End-to-end time not explained by measured stage intervals. |
| `total_sampled_frames` | Number of visual frames sent to the VLM. |
| `contains_expected` | Whether the model answer contains the expected answer string. |
| `manual_correct` | Optional manual correctness label. |

We distinguish between two timing views:

1. **Stage wall sum**: sums all individual stage runtimes. This can exceed end-to-end latency when stages run concurrently.
2. **Stage interval time**: computes the non-overlapping interval for each stage bucket. This better explains end-to-end latency.

---

## 3.5 E1: Cost Structure / Mechanism Experiments

### E1.1 Server-side video frame sampling and visual encoder cost

When serving Qwen2.5-VL with:

```bash
--media-io-kwargs '{"video": {"num_frames": 512}}'


### 3.5 E1: Cost Structure/ Mechanism Experiments

#### E1.1 Server side video frame sampling and visual encoder cost
To understand the cost structure 


Observed VLLM behavioir(IMPORTANT) - 

When serving QWEN2.5_VL with 
- --media-io-kwargs '{"video": {"num_frames": 512}}'

VLLM treats num_frames = 512 as an upper bound for server-side video sampling.
This upper bound is a serving-time config, and in practice its must be chosen
based on the available GPU memory, model context length and acceptable latency.

A larger frame cap allows the server to encode more video evidence, 
but it also increases the number of patches/tokens, visual encoder time, 
LLM prefill cost and KV cache memory.

For short clips, vLLM does not upsample the video to the configured cap. be 
The effective number of frames is bounded by the available video frames:

effective_frames = min(requested_frame_cap, total_available_frames)

For QWEN2.5VL, "temporal_patch_size=2", the sampled frame count is rounded down
to an even number.

The temporal grid size is then:

"grid_t = sampled_frames / 2" and the final visual token count is:
"visual_tokens = grid_t * grid_h * grid_w / merge_size2

In our experiment, `grid_h=26`, `grid_w=46`, and `merge_size=2`, so the system
produced 7,176 visual tokens for 48 sampled frames and 76,544 visual 
tokens when saturated at 512 sampled frames.


Changes in the 60fps code is not there.


### 3.6 E2: Routing Experiments


### 3.7 E3: Scheduling Experiments


### 3.8 E4: Joint Optimization techniques


### 3.9 E5: Request-Level vs Workflow level Comparison


### 3.10 Representative Figures 

### 3.11 Threats/Confounds





The number of frames that are controlled by the VLLM on the server side governs 
how much frames we can use.




Want to prove two things:
1 - better end to end latency
2 - better throughput

A generic system will treat a query as independent generic job:
-> Query arrives
-> submit ASR job
-> submit OCR job
-> submit visual job
-> later submit answer job(which will depend on all the above)

Key takeaway - a generic system does takes all the available tasks, and put them in a queue. If want to show that ordering matters, then we want to show that naive request-centric,


# Multimodal pipeline

Vision tokens are the sequence of vector embedding produced by the vision encoder from an input
image(typically from images patches or tiles). These tokens represent visual features
(e.g shapes, features, objects) and are concatenated with text tokens as input to LLM.

## Figure 

### Standard MM pipeline

Baseline
Encode everything --> many tokens --> expensive prefill
[ Image ]
     ↓
[ Preprocessing ]
     ↓
[ Vision Encoder ]
     ↓
[ Vision Tokens: v1, v2, ..., vN (~6k–7k) ] #1 image can be around 7K tokens
     ↓
                +----------------------+
[ Text Prompt ] |  Concatenation       |
                +----------------------+
                         ↓
            [ Combined Tokens: N + M ] (Here is where attention happens) 
                         ↓
                [ LLM Prefill ]
                         ↓
                [ KV Cache built ]
                         ↓
                [ LLM Decode ]
                         ↓
                    [ Output ]

### Our system

Encode SELECTED --> fewer tokens --> cheaper prefill

[ Image / Video ]
        ↓
[ Cheap Signal (CLIP / ASR / OCR / heuristic) ]
        ↓
[ Select relevant region / window ]
        ↓
[ Vision Encoder (partial) ]
        ↓
[ Few Vision Tokens ]
        ↓
        +----------------------+
[ Text ]|  Concatenation       |
        +----------------------+
                 ↓
         [ Smaller sequence ]
                 ↓
            [ LLM ]

### Side by side comparision 

Baseline:                     Your System:

Image                         Image
 ↓                            ↓
Encode ALL                    Select region
 ↓                            ↓
7000 tokens                   500 tokens
 ↓                            ↓
LLM prefill (expensive)       LLM prefill (cheaper)


Important to have the contrast figure where 




### 
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

# Junchens concern


Currently the trend is going that there is a one unified model that can do everything, why would
your approach be using different models be better ?

Even if one model can do everything e.g speech, OCR, visual reasoning : 
- it may still be worse than running on every query because :
-- latency is higher.
-- cost is higher.
-- unnecessary computation is done on easy queries.


Even if there exist models that perform well across speech, text, and vision, our argument is still that running the same heavy multimodal pipeline for every query is not always the right systems design. Different queries require different evidence and different amounts of computation. Our framework is meant to adaptively allocate compute and choose the right processing path. In one instantiation, that could be an ensemble of specialized models; in another, it could include a strong unified multimodal model as one of the available tools.


# General Observations


- Many of the current design choices reduce to a small set of core knobs:
  - whether to encode at all,
  - whether to use denser or higher-resolution visual processing,
  - and which modality should be invoked first.

- A key open question is whether a simple classifier could make the same routing decisions as the current LLM-based router.
  - Right now this is unclear.
  - We need to test whether lightweight query classification is sufficient, or whether the LLM router provides materially better decisions.

- This leads to the broader systems question:
  - **Why do we need LLM-based routing at all?**
  - We should justify what the LLM router adds beyond simple heuristics or fixed query-type classification.

## Model-specific notes

- **Qwen/Qwen2.5-VL-3B-Instruct** works well for OCR-style questions.
  - Current OCR model:
    - `ocr_model = "Qwen/Qwen2.5-VL-3B-Instruct"`
  - However, its structured JSON output can still be unreliable in some cases.

- We should explicitly identify which models are best for:
  - OCR quality,
  - reliable JSON output,
  - and ASR quality.

- Action item:
  - **Check which model performs best for ASR tasks.**

## Routing failures

- For some ASR-heavy questions, the wrong model is currently selected.
  - Example: `c6fuIEzOZ2E.mp4`
  - Even when the question should clearly route to an ASR-centered pipeline, the current system does not choose one.

- This suggests the router is still over-relying on visual retrieval in cases where the answer is primarily in speech.

## CLIP evaluation questions

- We need a direct comparison of when CLIP should and should not be used.
- In particular:
  - Is `CLIP + ASR + OCR` strictly better than using only one of them?
  - Or does CLIP sometimes hurt by incorrectly gating downstream processing?

## Failure mode: CLIP misses fine-grained visual details

- CLIP can miss small or fine-grained objects.
  - Example: `jJGbXCCU5yc_1`
  - In this case, CLIP fails to recognize the camera and instead misclassifies it as an umbrella.

- This suggests an important visual failure mode:
  - for small objects or subtle details, CLIP can fail when frame sampling is too sparse or the representation is too coarse.

- We should make a slide showing:
  - how CLIP misses low-level details,
  - and why higher-resolution or denser frame sampling is necessary for these cases.

## Failure mode: spoken-answer questions

- For spoken-answer questions, CLIP can miss the relevant regions entirely because the answer is not visually salient.
  - Example: `athabNMGceo_0`

- In these cases, the speech carries the answer, but CLIP is still being used as the primary retrieval mechanism.
  - This can cause the system to localize the wrong windows.

- Example question:
  - “Against a blue background, a man wearing black-framed glasses and a white short-sleeve shirt with a small bird pattern is explaining. Which of the following animals evolved hindgut fermentation?”

- For this class of questions, we likely need:
  - **ASR-first search**, not CLIP-first search.

## But ASR-only search also has limits

- We should also make slides explaining why **ASR-only retrieval is not always sufficient**.
- ASR-first search can fail when:
  - the transcript does not explicitly mention the answer,
  - the speech only indirectly refers to a visual object,
  - or the final answer still requires visual grounding.

- So the argument should not be:
  - “CLIP bad, ASR good.”
- Instead, the argument should be:
  - different question types require different first-stage retrieval strategies.

## End-to-end pipeline risk

- A major systems problem is that if CLIP retrieval is wrong, then every downstream stage that depends on those windows can also fail.
- This is especially damaging when OCR or ASR are run only on CLIP-selected regions.

- In other words:
  - **CLIP is currently acting as a gatekeeper.**
  - If it misses the right evidence, downstream OCR, ASR, and VLM reasoning never even see the correct region.

- Example failure:
  - question about the man working with a log:
    - “On the golden ground with black and gray caves and scattered wood, what did the man, who appeared for the first time when a shirtless man with a cloth strip tied around his waist and a beard appeared, do?”
  - Ground truth:
    - “lift a tree trunk with both hands”
  - CLIP-only retrieval instead leads to an incorrect higher-level interpretation:
    - “The man appears to be performing a ritual or ceremony involving the wooden logs and the cave.”

- This is a good concrete example of how incorrect retrieval can poison the full downstream pipeline.

## Systems question

- We should analyze whether, in our current pipeline, **encoding is actually the dominant cost**, or whether reasoning/inference cost is already comparable or larger.
- This matters because our systems story depends on where the true bottleneck is:
  - encoder cost,
  - retrieval cost,
  - or expensive reasoning cost.

## Scheduling idea

- Ranking execution configs differently under different load conditions is a good idea.
- Under low load:
  - we may want richer multimodal configs.
- Under high load:
  - we may want cheaper configs with early-exit paths.
- This gives us a principled systems story around adaptive compute allocation.

## Sharper takeaway

- Fixed modality pipelines are brittle.
- CLIP-first retrieval is not universally reliable.
- Small-object questions may require denser or higher-resolution visual inspection.
- Spoken-answer questions often require ASR-first retrieval.
- ASR-only retrieval is also insufficient for visually grounded questions.
- If the first retrieval stage is wrong, the full downstream pipeline can fail.
- Therefore, the system should route adaptively across modality and compute level rather than committing to one fixed retrieval path.




## ASR chunk execution behaviour

For a single video the ASR pipeline first splits the video into fixed length audio chunks, then submits those chunk transcription
jobs concurrently using a thread pool. 

e.g 0-30 30-60 60-90 90-120
    []    []    []    []



## Load vs concurrent ASR chunk requests

![Latency vs concurrent ASR chunk requests](/dataheart/hussainahmad/video-rlm/minimalv2/outputs/asr_load_sweep/latency_vs_concurrent_chunk_requests.png)

### Interpretation

- Blue: avg ASR chunk request latency, exclude chunk extraction
- Orange: average end-to-end latency
- As load increases, both curve rises sharply.
- Since chunking is excluded from blue, the slowdown is mainly in the ASR serving path.
- ASR becomes a major bottleneck at high concurrency.


The blue curve shows average ASR chunk request latency measured from client
send to client response, excluding local chunk extraction. The orange curve shows 
average end-to-end video latency.

### Experiment
[] Verify that different models that can do well on the accuracy, can do well
on.

- Experiment 2(Apr 2) - we do a comparision for checking whether ASR/video ratio is same by processing both the 
video and the asr in 30 seconds chunks, this comparision is wrong because when we send a video even though we send
a 10 min = 300 second chunk.


## Observations

A tiny ASR compared to a better model can miss subtle speech details.
(Watch more videos to confirm).(We did see this ASR small was better than tiny)

Its not necessary to do do heavy captioning for each query.
- e.g 

Another thought you can add is for late fusion questions maybe if the GPU has alot of resouces and is under low load we can do finer more robust scheduling.

The planner decides the workflow, and the workflow(which is an actual graph of steps). The workflow is an execution structure 
goes throgh a pruning process such 

# Terminology

Pruning = turning a huge search space into a tiny focused region using cheap signals and iterative refinement.
Cheap signals - fast, low cost signals used to decide where to look next and what skip.

signal type: semantic + temporal

OCR(on screen text)
- semantic + spatial

What it gives: 
- text-on-screen
- labels, signs, UI elements
- structured clues





In our system the cheap signals are :
- semantic 


### Bugs not fixed until now 

[] Currently in the run_asr_num_chunks_benchmark.py the chunking can be missed on 
the chunks border e.g if the chunks can 0-30, then we miss stuff on the chunk border. This can be seen in 
the -eRimFrm6kQ.mp4 video.

Currnet VIdeoMMe tasks are not pure ASR e.g
uses some visual to answers so cannot use a 
better model ASR does not make different to make 
accuracy argument. Need to have full ASR datasets.

### Comparision table

| Synthesis method | Role of audio | Role of video | When to use | Example query |
|---|---|---|---|---|
| **ASR-only** | Primary evidence source | Not used | Answer is directly spoken | “Which country does he mention first?” |
| **Visual-only / Single-window** | Not used | Primary evidence source | Answer is visible in one window | “What color is the car?” |
| **OCR-only** | Not used | Only used to supply frames for OCR | Answer is on-screen text | “What text appears in the white box?” |
| **Anchor-then-inspect** | Temporal anchor | Final answer source | Speech/text tells you when to look, vision tells you what happened | “What is he holding when he says ‘look here’?” |
| **Late-fusion joint synthesis** | One aligned evidence stream per window | One aligned evidence stream per window | Both modalities matter and should be combined per window | “What does he say while showing the object?” |
| **Window rerank** | Optional support for each candidate window | Optional support for each candidate window | Several windows may contain the answer; pick the best-supported one | “Which window best answers the question?” |
| **Map-reduce-like** | Per-window transcript or speech summary | Per-window visual summary | Long videos, distributed evidence, multi-window reasoning | “What happened before and after the announcement?” |
| **Coarse-to-fine** | Optional cheap anchor/filter | Expensive final inspection only on selected windows | Full visual processing is too expensive; only inspect promising windows | “Find the moment where the small object appears.” |
| **Mixed full** | Full speech evidence | Full visual evidence | Hard multimodal queries needing joint reasoning | “What does the speaker say when the red sign appears?” |


# Dataflows
![Fixed multimodal workflow](../diagrams/fixed-mm.md)


# Junchen Questions

Why is it an important yet an unsolved problem.

Multimodal queries may requires multiple source of evidence such as, on screen text from OCR, spoken content from ASR, and visual scene/action evidence from video captioning or only require a single source of evidence. In either case, todays LLM services expose a request level API for multimodal request, only seeing isolated requests, missing information about multimodal requests.


- Have to drive the point of current systems optimize 


- also all questions dont require the same workflow have to drive this point too.

- have to drive the point of how the current workflows scheduled in a way that 
are not optimimal for end to end performance.



# Text Based workflows

Task - summarize long meeting transcript, as the transcript is too long for one-prompt

e.g we have 10000 context window, but the meeting is 4x the context window size.

- Request 1 - summarize chunk 1 


# Consecutive LLM requests in our system

A request is consecutive when requests have multiple dependencies, such that the
the output of one request is an input to the other.

We classify a query being consecutive if the output of a one request is an input to the 


## ASR-anchored visual workflow
Query
--> ASR over chunsk
--> anchor extractor/lc


# Concepts 

## Pipelining in MM inference 

overlap - two or more pieces of work happen during the same period of time.

Pipelining means overlapping stages of different requests such that the system resources in this case the GPU is busy.

Note - the overlapping of stages for a single GPU is not what you think it is. The first idea that comes to mind is that if we 
have a multimodal request, we can use a 

Note - a GPU  single kernel that it can use to parellize work.

A single GPU is like one worker.
- so if we have a single MM request(encode, prefill, decode)
- then the GPU has to do the following
- At time = 1 it has to do encode
- At time = 2 it has to do prefill
- At time = 3 it has to do decode.

And the second request starts at time = 4.

Key takeaway - on a single GPU we mostly get sequential execution, better interleaving and not true overlap.
A single GPU cannot execute multipel inference stages. It can only switch between them. So for overlap of execution 
across stage(we re)

Overlap needs multiple GPUs, because one GPU alone time-shares work instead of running 
multiple stages truly in parallel.




e.g 

In the case of one GPU, the pipelining of a stage 

there is no stage pipeling for multimodal requests, but 
 




# Contrast work

## Category 1 - Accuracy-focused 
- VideoRAG
- VideoAgent 

Optimizes correctness/reasoning

## Category 2 - Single Request Systems
- Modserve
- token pruning papers(after encoding remove tokens that are not relevant to prefill)

Optimizes one query at a time.

## Category 3 - Multi-request workflow systems
- Parrot 
### ElasticMM

challenge 1 - fixed split of GPUs across text vs multimodal is
often wrong(performs poorly under bursty, changing workloads)

fix 1 - rebalance GPUs depending on current demand.
(modality aware load balancing fixes the problem of changing workloads by dynamically
reallocating resources between groups.)

challenge 2 - 

- Our system(partially)

Optimize throughput/scheduling.


## Category 1 - Optimizing the execution efficiency of MM queries.

optimizing how one query is executed as a computation which focuses on :
- what operators to run 
- which modality to use 
- how much evidence to inspect 
- which frames/windows/chunks to process
- whether to prune/retreive or stop early.


## Category 2 - Optimizing the execution efficiency of multimodal query serving.


Examples - 
Parrot
ElasticMM

Main goal: 

improve the serving effeiciney of many requests under resource constraints.

Typical techniques used -
- scheduling 
- batching 
- load balancing
- decoupling stages 
	•	ElasticMM / ModServe optimize multimodal query serving: stage decoupling, autoscaling, routing, batching, TTFT, throughput, GPU allocation.  ￼




