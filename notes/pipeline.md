flowchart TD
  A[Video] --> B[Chunker<br>(seg = 60s)]
  B --> C[Cheap Pass<br>1 FPS + Motion + CLIP]
  C --> D{Need more evidence?}

  D -- No --> Z[Answer + Provenance<br>(time ranges + evidence)]
  D -- Yes --> E[Select Segment / Window<br>(top-k + peaks)]
  E --> F[Mid-Cost Captioner<br>(window captions)]
  F --> G{Still ambiguous?}

  G -- No --> Z
  G -- Yes --> H[Expensive VLM / VQA<br>(high-res, short window)]
  H --> I[Judge / Critique<br>(hypothesis + missing evidence)]
  I --> D