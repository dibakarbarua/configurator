# Documentation preferences

- Use concise bullet points for explanatory documentation instead of prose paragraphs.
- Keep tables for metadata maps and comparisons, and code blocks for code flow and equations.
- Lead performance documentation with the model: compare total model compute with memory movement to estimate roofline TTFT and TPOT. Explain that the code calls TPOT ITL.
- Annotate code-flow steps with their purpose: estimate compute ceiling, estimate memory movement ceiling, calculate FLOPs per byte moved, and compare arithmetic intensity with hardware specifications to derive roofline metrics.
- Distinguish workload counts (FLOPs and bytes) from hardware rates (FLOP/s and bytes/s) when explaining ceilings.
- Explain capacity checks in terms of what is fixed, what is calculated, and whether the check applies to pooled GPUs or individual workers. State whether weights, activations, and KV cache are included.
- Briefly explain concurrency-limit calculations, throughput searches, SLO filtering, and returned metrics when showing their place in the code flow.
- Identify the files, functions, and metadata fields associated with each calculation. Explain resource types and their aggregation behavior.
- Keep documentation faithful to the current implementation and distinguish implemented behavior from planned extensions.
- Incorporate inline review feedback into the document for further review.
