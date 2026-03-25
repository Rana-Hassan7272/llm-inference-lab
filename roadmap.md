Project 4 — LLM Inference & Optimization
Total time: 8 weeks

Phase 1 — Foundation Setup
Weeks 1
Goal: Environment ready, first model running locally, first inference working.
Step 1 — Project structure
Create this folder structure first, nothing else:
llm-inference-lab/
├── models/
├── inference/
├── optimization/
├── benchmarks/
├── api/
├── dashboard/
├── configs/
├── results/
├── notebooks/
└── docker/
Step 2 — Environment setup
Install these exactly in this order:
pip install torch transformers accelerate
pip install bitsandbytes auto-gptq
pip install fastapi uvicorn
pip install mlflow locust
pip install pandas numpy matplotlib
Step 3 — First model running
Start with TinyLlama on Colab. Do not start with Mistral 7B yet — you will hit memory errors and get frustrated. TinyLlama runs on free T4 in under 2 minutes. Goal is just to see tokens generating. Nothing fancy yet.
Step 4 — Basic inference script
Write inference/base_inference.py that does exactly three things:

Loads model
Takes a text prompt
Returns generated text with timing

Record your first numbers manually in a notebook. How long did it take? How much memory?
Deliverable at end of Phase 1:
TinyLlama running, generating text, first latency number recorded.

Week 2
Goal: Understand and measure the full quantization tradeoff triangle — speed vs memory vs quality.
Step 1 — FP16 baseline
Run TinyLlama and Mistral 7B (on Colab) in full FP16. Record exactly:

GPU memory used
Time to first token
Tokens per second
A quality score — run 20 fixed questions, score answers 1-5 manually

This is your baseline. Every other experiment is compared against this.Phase 2 — Quantization Experiments

Step 2 — 8-bit quantization
Use bitsandbytes to load same model in 8-bit. Run identical 20 questions. Record same metrics. Write honest observations — where did quality drop? Where did it hold?
Step 3 — 4-bit quantization
Load in 4-bit with bitsandbytes. Same process. Same 20 questions. Record everything.
Step 4 — Build comparison table
Write a Python script that reads all your recorded results and outputs a clean comparison:
Mode    | Memory  | Latency | Tok/sec | Quality
FP16    | 14.2 GB | 180ms   | 45      | 4.6/5
8-bit   | 8.1 GB  | 195ms   | 41      | 4.3/5
4-bit   | 5.8 GB  | 165ms   | 52      | 3.9/5
Step 5 — Write your findings
This is critical. Write 300 words in a markdown file explaining what you found and why it happened. This becomes your README content and your interview talking points.
Deliverable at end of Phase 2:
Complete quantization comparison with honest quality measurements and written analysis.

Phase 3 — KV Cache and Batching
Week 3
Goal: Understand the two most important inference optimization techniques deeply.
Step 1 — KV Cache experiment
Write optimization/kv_cache_experiment.py that runs the same prompt twice — once with KV cache disabled, once enabled. Measure generation speed at different sequence lengths: 128, 256, 512, 1024 tokens.
Plot the results. The longer the sequence, the more dramatic the difference. This graph is extremely impressive in your README.
Step 2 — Static batching
Write optimization/batching.py that takes multiple prompts and processes them together. Measure throughput at batch sizes 1, 2, 4, 8. Record tokens per second for each.
Step 3 — vLLM setup
Install vLLM on Colab. Run the same batching experiment using vLLM's dynamic batching. Compare your manual batching results against vLLM. Document the difference honestly.
Step 4 — The adaptive router (this is your unique contribution)
Write inference/adaptive_router.py — a system that classifies incoming requests and routes them to different model configurations:
python# Simple but impressive logic
if prompt_length < 50 and task_type == "simple":
    use 4-bit, fast
elif task_type == "reasoning":
    use 8-bit, balanced  
else:
    use FP16, full quality
```

This one file turns your benchmarking project into an intelligent system. Spend real time here making it clean.

**Deliverable at end of Phase 3:**
KV cache graphs, batching comparison, and working adaptive router.

---

## Phase 4 — Streaming API
### Week 4

**Goal:** Production-ready FastAPI server with streaming responses.

**Step 1 — Basic FastAPI server**
Write `api/app.py` with three endpoints:
```
POST /generate          — standard generation
POST /generate/stream   — streaming tokens  
GET  /benchmark/{model} — run benchmark for a model
Step 2 — Streaming implementation
Implement proper token streaming using FastAPI's StreamingResponse. User sees tokens appear one by one exactly like ChatGPT. This is not hard but looks very impressive in a demo video.
Step 3 — Model manager class
Write inference/model_manager.py that handles loading, caching, and switching between different model configurations cleanly. This shows software engineering maturity, not just ML knowledge.
Step 4 — Connect adaptive router to API
Every incoming request to /generate passes through your adaptive router first. The router decides which model configuration to use. Log every routing decision to a file.
Step 5 — Test everything
Write tests/test_api.py with basic tests using pytest. Test that each endpoint returns correct format. Tests do not need to be comprehensive — just enough to show you write tests.
Deliverable at end of Phase 4:
Working streaming API with adaptive routing and basic tests.

Phase 5 — Benchmark Suite and Dashboard
Week 5
Goal: Professional benchmark system and visual dashboard.
Step 1 — Benchmark runner
Write benchmarks/runner.py that automatically runs all experiments and saves results to JSON:
python# One command runs everything
python benchmarks/runner.py --model tinyllama --modes fp16,8bit,4bit
```

**Step 2 — Load testing**
Use Locust to simulate multiple users hitting your API simultaneously. Test with 1, 5, 10, 20 concurrent users. Record how throughput changes. This is real production thinking.

**Step 3 — MLflow integration**
Log every experiment to MLflow:
- All hyperparameters
- All metrics
- Model artifacts
- Benchmark plots

Run `mlflow ui` — you now have a professional experiment dashboard.

**Step 4 — React dashboard**
Build `dashboard/` as a React app showing:
- Live benchmark comparison table
- Memory vs quality scatter plot
- Latency distribution chart
- Routing decision log

Keep it clean and simple. Four charts. No unnecessary complexity.

**Deliverable at end of Phase 5:**
One command runs all benchmarks, results visible in MLflow and React dashboard.

---

## Phase 6 — Docker and Deployment
### Week 6

**Goal:** Everything runs with one command anywhere.

**Step 1 — Dockerfile**
Write a Dockerfile that installs all dependencies and starts the FastAPI server.

**Step 2 — docker-compose**
Write `docker-compose.yml` with three services:
```
api        — FastAPI inference server
mlflow     — experiment tracking UI
dashboard  — React benchmark dashboard
```

One command: `docker-compose up` — everything starts.

**Step 3 — Deploy API to Railway**
Deploy your FastAPI inference server to Railway free tier. Use TinyLlama since it fits in free memory. You now have a live URL.

**Step 4 — README**
Write a professional README with:
- Architecture diagram
- Benchmark results table
- Live demo URL
- One command setup instructions
- Your written analysis of findings

**Step 5 — Demo video**
Record 90 seconds showing:
- Streaming response in action
- Adaptive router making a routing decision
- Benchmark dashboard with comparison charts

**Deliverable at end of Phase 6:**
Live deployment, professional README, demo video. Project 4 complete.

---
---

