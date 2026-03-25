# Phase 2 Findings: Quantization Trade-offs (FP16 vs 8-bit vs 4-bit)

This phase measured the expected quantization trade-off triangle: memory, latency/speed, and answer quality. Across 20 fixed questions per mode, the aggregated results were: FP16 (2.10 GB, 91.8 ms TTFT, 31.1 tok/s, 1.5/5), 8-bit (2.09 GB, 149.3 ms TTFT, 8.7 tok/s, 1.55/5), and 4-bit (2.09 GB, 101.5 ms TTFT, 15.3 tok/s, 3.1/5). The most surprising outcome is that 4-bit scored much higher quality than FP16 and 8-bit, which is opposite of the usual expectation (normally FP16 >= 8-bit >= 4-bit on quality).

For memory, all three modes are almost identical in this run (difference ~0.01 GB). That is far smaller than typical quantization savings and suggests this metric is likely capturing only partial runtime memory or that the loading configuration did not fully reflect expected FP16 vs quantized footprints. In other words, memory numbers are valid for this setup but should not be overgeneralized.

For speed, FP16 was fastest in throughput and had the best latency. 8-bit was the slowest by a large margin, while 4-bit was in between. This is also somewhat atypical for many GPUs where quantized inference can improve throughput, but it can happen depending on kernel support, model architecture, and runtime implementation details.

Quality trends indicate a probable pipeline/configuration issue rather than pure model capability differences. Many answers in FP16 and 8-bit contain repeated template tokens (for example instruction markers and malformed chat fragments), which likely dragged manual scores down. 4-bit outputs, while still imperfect, were more directly answer-like in multiple questions. This strongly suggests prompt formatting, generation settings, or chat-template handling had more impact than precision alone.

Conclusion: this experiment successfully captured measurable differences, but it also revealed that inference pipeline correctness is critical before drawing final quantization conclusions. The next improvement should be to standardize prompt formatting and rerun a smaller validation set to confirm whether quality ordering becomes more realistic.
