def select_vocab_mapping_names(llm_path):
    llm_name = str(llm_path).lower()
    if "qwen2.5" in llm_name or "qwen2_5" in llm_name:
        return (
            ["opt_to_qwen2_5.json", "gpt2_to_qwen2_5.json", "llama_small_to_qwen2_5.json", "bloom_to_qwen2_5.json"],
            ["qwen2_5_to_opt.json", "qwen2_5_to_gpt2.json", "qwen2_5_to_llama_small.json", "qwen2_5_to_bloom.json"],
        )
    if "gemma" in llm_name:
        return (
            ["opt_to_gemma.json", "gpt2_to_gemma.json", "llama_small_to_gemma.json", "bloom_to_gemma.json"],
            ["gemma_to_opt.json", "gemma_to_gpt2.json", "gemma_to_llama_small.json", "gemma_to_bloom.json"],
        )
    if "llama" in llm_name:
        return (
            ["opt_to_llama.json", "gpt2_to_llama.json", "llama_small_to_llama.json", "bloom_to_llama.json"],
            ["llama_to_opt.json", "llama_to_gpt2.json", "llama_to_llama_small", "llama_to_bloom.json"],
        )
    raise ValueError(f"Unsupported llm_pretrained for vocab mapping selection: {llm_path}")
