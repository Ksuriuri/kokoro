"""Japanese voice + pitch-accent finetuning for Kokoro-82M.

    prepare  : wav + transcript -> phonemes, forced-aligned durations, F0
    train    : teacher-forced finetune (voice vector, accent-mark embeddings, model)
    export   : voicepack + kokoro-format weights + config.json
    infer    : synthesize with an exported model or a training checkpoint
    smoke    : sanity checks, runs on CPU
"""
