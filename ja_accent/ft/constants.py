REPO_ID = 'hexgrad/Kokoro-82M'

SAMPLE_RATE = 24_000
# One duration-predictor frame is 600 samples (25 ms); F0/energy curves run
# at twice that rate (hop 300, 12.5 ms).
SAMPLES_PER_FRAME = 600
FRAMES_PER_SEC = SAMPLE_RATE / SAMPLES_PER_FRAME
CURVE_HOP = SAMPLES_PER_FRAME // 2

MAX_PHONEMES = 510

# StyleTTS2 mel settings at 24 kHz.
MEL_N_FFT = 2048
MEL_WIN = 1200
MEL_HOP = 300
N_MELS = 80
