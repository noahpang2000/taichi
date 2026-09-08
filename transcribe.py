import os
import whisper_timestamped as whisper
from whisper.utils import get_writer
import time
import opencc

AUDIO_DIR = "./data/Liu JiShun Tai Ji Class" 

OUTPUT_DIR = "./data/taichi_transcripts_timestamped"

# This prompt helps the AI understand the context to avoid mistranslating homophones.
# We include common Taichi terms so the model is biased toward martial arts vocabulary.
INITIAL_PROMPT = "這是一段關於太極拳、内家拳、推手、發勁、氣勢、虛實和鬆沉的講課錄音。"

def main():
    print("Loading Whisper model... (This may take a moment)")

    model = whisper.load_model("turbo") 
    # Initialize OpenCC converter (s2t = Simplified to Traditional)
    converter = opencc.OpenCC('s2t')

    srt_writer = get_writer("srt", OUTPUT_DIR)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith(".mp3"):
            audio_path = os.path.join(AUDIO_DIR, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
            
            if os.path.exists(output_path):
                print(f"Skipping {filename}, transcript already exists.")
                continue
                
            print(f"Transcribing {filename}...")
            start_time = time.time()
            
            try:
                audio = whisper.load_audio(audio_path)

                result = whisper.transcribe(
                    model,
                    audio,
                    language="zh",
                    initial_prompt=INITIAL_PROMPT,
                    condition_on_previous_text=False,  # Stops errors from snowballing into future segments
                    compression_ratio_threshold=2.4,   # Detects repetition loops and triggers fallbacks
                    no_speech_threshold=0.6
                    # fp16=torch.cuda.is_available()
                )

                # Force the entire output string into Traditional Chinese
                final_text = converter.convert(result["text"])

                # Save raw text (for RAG)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(final_text)

                # Save SRT file (for reference)
                for segment in result["segments"]:
                    segment["text"] = converter.convert(segment["text"])
                    if "words" in segment:
                        for word in segment["words"]:
                            word["text"] = converter.convert(word["text"])

                            # whisper_timestamped outputs word data to "text" key, but official whisper get_writer expects it in "word" key
                            word["word"] = word["text"]
                srt_writer(result, audio_path, {})
                
                elapsed = time.time() - start_time
                print(f"Finished {filename} in {elapsed:.2f} seconds.")
                
            except Exception as e:
                print(f"Error transcribing {filename}: {e}")

if __name__ == "__main__":
    main()