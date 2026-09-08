import os
import re
from collections import Counter

# Point this to the folder where your transcribed .txt files are
TRANSCRIPT_DIR = "./data/taichi_transcripts_timestamped"
CLEANED_DIR = "./data/cleaned_transcripts"

# A dictionary of specific, known transcription errors to fix.
# Key: The bad text to find. Value: The corrected text.
# We use longer phrases to avoid accidentally replacing legitimate uses of "心".
CORRECTIONS = {
    "有股心": "有股勁",
    "內心打自己": "內勁打自己",
    "發心": "發勁",
    "聽心": "聽勁",
    "懂心": "懂勁",
    "懂精": "懂勁",
    "聽進": "聽勁",
    "化嗯": "化", 
    "化呢": "化",  
    "單田": "丹田",
    "命人": "命門",
    "寒兄八輩": "含胸拔背",
    "護臀": "護肫",
    "三藏": "三陽",
    "皇子": "擋子",
    "當天": "丹田",
    "堅果": "肩骨",
    "用皺提": "用肘提",
    "用堅提": "用肩提"
}

def audit_taichi_terms():
    """
    Scans transcripts to see what characters commonly follow Taichi action words.
    This helps discover new errors to add to the CORRECTIONS dictionary.
    """
    if not os.path.exists(TRANSCRIPT_DIR):
        print(f"Error: Directory {TRANSCRIPT_DIR} not found.")
        return

    # Look for what character immediately follows these common prefixes
    prefixes_to_check = ["發", "聽", "懂", "借", "內", "化", "引", "護"]
    results = {prefix: Counter() for prefix in prefixes_to_check}

    print("Auditing transcripts for potential errors...")
    
    for filename in os.listdir(TRANSCRIPT_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(TRANSCRIPT_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()
                
            for prefix in prefixes_to_check:
                # Regex to find the prefix and exactly one Chinese character after it
                matches = re.findall(rf'{prefix}([\u4e00-\u9fa5])', text)
                results[prefix].update(matches)

    print("\n--- AUDIT RESULTS ---")
    for prefix, counter in results.items():
        print(f"\nCharacters following '{prefix}':")
        # Print the top 5 most common characters that followed this prefix
        for char, count in counter.most_common(5):
            print(f"  {prefix}{char}: {count} times")
    print("\nIf you see things like '發心: 150 times', add it to your CORRECTIONS dict!")

def clean_text_files():
    if not os.path.exists(TRANSCRIPT_DIR):
        print(f"Error: Directory {TRANSCRIPT_DIR} not found.")
        return

    if not os.path.exists(CLEANED_DIR):
        os.makedirs(CLEANED_DIR)

    files_processed = 0
    replacements_made = 0

    for filename in os.listdir(TRANSCRIPT_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(TRANSCRIPT_DIR, filename)
            
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()
                
            original_text = text
            
            # Apply all corrections
            for bad_phrase, good_phrase in CORRECTIONS.items():
                text = text.replace(bad_phrase, good_phrase)
                
            

            cleaned_filepath = os.path.join(CLEANED_DIR, filename)
            with open(cleaned_filepath, "w", encoding="utf-8") as file:
                file.write(text)

            if text != original_text:
                replacements_made += 1
                print(f"Fixed errors in: {filename}")
            
            
            files_processed += 1

    print(f"\nDone! Processed {files_processed} files. Made updates to {replacements_made} files.")

if __name__ == "__main__":
    # First, run the audit to discover errors
    audit_taichi_terms()
    
    # Then, run the cleaner to apply your CORRECTIONS dictionary
    clean_text_files()