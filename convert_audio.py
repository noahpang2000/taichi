import os
import subprocess

# Point this to the folder containing your media files
AUDIO_DIR = "./data/Liu JiShun Tai Ji Class" 

def convert_media_to_mp3():
    """
    Scans the directory for .wma and .mpg files and uses FFmpeg to 
    convert them to high-quality .mp3 files.
    """
    if not os.path.exists(AUDIO_DIR):
        print(f"Error: Directory '{AUDIO_DIR}' not found.")
        return

    print("Scanning for .wma and .mpg files...")
    converted_count = 0

    for filename in os.listdir(AUDIO_DIR):
        # We use .lower() just in case some files are uppercase like .WMA
        if filename.lower().endswith((".wma", ".mpg")):
            
            input_path = os.path.join(AUDIO_DIR, filename)
            
            # Strip the old extension and add .mp3
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(AUDIO_DIR, base_name + ".mp3")
            
            if os.path.exists(output_path):
                print(f"Skipping '{filename}': MP3 version already exists.")
                continue
                
            print(f"Converting '{filename}' to MP3...")
            
            # -i: specifies the input file
            # -q:a 2: specifies variable bitrate audio (high quality)
            # -vn: drops the video track if it's an MPG file to save space
            command = [
                "ffmpeg",
                "-i", input_path,
                "-vn", 
                "-q:a", "2", 
                output_path
            ]
            
            try:
                # subprocess.run executes the command line tool from within Python
                # We hide the standard output so it doesn't flood your terminal
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"  -> Success: Saved as '{base_name}.mp3'")
                converted_count += 1
            except subprocess.CalledProcessError as e:
                print(f"  -> Error converting '{filename}'. Is FFmpeg installed?")
            except FileNotFoundError:
                print("\nCRITICAL ERROR: 'ffmpeg' is not installed or not in your PATH.")
                print("Please install it using: brew install ffmpeg")
                return

    print(f"\nDone! Successfully converted {converted_count} files.")

if __name__ == "__main__":
    convert_media_to_mp3()