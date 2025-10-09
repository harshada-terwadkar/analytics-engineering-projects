import os
import csv
import numpy as np
import subprocess
import random
from PIL import Image, ImageDraw, ImageFont
from moviepy import ColorClip, CompositeVideoClip, ImageClip, AudioFileClip
import textwrap
from datetime import datetime

def get_quote_for_date(csv_path, target_date_str):
    """
    Extracts the quote and author for a specific date from the CSV file.
    The CSV file is expected to have three columns: 'date', 'quote', and 'author'.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'")
        return None, None
    try:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Assuming date format is MM/DD/YY
                if row.get('date') == target_date_str:
                    return row.get('quote'), row.get('author')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return None, None

def find_bold_font():
    bold_font_names = [
        "Arial-Bold.ttf",
        "Arial Bold.ttf",
        "Helvetica-Bold",
        "DejaVuSans-Bold.ttf",
        "Roboto-Bold.ttf",
        "Arial_Bold.ttf",
    ]
    for font_name in bold_font_names:
        try:
            return ImageFont.truetype(font_name, 1)
        except IOError:
            continue
    return None

def create_text_image(text, font_size, max_width):
    bold_font = find_bold_font()
    font = ImageFont.truetype(bold_font.path, font_size) if bold_font else ImageFont.load_default()
    dummy_img = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    lines = []
    for line in text.split('\n'):
        words = line.split()
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            w = dummy_draw.textlength(test_line, font=font)  # Only one value returned
            if w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
    wrapped_text = "\n".join(lines)
    left, top, right, bottom = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    text_width = right - left
    text_height = bottom - top
    img = Image.new('RGBA', (text_width + 40, text_height + 40), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.multiline_text((20, 20), wrapped_text, font=font, fill=(255, 255, 255), align='center')
    return img

def create_video_with_quote(quote, author, background_image_path, audio_file_path,
                            output_filename="daily_quote_video.mp4"):
    video_width = 1080
    video_height = 1920
    video_duration = 5
    background = (ImageClip(background_image_path)
                  .resized((video_width, video_height))
                  .with_duration(video_duration))
    max_text_width = int(video_width * 0.8)
    quote_img = create_text_image(quote, 60, max_text_width)
    quote_clip = ImageClip(np.array(quote_img)).with_start(0).with_duration(video_duration)
    author_img = create_text_image(f"- {author}", 50, max_text_width)
    author_clip = ImageClip(np.array(author_img)).with_start(0).with_duration(video_duration)

    quote_y_pos = video_height * 0.25
    quote_height = quote_img.height
    margin = 40
    author_y_pos = quote_y_pos + quote_height + margin

    quote_clip = quote_clip.with_position(('center', quote_y_pos))
    author_clip = author_clip.with_position(('center', author_y_pos))  # Center to avoid right cutoff

    temp_audio_path = "temp_audio_slice.mp3"
    try:
        print("Slicing audio with ffmpeg...")
        command = [
            'ffmpeg',
            '-i', audio_file_path,
            '-ss', '00:00:10',
            '-t', '00:00:05',
            '-c:a', 'copy',
            temp_audio_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        audio_clip = AudioFileClip(temp_audio_path)
        print("Audio sliced successfully.")
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure it is installed and in your system's PATH.")
        audio_clip = None
    except subprocess.CalledProcessError as e:
        print(f"Error slicing audio with ffmpeg: {e}")
        audio_clip = None
    except Exception as e:
        print(f"Error loading or slicing audio file: {e}")
        audio_clip = None
    final_video = CompositeVideoClip([background, quote_clip, author_clip])
    if audio_clip:
        final_video = final_video.with_audio(audio_clip)
    try:
        final_video.write_videofile(output_filename, fps=24, codec="libx264", audio_codec="aac")
        print(f"Successfully created video: {output_filename}")
    except Exception as e:
        print(f"Error creating video: {e}")
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

if __name__ == "__main__":
    csv_file_path = os.path.join(os.path.dirname(__file__), 'quotes.csv')
    background_path = os.path.join(os.path.dirname(__file__), 'sunrise_image.png')
    audio_path = os.path.join(os.path.dirname(__file__), 'energetic_music.mp3')

    today = datetime.now()
    today_str = today.strftime('%-m/%-d/%y')  # For CSV lookup

    quote_of_the_day, author_of_the_day = get_quote_for_date(csv_file_path, today_str)

    if not quote_of_the_day:
        print(f"No quote found in the CSV for the date: {today_str}.")
        print("Please ensure your 'quotes.csv' file has a 'date' column with today's date.")
    else:
        print(f"Generating video for the quote of the day:")
        print(f"Date: {today.strftime('%-m/%-d/%y')}")
        print(f"Quote: \"{quote_of_the_day}\"")
        print(f"Author: {author_of_the_day}")
        create_video_with_quote(quote_of_the_day, author_of_the_day, background_path, audio_path)