import streamlit as st
import whisper
import ffmpeg
from better_profanity import profanity
import os
import tempfile

st.title("🧼 MP4 Profanity Muter")

uploaded_file = st.file_uploader("Upload MP4", type=["mp4"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        input_path = temp_video.name
        audio_path = input_path.replace(".mp4", ".wav")
        output_path = input_path.replace(".mp4", "_clean.mp4")

    st.write("🔊 Extracting audio...")
    ffmpeg.input(input_path).output(audio_path, ac=1, ar='16000').run(overwrite_output=True)

    st.write("🧠 Transcribing...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)

    profanity.load_censor_words()
    mute_ranges = []
    for seg in result["segments"]:
        for word in seg.get("words", []):
            if profanity.contains_profanity(word['word'].lower()):
                mute_ranges.append((word['start'], word['end']))

    if mute_ranges:
        st.write(f"🚫 Found {len(mute_ranges)} bad words... muting.")
        conditions = "+".join([f"between(t,{s},{e})" for s, e in mute_ranges])
        volume_filter = f"volume=enable='{conditions}':volume=0"

        ffmpeg.input(input_path).output(
            output_path,
            af=volume_filter,
            vcodec='copy',
            acodec='aac'
        ).run(overwrite_output=True)

        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Clean Video", f, file_name="sanitized.mp4")

    else:
        st.success("✅ No profanity found!")
