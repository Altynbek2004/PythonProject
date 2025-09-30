from scipy.io import wavfile
import numpy as np

from audio_steganography import LSBSteganography, AudioSteganographyDetector

# 1-әдіс: Қарапайым синус толқыны жасау
sample_rate = 44100
duration = 3  # секунд
t = np.linspace(0, duration, int(sample_rate * duration))
audio = np.sin(2 * np.pi * 440 * t) * 32767
audio = audio.astype(np.int16)
wavfile.write('my_audio.wav', sample_rate, audio)

# 2-әдіс: Хабарлама жасыру
LSBSteganography.hide_message('my_audio.wav', 'Hacking Nurik', 'stego.wav')

# 3-әдіс: Талдау
detector = AudioSteganographyDetector('stego.wav')
detector.load_audio()
# ... талдау жалғасады