# Өз файлыңызды тексеру
from audio_steganography import AudioSteganographyDetector

detector = AudioSteganographyDetector('output.wav')
detector.load_audio()

lsb = detector.lsb_analysis()
spectral = detector.spectral_analysis()
stat = detector.statistical_analysis()

detector.generate_report(lsb, spectral, stat)
detector.visualize_results(spectral)