import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fft, fftfreq
import wave
import struct
import os


class AudioSteganographyDetector:
    """
    Hidden data detection and analysis system in audio files
    """

    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sample_rate = None
        self.audio_data = None
        self.duration = None

    def load_audio(self):
        """Load audio file"""
        try:
            self.sample_rate, self.audio_data = wavfile.read(self.audio_file)

            # Convert stereo to mono if needed
            if len(self.audio_data.shape) > 1:
                self.audio_data = self.audio_data[:, 0]

            self.duration = len(self.audio_data) / self.sample_rate
            print(f"✓ File loaded: {self.audio_file}")
            print(f"  Sample Rate: {self.sample_rate} Hz")
            print(f"  Duration: {self.duration:.2f} seconds")
            print(f"  Number of samples: {len(self.audio_data)}")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def lsb_analysis(self):
        """LSB (Least Significant Bit) analysis"""
        print("\n=== LSB Analysis ===")

        # Extract LSB bit
        lsb_data = self.audio_data & 1

        # LSB statistics
        zeros = np.sum(lsb_data == 0)
        ones = np.sum(lsb_data == 1)
        total = len(lsb_data)

        zero_ratio = zeros / total
        one_ratio = ones / total

        print(f"Number of 0s: {zeros} ({zero_ratio * 100:.2f}%)")
        print(f"Number of 1s: {ones} ({one_ratio * 100:.2f}%)")

        # Chi-square test
        expected = total / 2
        chi_square = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected

        print(f"Chi-square value: {chi_square:.4f}")

        # Suspicion threshold
        if abs(zero_ratio - 0.5) > 0.05:
            print("⚠ SUSPICIOUS: LSB distribution is abnormal!")
            suspicion_level = "HIGH"
        elif chi_square > 10:
            print("⚠ WARNING: Chi-square value is high")
            suspicion_level = "MEDIUM"
        else:
            print("✓ LSB distribution is normal")
            suspicion_level = "LOW"

        return {
            'zero_ratio': zero_ratio,
            'one_ratio': one_ratio,
            'chi_square': chi_square,
            'suspicion_level': suspicion_level
        }

    def spectral_analysis(self):
        """Spectral analysis"""
        print("\n=== Spectral Analysis ===")

        # Calculate FFT
        fft_data = fft(self.audio_data)
        freqs = fftfreq(len(self.audio_data), 1 / self.sample_rate)

        # Only positive frequencies
        positive_freqs = freqs[:len(freqs) // 2]
        magnitude = np.abs(fft_data[:len(fft_data) // 2])

        # Search for anomalies in high frequencies
        high_freq_threshold = self.sample_rate / 4
        high_freq_mask = positive_freqs > high_freq_threshold
        high_freq_energy = np.sum(magnitude[high_freq_mask] ** 2)
        total_energy = np.sum(magnitude ** 2)

        high_freq_ratio = high_freq_energy / total_energy

        print(f"High frequency energy: {high_freq_ratio * 100:.2f}%")

        if high_freq_ratio > 0.15:
            print("⚠ SUSPICIOUS: Anomaly in high frequencies!")
            suspicion = True
        else:
            print("✓ Spectrum is normal")
            suspicion = False

        return {
            'high_freq_ratio': high_freq_ratio,
            'suspicion': suspicion,
            'freqs': positive_freqs,
            'magnitude': magnitude
        }

    def statistical_analysis(self):
        """Statistical analysis"""
        print("\n=== Statistical Analysis ===")

        # Basic statistics
        mean = np.mean(self.audio_data)
        std = np.std(self.audio_data)
        variance = np.var(self.audio_data)

        print(f"Mean value: {mean:.2f}")
        print(f"Standard deviation: {std:.2f}")
        print(f"Variance: {variance:.2f}")

        # Calculate entropy
        hist, _ = np.histogram(self.audio_data, bins=256)
        hist = hist / len(self.audio_data)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))

        print(f"Entropy: {entropy:.4f} bits")

        # For normal audio, entropy is between 7-8
        if entropy > 7.5:
            print("⚠ WARNING: High entropy - hidden data may be present")
            suspicion = True
        else:
            print("✓ Entropy is at normal level")
            suspicion = False

        return {
            'mean': mean,
            'std': std,
            'entropy': entropy,
            'suspicion': suspicion
        }

    def generate_report(self, lsb_result, spectral_result, stat_result):
        """Generate full report"""
        print("\n" + "=" * 60)
        print("STEGANOGRAPHY DETECTION REPORT")
        print("=" * 60)
        print(f"File: {self.audio_file}")
        print(f"Date: {np.datetime64('today')}")
        print("\n--- RESULTS ---")
        print(f"1. LSB Analysis: Suspicion level - {lsb_result['suspicion_level']}")
        print(f"2. Spectral Analysis: {'SUSPICIOUS' if spectral_result['suspicion'] else 'CLEAN'}")
        print(f"3. Statistical Analysis: {'SUSPICIOUS' if stat_result['suspicion'] else 'CLEAN'}")

        # Overall conclusion
        suspicion_count = sum([
            lsb_result['suspicion_level'] == 'HIGH',
            spectral_result['suspicion'],
            stat_result['suspicion']
        ])

        print("\n--- CONCLUSION ---")
        if suspicion_count >= 2:
            print("🔴 HIGH PROBABILITY: File contains steganography!")
        elif suspicion_count == 1:
            print("🟡 MEDIUM PROBABILITY: Additional verification needed")
        else:
            print("🟢 LOW PROBABILITY: File appears to be clean")

        print("=" * 60)

    def visualize_results(self, spectral_result):
        """Visualize results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # 1. Waveform
        time = np.linspace(0, self.duration, len(self.audio_data))
        axes[0].plot(time[:1000], self.audio_data[:1000], color='blue', linewidth=0.5)
        axes[0].set_title('Audio Waveform (first 1000 samples)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)

        # 2. LSB distribution
        lsb_data = self.audio_data & 1
        axes[1].hist(lsb_data, bins=2, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_title('LSB Bit Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Bit value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_xticks([0, 1])
        axes[1].grid(True, alpha=0.3)

        # 3. Spectrogram
        freqs = spectral_result['freqs']
        magnitude = spectral_result['magnitude']
        axes[2].semilogy(freqs[:len(freqs) // 10], magnitude[:len(magnitude) // 10], color='red', linewidth=0.8)
        axes[2].set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('Magnitude (log scale)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('steganography_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: steganography_analysis.png")
        plt.show()


# LSB Steganography creation (for testing)
class LSBSteganography:
    """Hide message in audio file using LSB method"""

    @staticmethod
    def hide_message(audio_file, message, output_file):
        """Hide message in audio file"""
        print(f"\n=== Hiding Message ===")

        # Load audio
        sample_rate, audio_data = wavfile.read(audio_file)

        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        # Convert message to binary
        binary_message = ''.join(format(ord(c), '08b') for c in message)
        binary_message += '1111111111111110'  # Terminator

        if len(binary_message) > len(audio_data):
            print("✗ Error: Message is too long!")
            return False

        # Write to LSB
        stego_audio = audio_data.copy()
        for i, bit in enumerate(binary_message):
            stego_audio[i] = (stego_audio[i] & ~1) | int(bit)

        # Save
        wavfile.write(output_file, sample_rate, stego_audio.astype(audio_data.dtype))
        print(f"✓ Message hidden: {output_file}")
        print(f"  Message: '{message}'")
        print(f"  Length: {len(message)} characters")
        return True

    @staticmethod
    def extract_message(audio_file):
        """Extract hidden message"""
        print(f"\n=== Extracting Message ===")

        sample_rate, audio_data = wavfile.read(audio_file)

        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        # Extract LSB bit
        binary_message = ''
        for sample in audio_data:
            binary_message += str(sample & 1)

        # Search for terminator
        terminator = '1111111111111110'
        end_index = binary_message.find(terminator)

        if end_index == -1:
            print("✗ Message not found")
            return None

        binary_message = binary_message[:end_index]

        # Convert binary to text
        message = ''
        for i in range(0, len(binary_message), 8):
            byte = binary_message[i:i + 8]
            if len(byte) == 8:
                message += chr(int(byte, 2))

        print(f"✓ Message found: '{message}'")
        return message


# DEMONSTRATION
def demo():
    """Full demonstration"""
    print("=" * 60)
    print("AUDIO STEGANOGRAPHY DETECTION AND ANALYSIS")
    print("Digital Forensics Project")
    print("=" * 60)

    # 1. Create clean audio file (for testing)
    print("\n[1] Creating test audio file...")
    duration = 2
    sample_rate = 44100
    frequency = 440  # A4 note tone
    t = np.linspace(0, duration, int(sample_rate * duration))
    clean_audio = np.sin(2 * np.pi * frequency * t) * 32767
    clean_audio = clean_audio.astype(np.int16)

    clean_file = 'clean_audio.wav'
    wavfile.write(clean_file, sample_rate, clean_audio)
    print(f"✓ Clean file created: {clean_file}")

    # 2. Hide message
    print("\n[2] Hiding message...")
    stego_file = 'stego_audio.wav'
    message = "This is a hidden message!"
    LSBSteganography.hide_message(clean_file, message, stego_file)

    # 3. Analyze clean file
    print("\n" + "=" * 60)
    print("[3] ANALYZING CLEAN FILE")
    print("=" * 60)
    detector_clean = AudioSteganographyDetector(clean_file)
    detector_clean.load_audio()

    lsb_clean = detector_clean.lsb_analysis()
    spectral_clean = detector_clean.spectral_analysis()
    stat_clean = detector_clean.statistical_analysis()
    detector_clean.generate_report(lsb_clean, spectral_clean, stat_clean)

    # 4. Analyze stego file
    print("\n" + "=" * 60)
    print("[4] ANALYZING STEGO FILE")
    print("=" * 60)
    detector_stego = AudioSteganographyDetector(stego_file)
    detector_stego.load_audio()

    lsb_stego = detector_stego.lsb_analysis()
    spectral_stego = detector_stego.spectral_analysis()
    stat_stego = detector_stego.statistical_analysis()
    detector_stego.generate_report(lsb_stego, spectral_stego, stat_stego)

    # 5. Extract message
    print("\n[5] Extracting hidden message...")
    extracted = LSBSteganography.extract_message(stego_file)

    # 6. Visualization
    print("\n[6] Creating visualization...")
    detector_stego.visualize_results(spectral_stego)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    demo()