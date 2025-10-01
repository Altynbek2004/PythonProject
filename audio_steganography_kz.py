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
    Audio файлдарындағы жасырын деректерді анықтау және талдау жүйесі
    """

    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sample_rate = None
        self.audio_data = None
        self.duration = None

    def load_audio(self):
        """Audio файлды жүктеу"""
        try:
            self.sample_rate, self.audio_data = wavfile.read(self.audio_file)

            # Stereo болса, mono-ға айналдыру
            if len(self.audio_data.shape) > 1:
                self.audio_data = self.audio_data[:, 0]

            self.duration = len(self.audio_data) / self.sample_rate
            print(f"✓ Файл жүктелді: {self.audio_file}")
            print(f"  Sample Rate: {self.sample_rate} Hz")
            print(f"  Ұзақтығы: {self.duration:.2f} секунд")
            print(f"  Үлгілер саны: {len(self.audio_data)}")
            return True
        except Exception as e:
            print(f"✗ Қате: {e}")
            return False

    def lsb_analysis(self):
        """LSB (Least Significant Bit) талдауы"""
        print("\n=== LSB Талдауы ===")

        # LSB битін алу
        lsb_data = self.audio_data & 1

        # LSB статистикасы
        zeros = np.sum(lsb_data == 0)
        ones = np.sum(lsb_data == 1)
        total = len(lsb_data)

        zero_ratio = zeros / total
        one_ratio = ones / total

        print(f"0-лер саны: {zeros} ({zero_ratio * 100:.2f}%)")
        print(f"1-лер саны: {ones} ({one_ratio * 100:.2f}%)")

        # Chi-square тесті
        expected = total / 2
        chi_square = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected

        print(f"Chi-square мәні: {chi_square:.4f}")

        # Күдікті деп саналу шегі
        if abs(zero_ratio - 0.5) > 0.05:
            print("⚠ КҮДІКТІ: LSB таралуы қалыпты емес!")
            suspicion_level = "ЖОҒАРЫ"
        elif chi_square > 10:
            print("⚠ ЕСКЕРТУ: Chi-square мәні жоғары")
            suspicion_level = "ОРТАША"
        else:
            print("✓ LSB таралуы қалыпты")
            suspicion_level = "ТӨМЕН"

        return {
            'zero_ratio': zero_ratio,
            'one_ratio': one_ratio,
            'chi_square': chi_square,
            'suspicion_level': suspicion_level
        }

    def spectral_analysis(self):
        """Спектралды талдау"""
        print("\n=== Спектралды Талдау ===")

        # FFT есептеу
        fft_data = fft(self.audio_data)
        freqs = fftfreq(len(self.audio_data), 1 / self.sample_rate)

        # Тек оң жиіліктер
        positive_freqs = freqs[:len(freqs) // 2]
        magnitude = np.abs(fft_data[:len(fft_data) // 2])

        # Жоғары жиіліктердегі аномалияларды іздеу
        high_freq_threshold = self.sample_rate / 4
        high_freq_mask = positive_freqs > high_freq_threshold
        high_freq_energy = np.sum(magnitude[high_freq_mask] ** 2)
        total_energy = np.sum(magnitude ** 2)

        high_freq_ratio = high_freq_energy / total_energy

        print(f"Жоғары жиілік энергиясы: {high_freq_ratio * 100:.2f}%")

        if high_freq_ratio > 0.15:
            print("⚠ КҮДІКТІ: Жоғары жиіліктерде аномалия бар!")
            suspicion = True
        else:
            print("✓ Спектр қалыпты")
            suspicion = False

        return {
            'high_freq_ratio': high_freq_ratio,
            'suspicion': suspicion,
            'freqs': positive_freqs,
            'magnitude': magnitude
        }

    def statistical_analysis(self):
        """Статистикалық талдау"""
        print("\n=== Статистикалық Талдау ===")

        # Негізгі статистика
        mean = np.mean(self.audio_data)
        std = np.std(self.audio_data)
        variance = np.var(self.audio_data)

        print(f"Орташа мән: {mean:.2f}")
        print(f"Стандартты ауытқу: {std:.2f}")
        print(f"Дисперсия: {variance:.2f}")

        # Энтропия есептеу
        hist, _ = np.histogram(self.audio_data, bins=256)
        hist = hist / len(self.audio_data)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))

        print(f"Энтропия: {entropy:.4f} бит")

        # Қалыпты audio үшін энтропия 7-8 аралығында
        if entropy > 7.5:
            print("⚠ ЕСКЕРТУ: Жоғары энтропия - жасырын деректер болуы мүмкін")
            suspicion = True
        else:
            print("✓ Энтропия қалыпты деңгейде")
            suspicion = False

        return {
            'mean': mean,
            'std': std,
            'entropy': entropy,
            'suspicion': suspicion
        }

    def generate_report(self, lsb_result, spectral_result, stat_result):
        """Толық есепті жасау"""
        print("\n" + "=" * 60)
        print("СТЕГАНОГРАФИЯ АНЫҚТАУ ЕСЕБІ")
        print("=" * 60)
        print(f"Файл: {self.audio_file}")
        print(f"Дата: {np.datetime64('today')}")
        print("\n--- НӘТИЖЕЛЕР ---")
        print(f"1. LSB Талдауы: Күдік деңгейі - {lsb_result['suspicion_level']}")
        print(f"2. Спектралды Талдау: {'КҮДІКТІ' if spectral_result['suspicion'] else 'ТАЗА'}")
        print(f"3. Статистикалық Талдау: {'КҮДІКТІ' if stat_result['suspicion'] else 'ТАЗА'}")

        # Жалпы қорытынды
        suspicion_count = sum([
            lsb_result['suspicion_level'] == 'ЖОҒАРЫ',
            spectral_result['suspicion'],
            stat_result['suspicion']
        ])

        print("\n--- ҚОРЫТЫНДЫ ---")
        if suspicion_count >= 2:
            print("🔴 ЖОҒАРЫ ЫҚТИМАЛДЫҚ: Файлда стеганография бар!")
        elif suspicion_count == 1:
            print("🟡 ОРТАША ЫҚТИМАЛДЫҚ: Қосымша тексеру қажет")
        else:
            print("🟢 ТӨМЕН ЫҚТИМАЛДЫҚ: Файл таза болып көрінеді")

        print("=" * 60)

    def visualize_results(self, spectral_result):
        """Нәтижелерді визуализациялау"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # 1. Толқын формасы
        time = np.linspace(0, self.duration, len(self.audio_data))
        axes[0].plot(time[:1000], self.audio_data[:1000], color='blue', linewidth=0.5)
        axes[0].set_title('Audio Толқын Формасы (алғашқы 1000 үлгі)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Уақыт (с)')
        axes[0].set_ylabel('Амплитуда')
        axes[0].grid(True, alpha=0.3)

        # 2. LSB таралуы
        lsb_data = self.audio_data & 1
        axes[1].hist(lsb_data, bins=2, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_title('LSB Битінің Таралуы', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Бит мәні')
        axes[1].set_ylabel('Жиілік')
        axes[1].set_xticks([0, 1])
        axes[1].grid(True, alpha=0.3)

        # 3. Спектрограмма
        freqs = spectral_result['freqs']
        magnitude = spectral_result['magnitude']
        axes[2].semilogy(freqs[:len(freqs) // 10], magnitude[:len(magnitude) // 10], color='red', linewidth=0.8)
        axes[2].set_title('Жиілік Спектрі', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Жиілік (Hz)')
        axes[2].set_ylabel('Магнитуда (log scale)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('steganography_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Визуализация сақталды: steganography_analysis.png")
        plt.show()


# LSB Steganography жасау (тестілеу үшін)
class LSBSteganography:
    """LSB әдісімен audio файлға хабарлама жасыру"""

    @staticmethod
    def hide_message(audio_file, message, output_file):
        """Хабарламаны audio файлға жасыру"""
        print(f"\n=== Хабарламаны жасыру ===")

        # Audio жүктеу
        sample_rate, audio_data = wavfile.read(audio_file)

        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        # Хабарламаны binary-ға айналдыру
        binary_message = ''.join(format(ord(c), '08b') for c in message)
        binary_message += '1111111111111110'  # Аяқтау белгісі

        if len(binary_message) > len(audio_data):
            print("✗ Қате: Хабарлама тым ұзын!")
            return False

        # LSB-ға жазу
        stego_audio = audio_data.copy()
        for i, bit in enumerate(binary_message):
            stego_audio[i] = (stego_audio[i] & ~1) | int(bit)

        # Сақтау
        wavfile.write(output_file, sample_rate, stego_audio.astype(audio_data.dtype))
        print(f"✓ Хабарлама жасырылды: {output_file}")
        print(f"  Хабарлама: '{message}'")
        print(f"  Ұзындығы: {len(message)} символ")
        return True

    @staticmethod
    def extract_message(audio_file):
        """Жасырылған хабарламаны алу"""
        print(f"\n=== Хабарламаны алу ===")

        sample_rate, audio_data = wavfile.read(audio_file)

        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        # LSB битін алу
        binary_message = ''
        for sample in audio_data:
            binary_message += str(sample & 1)

        # Аяқтау белгісін іздеу
        terminator = '1111111111111110'
        end_index = binary_message.find(terminator)

        if end_index == -1:
            print("✗ Хабарлама табылмады")
            return None

        binary_message = binary_message[:end_index]

        # Binary-дан text-ке айналдыру
        message = ''
        for i in range(0, len(binary_message), 8):
            byte = binary_message[i:i + 8]
            if len(byte) == 8:
                message += chr(int(byte, 2))

        print(f"✓ Хабарлама табылды: '{message}'")
        return message


# ДЕМОНСТРАЦИЯ
def demo():
    """Толық демонстрация"""
    print("=" * 60)
    print("AUDIO STEGANOGRAPHY DETECTION AND ANALYSIS")
    print("Digital Forensics Проекті")
    print("=" * 60)

    # 1. Таза audio файл жасау (тестілеу үшін)
    print("\n[1] Тестілік audio файл жасау...")
    duration = 2
    sample_rate = 44100
    frequency = 440  # A4 ноталық тон
    t = np.linspace(0, duration, int(sample_rate * duration))
    clean_audio = np.sin(2 * np.pi * frequency * t) * 32767
    clean_audio = clean_audio.astype(np.int16)

    clean_file = 'clean_audio.wav'
    wavfile.write(clean_file, sample_rate, clean_audio)
    print(f"✓ Таза файл жасалды: {clean_file}")

    # 2. Хабарламаны жасыру
    print("\n[2] Хабарламаны жасыру...")
    stego_file = 'stego_audio.wav'
    message = "Бұл жасырын хабарлама!"
    LSBSteganography.hide_message(clean_file, message, stego_file)

    # 3. Таза файлды талдау
    print("\n" + "=" * 60)
    print("[3] ТАЗА ФАЙЛДЫ ТАЛДАУ")
    print("=" * 60)
    detector_clean = AudioSteganographyDetector(clean_file)
    detector_clean.load_audio()

    lsb_clean = detector_clean.lsb_analysis()
    spectral_clean = detector_clean.spectral_analysis()
    stat_clean = detector_clean.statistical_analysis()
    detector_clean.generate_report(lsb_clean, spectral_clean, stat_clean)

    # 4. Стего файлды талдау
    print("\n" + "=" * 60)
    print("[4] СТЕГО ФАЙЛДЫ ТАЛДАУ")
    print("=" * 60)
    detector_stego = AudioSteganographyDetector(stego_file)
    detector_stego.load_audio()

    lsb_stego = detector_stego.lsb_analysis()
    spectral_stego = detector_stego.spectral_analysis()
    stat_stego = detector_stego.statistical_analysis()
    detector_stego.generate_report(lsb_stego, spectral_stego, stat_stego)

    # 5. Хабарламаны алу
    print("\n[5] Жасырылған хабарламаны алу...")
    extracted = LSBSteganography.extract_message(stego_file)

    # 6. Визуализация
    print("\n[6] Визуализация жасау...")
    detector_stego.visualize_results(spectral_stego)

    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ АЯҚТАЛДЫ")
    print("=" * 60)


if __name__ == "__main__":
    demo()