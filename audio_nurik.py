# Создам и продемонстрирую простой LSB-стегокодер для WAV (16-bit PCM).
# Функции: embed_text_to_wav(in_wav, out_wav, message) и extract_text_from_wav(wav_path).
# Сгенерирую тестовый WAV (синус), запишу в него сообщение на русском, извлеку и покажу результат.
import wave, struct, math, os

def create_sine_wav(path, duration_s=1.0, freq=440.0, samplerate=44100, amplitude=16000):
    n_frames = int(duration_s * samplerate)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)  # моно
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        frames = bytearray()
        for i in range(n_frames):
            t = i / samplerate
            sample = int(amplitude * math.sin(2 * math.pi * freq * t))
            frames.extend(struct.pack('<h', sample))
        wf.writeframes(frames)

def embed_text_to_wav(in_wav, out_wav, message):
    # Открываем WAV и читаем параметры
    with wave.open(in_wav, 'rb') as wf:
        params = wf.getparams()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError("Этот скрипт поддерживает только 16-bit PCM WAV (sampwidth=2).")
    # Подготовка данных для встраивания: сначала 32-битная длина (в байтах), потом байты сообщения (utf-8)
    msg_bytes = message.encode('utf-8')
    length = len(msg_bytes)
    header = length.to_bytes(4, byteorder='big')  # 4 байта длины
    data = header + msg_bytes
    total_bits = len(data) * 8
    available_bits = n_frames * n_channels  # по 1 биту на сэмпл
    if total_bits > available_bits:
        raise ValueError(f"Недостаточно места: нужно {total_bits} бит, доступно {available_bits} бит.")
    # Преобразуем сэмплы в список int16
    samples = list(struct.unpack('<' + 'h' * (len(raw)//2), raw))
    # Функция для получения следующего бита из data
    bit_idx = 0
    def get_bit(i):
        byte_idx = i // 8
        bit_in_byte = 7 - (i % 8)  # старший бит первым
        return (data[byte_idx] >> bit_in_byte) & 1
    # Встраиваем биты в младший бит каждого сэмпла
    for i in range(total_bits):
        bit = get_bit(i)
        s = samples[i]
        # заменяем младший бит
        if (s & 1) != bit:
            if s & 1 == 1:
                samples[i] = s - 1
            else:
                samples[i] = s + 1
    # Собираем обратно байты
    new_raw = struct.pack('<' + 'h' * len(samples), *samples)
    # Записываем в новый WAV
    with wave.open(out_wav, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(new_raw)

def extract_text_from_wav(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError("Этот скрипт поддерживает только 16-bit PCM WAV (sampwidth=2).")
    samples = list(struct.unpack('<' + 'h' * (len(raw)//2), raw))
    # Сначала читаем 32 бита заголовка длины
    bits = []
    for i in range(32):
        bits.append(samples[i] & 1)
    # Преобразуем в длину
    length = 0
    for b in bits:
        length = (length << 1) | b
    # Теперь читаем length байт = length*8 бит
    total_bits = length * 8
    if total_bits == 0:
        return ""
    data_bits = []
    for i in range(32, 32 + total_bits):
        data_bits.append(samples[i] & 1)
    # Собираем байты
    b_arr = bytearray()
    for i in range(0, len(data_bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | data_bits[i + j]
        b_arr.append(byte)
    return b_arr.decode('utf-8', errors='replace')

# Создаём тестовый WAV
orig = os.path.join(os.getcwd(), 'test_sine.wav')
stego = os.path.join(os.getcwd(), 'test_sine_stego.wav')

create_sine_wav(orig, duration_s=1.0, freq=440.0)
message = "Привет, это скрытое сообщение!"
embed_text_to_wav(orig, stego, message)
extracted = extract_text_from_wav(stego)

print("Исходный файл:", orig)
print("Файл со встроенным текстом:", stego)
print("Встраиваемое сообщение:", message)
print("Извлечённое сообщение:", extracted)

# Выведем размер файлов и доступные первые 64 байта (для проверки)
print("\nРазмеры файлов:")
print(os.path.getsize(orig), "байт (оригинал)")
print(os.path.getsize(stego), "байт (стего)")

# Предоставим ссылки для загрузки (если UI поддерживает)
print("\nСсылки для скачивания (если доступны):")
print(f"[Download original](/mnt/data/{os.path.basename(orig)})")
print(f"[Download stego](/mnt/data/{os.path.basename(stego)})")

# Сохраняем используемый скрипт в файл для пользователя
script_path = 'lsb_stego_wav.py'
script_code = r'''# lsb_stego_wav.py
# Простой LSB-стегокодер/декодер WAV (16-bit PCM)
import wave, struct

def embed_text_to_wav(in_wav, out_wav, message):
    with wave.open(in_wav, 'rb') as wf:
        params = wf.getparams()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError("Поддерживается только 16-bit WAV")
    msg_bytes = message.encode('utf-8')
    length = len(msg_bytes)
    header = length.to_bytes(4, byteorder='big')
    data = header + msg_bytes
    total_bits = len(data) * 8
    available_bits = n_frames * n_channels
    if total_bits > available_bits:
        raise ValueError("Недостаточно места в WAV")
    samples = list(struct.unpack('<' + 'h' * (len(raw)//2), raw))
    def get_bit(i):
        byte_idx = i // 8
        bit_in_byte = 7 - (i % 8)
        return (data[byte_idx] >> bit_in_byte) & 1
    for i in range(total_bits):
        bit = get_bit(i)
        s = samples[i]
        if (s & 1) != bit:
            if s & 1 == 1:
                samples[i] = s - 1
            else:
                samples[i] = s + 1
    new_raw = struct.pack('<' + 'h' * len(samples), *samples)
    with wave.open(out_wav, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(new_raw)

def extract_text_from_wav(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError("Поддерживается только 16-bit WAV")
    samples = list(struct.unpack('<' + 'h' * (len(raw)//2), raw))
    bits = []
    for i in range(32):
        bits.append(samples[i] & 1)
    length = 0
    for b in bits:
        length = (length << 1) | b
    total_bits = length * 8
    data_bits = []
    for i in range(32, 32 + total_bits):
        data_bits.append(samples[i] & 1)
    b_arr = bytearray()
    for i in range(0, len(data_bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | data_bits[i + j]
        b_arr.append(byte)
    return b_arr.decode('utf-8', errors='replace')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python lsb_stego_wav.py embed|extract ...")
'''
with open(script_path, 'w', encoding='utf-8') as f:
    f.write(script_code)

print("\nСкрипт сохранён как:", script_path)