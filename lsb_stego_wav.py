# lsb_stego_wav.py
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
