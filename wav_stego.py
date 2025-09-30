#!/usr/bin/env python3
"""
wav_stego.py

Простой LSB стегокодер/декодер для WAV 16-bit PCM.
Использование:
  Встраивание:
    python wav_stego.py embed input.wav output.wav "текст для встраивания"
  Извлечение:
    python wav_stego.py extract stego.wav
Примечание: работает только с WAV PCM 16-bit (sampwidth=2). Для стерео использует оба канала.
"""
import sys
import wave
import struct
import os

def capacity_info(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
    available_bits = n_frames * n_channels  # 1 бит на сэмпл
    return available_bits

def embed_text_to_wav(in_wav, out_wav, message):
    with wave.open(in_wav, 'rb') as wf:
        params = wf.getparams()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError("Поддерживается только WAV 16-bit (sampwidth=2).")

    # Подготовка данных: 4-байтная длина + UTF-8 байты
    msg_bytes = message.encode('utf-8')
    length = len(msg_bytes)
    header = length.to_bytes(4, byteorder='big')
    data = header + msg_bytes
    total_bits = len(data) * 8

    available_bits = n_frames * n_channels
    if total_bits > available_bits:
        raise ValueError(f"Недостаточно пространства в {in_wav}: нужно {total_bits} бит, доступно {available_bits} бит.")

    # Распаковка сэмплов (int16)
    samples = list(struct.unpack('<' + 'h' * (len(raw)//2), raw))

    # Функция для получения бита из data (старший бит в байте первый)
    def get_bit(i):
        byte_idx = i // 8
        bit_in_byte = 7 - (i % 8)
        return (data[byte_idx] >> bit_in_byte) & 1

    # Встраиваем биты в младший бит каждого сэмпла по порядку
    for i in range(total_bits):
        bit = get_bit(i)
        s = samples[i]
        if (s & 1) != bit:
            # корректно изменяем значение, не выходя за пределы int16
            if s == 32767:
                samples[i] = s - 1 if bit == 0 else s
            elif s == -32768:
                samples[i] = s + 1 if bit == 1 else s
            else:
                samples[i] = s | 1 if bit == 1 else s & ~1

    new_raw = struct.pack('<' + 'h' * len(samples), *samples)
    with wave.open(out_wav, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(new_raw)
    print(f"Встроено {len(msg_bytes)} байт ({len(data)*8} бит) в {out_wav}.")

def extract_text_from_wav(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError("Поддерживается только WAV 16-bit (sampwidth=2).")

    samples = list(struct.unpack('<' + 'h' * (len(raw)//2), raw))

    # Читаем первые 32 бита как длину (big-endian)
    bits = [samples[i] & 1 for i in range(32)]
    length = 0
    for b in bits:
        length = (length << 1) | b

    if length == 0:
        print("Длина сообщения = 0. Похоже, сообщения нет.")
        return ""

    total_bits = length * 8
    if 32 + total_bits > len(samples):
        raise ValueError("Файл похоже повреждён или длина некорректна.")

    data_bits = [samples[i] & 1 for i in range(32, 32 + total_bits)]

    # Собираем байты
    b_arr = bytearray()
    for i in range(0, len(data_bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | data_bits[i + j]
        b_arr.append(byte)

    try:
        text = b_arr.decode('utf-8')
    except Exception:
        text = b_arr.decode('utf-8', errors='replace')
    return text

def print_usage():
    print("Usage:")
    print("  Embed:   python wav_stego.py embed input.wav output.wav \"текст\"")
    print("  Extract: python wav_stego.py extract stego.wav")
    print("  Capacity check: python wav_stego.py capacity file.wav")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == 'embed':
        if len(sys.argv) < 5:
            print_usage()
            sys.exit(1)
        input_wav = sys.argv[2]
        output_wav = sys.argv[3]
        message = sys.argv[4]
        try:
            cap = capacity_info(input_wav)
            print(f"Доступно бит: {cap}")
            embed_text_to_wav(input_wav, output_wav, message)
        except Exception as e:
            print("Ошибка:", e)
            sys.exit(1)
    elif cmd == 'extract':
        wav = sys.argv[2]
        try:
            text = extract_text_from_wav(wav)
            print("Извлечённый текст:")
            print(text)
        except Exception as e:
            print("Ошибка:", e)
            sys.exit(1)
    elif cmd == 'capacity':
        wav = sys.argv[2]
        try:
            cap = capacity_info(wav)
            print(f"Доступно бит: {cap} (бит) — это ~{cap//8} байт.")
        except Exception as e:
            print("Ошибка:", e)
            sys.exit(1)
    else:
        print_usage()
        sys.exit(1)