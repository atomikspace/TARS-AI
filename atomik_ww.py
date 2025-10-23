#!/usr/bin/env python3
"""
Wake Word Detection System

Author: Olivier Dion (@atomikspace)
Email: olivierdion1@hotmail.com
License: All Rights Reserved

Features:
- Custom wake word support
- Automatic speech detection (VAD)
- MFCC-based voice fingerprinting
- Real-time audio processing
- Data augmentation for improved accuracy
"""

import numpy as np
import sounddevice as sd
from collections import deque
import pickle
import os
from scipy.fftpack import dct
from scipy import signal
import time

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, energy_threshold=0.008, silence_duration=0.5):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_frames = int(silence_duration * sample_rate / 1024)
        
    def get_energy(self, audio_chunk):
        return np.sqrt(np.mean(audio_chunk ** 2))
    
    def is_speech(self, audio_chunk):
        energy = self.get_energy(audio_chunk)
        return energy > self.energy_threshold
    
    def trim_silence(self, audio_array, chunk_size=1024):
        chunks = [audio_array[i:i+chunk_size] for i in range(0, len(audio_array), chunk_size)]
        
        start_idx = 0
        for i, chunk in enumerate(chunks):
            if self.is_speech(chunk):
                start_idx = max(0, i - 1)
                break
        
        end_idx = len(chunks)
        for i in range(len(chunks) - 1, -1, -1):
            if self.is_speech(chunks[i]):
                end_idx = min(len(chunks), i + 2)
                break
        
        start_sample = start_idx * chunk_size
        end_sample = min(end_idx * chunk_size, len(audio_array))
        
        return audio_array[start_sample:end_sample]


class MFCCExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.n_mels = 40
        self.mel_filters = self.create_mel_filterbank()
    
    def hz_to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700.0)
    
    def mel_to_hz(self, mel):
        return 700 * (10**(mel / 2595.0) - 1)
    
    def create_mel_filterbank(self):
        low_freq_mel = 0
        high_freq_mel = self.hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        fbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for m in range(1, self.n_mels + 1):
            f_left = bin_points[m - 1]
            f_center = bin_points[m]
            f_right = bin_points[m + 1]
            
            for k in range(f_left, f_center):
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
        
        return fbank
    
    def extract_mfcc(self, audio):
        if len(audio) < self.n_fft:
            return None
        
        emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        frame_length = self.n_fft
        frame_step = frame_length // 2
        num_frames = 1 + int(np.floor((len(emphasized) - frame_length) / frame_step))
        
        frames = np.zeros((num_frames, frame_length))
        for i in range(num_frames):
            start = i * frame_step
            frames[i] = emphasized[start:start + frame_length]
        
        frames *= np.hamming(frame_length)
        
        mag_frames = np.absolute(np.fft.rfft(frames, self.n_fft))
        pow_frames = ((1.0 / self.n_fft) * (mag_frames ** 2))
        
        filter_banks = np.dot(pow_frames, self.mel_filters.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)
        
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :self.n_mfcc]
        
        mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
        
        return mfcc


class WakeWordDetector:
    def __init__(self, wake_word="hey tars", sample_rate=16000, threshold=0.6, augment_data=True):
        self.wake_word = wake_word
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.augment_data = augment_data
        self.mfcc_extractor = MFCCExtractor(sample_rate=sample_rate)
        self.vad = VoiceActivityDetector(sample_rate=sample_rate, energy_threshold=0.008)
        self.buffer = deque(maxlen=sample_rate * 3)
        self.templates = []
        self.last_detection_time = 0
        self.cooldown = 1.5  # Reduced from 2.0
        self.last_check_time = 0
        self.check_interval = 0.1  # Check every 100ms
    
    def time_stretch(self, audio, rate):
        """Stretch audio in time without changing pitch"""
        indices = np.round(np.arange(0, len(audio), rate))
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices]
    
    def pitch_shift(self, audio, semitones):
        """Shift pitch of audio"""
        factor = 2 ** (semitones / 12.0)
        indices = np.round(np.arange(0, len(audio), factor))
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices]
    
    def add_noise(self, audio, noise_level=0.005):
        """Add slight background noise"""
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    def augment_audio(self, audio):
        """Create augmented versions of the audio"""
        augmented = []
        
        # Original
        augmented.append(audio)
        
        # Speed variations (90% and 110% speed)
        augmented.append(self.time_stretch(audio, 0.9))
        augmented.append(self.time_stretch(audio, 1.1))
        
        # Pitch variations (lower and higher)
        augmented.append(self.pitch_shift(audio, -2))  # Lower pitch (male-ish)
        augmented.append(self.pitch_shift(audio, 2))   # Higher pitch (female-ish)
        
        # With slight noise
        augmented.append(self.add_noise(audio, 0.003))
        
        return augmented
    
    def record_template(self):
        print(f"\nGet ready to record...")
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print(f"\n   Listening... SAY '{self.wake_word.upper()}' now!")
        
        recording = []
        speech_started = False
        silence_count = 0
        max_silence_frames = 15
        
        def callback(indata, frames, time_info, status):
            nonlocal speech_started, silence_count
            
            audio_chunk = indata[:, 0]
            
            if not speech_started:
                if self.vad.is_speech(audio_chunk):
                    speech_started = True
                    print("   Recording...", end="", flush=True)
                    recording.extend(audio_chunk)
            else:
                recording.extend(audio_chunk)
                if self.vad.is_speech(audio_chunk):
                    silence_count = 0
                    print("█", end="", flush=True)
                else:
                    silence_count += 1
                    print(".", end="", flush=True)
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                           callback=callback, blocksize=512):
            while not speech_started or silence_count < max_silence_frames:
                time.sleep(0.01)
        
        print(" [DONE]")
        
        audio_array = np.array(recording, dtype=np.float32)
        audio_array = self.vad.trim_silence(audio_array)
        
        duration = len(audio_array) / self.sample_rate
        energy = np.sqrt(np.mean(audio_array ** 2))
        
        print(f"   Duration: {duration:.1f}s, Audio level: {energy:.4f}")
        
        if duration < 0.3:
            print("   WARNING: Too short! Try again and speak the full phrase.")
            return False
        
        if duration > 3.0:
            print("   WARNING: Too long! Keep it under 3 seconds.")
            return False
        
        if energy < 0.005:
            print("   WARNING: Audio too quiet! Increase mic volume or speak louder.")
            return False
        
        mfcc = self.mfcc_extractor.extract_mfcc(audio_array)
        
        if mfcc is not None:
            self.templates.append(mfcc)
            templates_added = 1
            
            # Create augmented versions if enabled
            if self.augment_data:
                print("   Generating augmented versions...")
                augmented_audios = self.augment_audio(audio_array)
                
                for aug_audio in augmented_audios[1:]:  # Skip original (already added)
                    aug_mfcc = self.mfcc_extractor.extract_mfcc(aug_audio)
                    if aug_mfcc is not None:
                        self.templates.append(aug_mfcc)
                        templates_added += 1
                
                print(f"   Created {templates_added} templates (1 original + {templates_added-1} augmented)")
            else:
                print(f"   Template {len(self.templates)} recorded successfully!")
            
            return True
        else:
            print("   Failed to extract features")
            return False
    
    def add_audio(self, audio_chunk):
        self.buffer.extend(audio_chunk)
    
    def cosine_similarity(self, mfcc1, mfcc2):
        v1 = mfcc1.flatten()
        v2 = mfcc2.flatten()
        
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def detect(self):
        if len(self.templates) == 0:
            return False, 0.0
        
        # Reduced window: only need 1 second for "hey tars"
        min_window = int(self.sample_rate * 1.0)
        if len(self.buffer) < min_window:
            return False, 0.0
        
        # Cooldown after detection
        if time.time() - self.last_detection_time < self.cooldown:
            return False, 0.0
        
        # Only check every 100ms to reduce CPU
        current_time = time.time()
        if current_time - self.last_check_time < self.check_interval:
            return False, 0.0
        self.last_check_time = current_time
        
        # Use 1 second window (faster than 1.5s)
        window_size = min_window
        audio_window = np.array(list(self.buffer)[-window_size:], dtype=np.float32)
        
        # Quick VAD check first
        if not self.vad.is_speech(audio_window[:1024]):
            return False, 0.0
        
        # Extract MFCC
        current_mfcc = self.mfcc_extractor.extract_mfcc(audio_window)
        if current_mfcc is None:
            return False, 0.0
        
        # Check against templates - early exit on high confidence
        max_similarity = 0.0
        for template in self.templates:
            similarity = self.cosine_similarity(current_mfcc, template)
            max_similarity = max(max_similarity, similarity)
            
            # Early detection for very high confidence
            if similarity > 0.8:
                self.last_detection_time = time.time()
                return True, similarity
        
        # Normal threshold check
        if max_similarity >= self.threshold:
            self.last_detection_time = time.time()
            return True, max_similarity
        
        return False, max_similarity
    
    def save_templates(self, filename=None):
        if filename is None:
            filename = f"{self.wake_word.replace(' ', '_')}_templates.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.templates, f)
        print(f"Templates saved to {filename}")
    
    def load_templates(self, filename=None):
        if filename is None:
            filename = f"{self.wake_word.replace(' ', '_')}_templates.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.templates = pickle.load(f)
            print(f"Loaded {len(self.templates)} templates from {filename}")
            return True
        return False


def main():
    # CONFIGURATION
    WAKE_WORD = "hey tars"
    SAMPLE_RATE = 16000
    THRESHOLD = 0.6
    AUGMENT_DATA = True  # Enable data augmentation for better accuracy
    
    # Performance tuning:
    # - 1.0s detection window (fast for short phrases)
    # - Checks every 100ms (reduces CPU, still responsive)
    # - Early exit on high confidence (>0.8)
    # - 1.5s cooldown between detections
    
    print("=" * 60)
    print(f"        WAKE WORD DETECTOR: '{WAKE_WORD.upper()}'")
    print("=" * 60)
    print()
    
    detector = WakeWordDetector(
        wake_word=WAKE_WORD, 
        sample_rate=SAMPLE_RATE, 
        threshold=THRESHOLD,
        augment_data=AUGMENT_DATA
    )

    print()
    
    if not detector.load_templates():
        print("=" * 60)
        print(f"SETUP: Record '{WAKE_WORD.upper()}' 5 times")
        print("=" * 60)
        print()
        print("HOW IT WORKS:")
        print("- The system will listen for you to start speaking")
        print("- Say your wake word clearly")
        print("- It automatically stops recording when you finish")
        print("- Augments each recording (speed/pitch variations)")
        print("- To start over, delete the hey_tars_templates.pkl file")
        print()
        print("TIPS FOR BEST RESULTS:")
        print("- Speak naturally at normal volume")
        print("- Say the phrase the same way you'll use it")
        print("- Record in the same environment you'll use it")
        print("- Keep background noise consistent")
        print("- Vary slightly between recordings (speed/tone)")
        print()
        print("NOTE: Each recording creates ~6 templates via augmentation")
        print("      (original + speed/pitch variations)")
        print()
        print("Press ENTER when ready to start recording...")
        input()
        print()
        
        num_templates = 5
        for i in range(num_templates):
            success = detector.record_template()
            
            if not success and i == 0:
                print("\nLet's try again. Speak louder and clearer!")
                detector.record_template()
            
            if i < num_templates - 1:
                print("\n   Preparing for next recording...\n")
                time.sleep(1)
        
        detector.save_templates()
        print()
        print(f"Training complete! Created {len(detector.templates)} total templates")
        print()
    
    print("=" * 60)
    print(f"[READY] Say '{WAKE_WORD.upper()}' to trigger detection")
    print("   (Press Ctrl+C to stop)")
    print("=" * 60)
    print()
    
    try:
        def audio_callback(indata, frames, time_info, status):
            audio_np = indata[:, 0]
            detector.add_audio(audio_np)
            
            detected, confidence = detector.detect()
            
            if confidence > 0.3:
                bars = int(confidence * 20)
                print(f"\r[{'█' * bars}{' ' * (20-bars)}] {confidence:.2f}", end='', flush=True)
            
            if detected:
                print("\n" + "=" * 60)
                print(f">>> WAKE WORD DETECTED <<<")
                print(f"    Confidence: {confidence:.2f}")
                print("=" * 60)
                print()
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                           callback=audio_callback, blocksize=512):  # Smaller blocks = faster response
            while True:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        print("Done!")


if __name__ == "__main__":
    main()
