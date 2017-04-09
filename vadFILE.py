import collections
import contextlib
import sys
import wave
import webrtcvad


class vadFILE():

    def __init__(self):
        pass


    def read_wave(self,path):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate


    def write_wave(self,path, audio, sample_rate):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)




    def frame_generator(self,frame_duration_ms, audio, sample_rate):
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n


    def vad_collector(self,sample_rate, frame_duration_ms,
                      padding_duration_ms, vad, frames):
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        for frame in frames:
            sys.stdout.write(
                '1' if vad.is_speech(frame.bytes, sample_rate) else '0')
            if not triggered:
                ring_buffer.append(frame)
                num_voiced = len([f for f in ring_buffer
                                  if vad.is_speech(f.bytes, sample_rate)])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp,))
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append(frame)
                num_unvoiced = len([f for f in ring_buffer
                                    if not vad.is_speech(f.bytes, sample_rate)])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        if triggered:
            sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        sys.stdout.write('\n')
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames])


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

if __name__ == '__main__':
    VF=vadFILE()
    audio, sample_rate = VF.read_wave('test.wav')
    vad = webrtcvad.Vad(3) # 0~3
    frames = VF.frame_generator(20, audio, sample_rate)
    frames = list(frames)
    segments = VF.vad_collector(sample_rate, 20, 300, vad, frames)
    #write_wave('chunk.wav', segments, sample_rate)
    wav_data=""
    for i, segment in enumerate(segments):
        wav_data+=segment
    VF.write_wave('chunk1.wav', wav_data, sample_rate)