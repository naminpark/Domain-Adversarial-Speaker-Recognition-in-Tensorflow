#__author__ = 'naminpark'

import tensorflow as tf
from vadFILE import *
import os
import webrtcvad

def allfiles(path):
    res = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.wav':
                filepath = os.path.join(root, file)
                res.append(filepath)

    return res

inputPath='/Users/naminpark/Desktop/test/python_speech_analysis_synthesis/speech/'
outputPath='/Users/naminpark/Desktop/test/python_speech_analysis_synthesis/VAD/'

fileList=allfiles(inputPath)
if not os.path.isdir(outputPath):
    os.mkdir(outputPath)



VF=vadFILE()

for filename in fileList:
    audio, sample_rate = VF.read_wave(filename)
    vad = webrtcvad.Vad(3) # 0~3
    frames = VF.frame_generator(20, audio, sample_rate)
    frames = list(frames)
    segments = VF.vad_collector(sample_rate, 20, 300, vad, frames)
    #write_wave('chunk.wav', segments, sample_rate)
    wav_data=""
    for i, segment in enumerate(segments):
        wav_data+=segment

    splitfilename=filename.split('/')

    filename=splitfilename[-1].split(".")[0]+'_VAD.wav'
    VF.write_wave(outputPath+filename, wav_data, sample_rate)

