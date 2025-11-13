import os
import pandas as pd
import scipy as sc
import numpy as np
import plotly.express as px

origSamplingRate = 1000
newSamplingRate = 200
q = int(origSamplingRate/newSamplingRate) # step size for down sampling
windowSize=4 #4 seconds
overlapSize=0.1 #percent of overlapped points between segments
noOfSamples = newSamplingRate * windowSize # = 800
bandpassWindow = (4,50) #Hz


def loadSession(k, basePath='SEED-IV'):
    sessionPath=f'{basePath}/eeg_raw_data/{k}/'
    sessionSubjects=os.listdir(sessionPath)
    s=[]
    for i,subjectFile in enumerate(sessionSubjects):
        sub=sc.io.loadmat(sessionPath+subjectFile)
        # sub = {int(re.search(r'(\d+)$', k).group(1))-1: v for k, v in sub.items() if not k.startswith('__')}
        sub = [v for k, v in sub.items() if not k.startswith('__')]
        s.append(sub)
    return s


def getChannel(channel,basePath='SEED-IV'):
    channelsMapping=pd.read_excel(f'{basePath}/Channel Order.xlsx',header=None, names=['channels']).reset_index() 
    channelsMapping.set_index('channels', inplace=True)
    return channelsMapping.loc[channel]['index'] 

def loadSubject(session,subject,basePath='SEED-IV'):
    '''This function is 1-based'''
    for file in os.listdir(f'{basePath}/eeg_raw_data/{session}/'):
        if file.startswith(f'{subject}_'):
            subData=sc.io.loadmat(f'{basePath}/eeg_raw_data/{session}/{file}')
            break
    subData = [v for k, v in subData.items() if not k.startswith('__')]
    return subData

def downSample(trial):
    return np.array([ch[::q] for ch in trial])

def segmentChannel(ch):
    '''
    This function segments the channel with window size of 800 samples while applying overlapping of size 10% , additionally if the 
    channel isn't divisible by the window size , the last segment will be ch[-window size] , which means its overlap with the previous
    segment can be any value from 10% to 99%
    '''
    s = []
    stepSize= int(newSamplingRate * windowSize *(1-overlapSize))
    segmentsCount = int(np.floor((len(ch) - noOfSamples) / stepSize)) + 1
    for i in range(segmentsCount):
        start=i*stepSize
        end=(i*stepSize)+noOfSamples
        s.append(ch[start:end])

    #to cover the whole signal
    if end+1< len(ch):
        s.append(ch[-noOfSamples:])
    return np.array(s)

def segmentTrial(trial):
    return [segmentChannel(ch) for ch in trial]

def preProcess(subData):
    f'''This function applies band pass filter {bandpassWindow} then down sampling to 200 Hz'''
    b, a = sc.signal.butter(4, Wn=bandpassWindow, btype='bandpass', fs=origSamplingRate)
    s = [sc.signal.lfilter(b, a, trial) for trial in subData]
    s = [downSample(trial)  for trial in s]
    s = [sc.stats.zscore(trial, axis=1) for trial in s]
    s = [segmentTrial(trial)  for trial in s]
    return s

def draw(s,rndr=''):
    fig = px.line(s)
    fig.show(renderer=rndr)