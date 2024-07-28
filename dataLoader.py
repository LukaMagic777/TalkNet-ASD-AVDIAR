import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile

def generate_audio_set(dataPath, batchList, useAvdiar):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        if(useAvdiar):
            videoName = data[0][:13]
        else:
            videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):   
    noiseName = random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(numpy.int16)

def load_audio(data, dataPath, numFrames, audioAug, audioSet=None, randomInterval=False):
    dataName = data[0]
    fps = float(data[2])
    audio = audioSet[dataName]

    if audioAug:
        augType = random.randint(0, 1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)

    sample_rate = 16000
    interval_length = sample_rate

    audio_segments = []
    if randomInterval:
        # For training: randomly select a 1-second interval
        max_start = len(audio) - interval_length
        if max_start < 0:
            max_start = 0
        start = random.randint(0, max_start)
        audio = audio[start:start + interval_length]
        audio = python_speech_features.mfcc(audio, sample_rate, numcep=13, winlen=0.025 * 25 / fps, winstep=0.010 * 25 / fps)
        audio_segments.append(audio)
    else:
        # For evaluation: use every 1-second segment
        for start in range(0, len(audio), interval_length):
            segment = audio[start:start + interval_length]
            if len(segment) < interval_length:
                # Padding if the last segment is shorter than 1 second
                segment = numpy.pad(segment, (0, interval_length - len(segment)), 'wrap')
            mfcc = python_speech_features.mfcc(segment, sample_rate, numcep=13, winlen=0.025 * 25 / fps, winstep=0.010 * 25 / fps)
            audio_segments.append(mfcc)

    # Stack all segments and ensure correct shape
    audio = numpy.vstack(audio_segments)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage = maxAudio - audio.shape[0]
        audio = numpy.pad(audio, ((0, shortage), (0, 0)), 'wrap')
    audio = audio[:maxAudio, :]
    return audio

def load_visual(data, dataPath, numFrames, visualAug, useAvdiar, randomInterval=False):
    dataName = data[0]
    if useAvdiar:
        videoName = data[0][:13]
    else:
        videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg" % faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False)

    H = 112
    faces = []

    if visualAug:
        new = int(H * random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H / 2, H / 2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate'])
    else:
        augType = 'orig'

    if randomInterval:
        max_start = len(sortedFaceFiles) - numFrames
        start = random.randint(0, max_start)
        selectedFiles = sortedFaceFiles[start:start + numFrames]
    else:
        selectedFiles = sortedFaceFiles

    for faceFile in selectedFiles:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H, H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y + new, x:x + new], (H, H)))
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H, H)))
    faces = numpy.array(faces)
    return faces


def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, useAvdiar, **kwargs):
        self.audioPath = audioPath
        self.visualPath = visualPath
        self.miniBatch = []
        self.useAvdiar = useAvdiar
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         
        start = 0        
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end     

    def __getitem__(self, index):
        batchList = self.miniBatch[index]
        numFrames = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels = [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList, useAvdiar=self.useAvdiar) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')            
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug=True, audioSet=audioSet, randomInterval=True))  
            visualFeatures.append(load_visual(data, self.visualPath, numFrames, visualAug=True, useAvdiar=self.useAvdiar, randomInterval=True))
            labels.append(load_label(data, numFrames))
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)

class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, useAvdiar, **kwargs):
        self.audioPath = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.useAvdiar = useAvdiar

    def __getitem__(self, index):
        line = [self.miniBatch[index]]
        numFrames = int(line[0].split('\t')[1])
        audioSet = generate_audio_set(self.audioPath, line, useAvdiar=self.useAvdiar)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug=False, audioSet=audioSet, randomInterval=False)]
        visualFeatures = [load_visual(data, self.visualPath, numFrames, visualAug=False, useAvdiar=self.useAvdiar, randomInterval=False)]
        labels = [load_label(data, numFrames)]         
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)
