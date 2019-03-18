
import numpy as np
from scipy.io import wavfile
import subprocess

wav_self_path = 'G:\yinpin_data\self_wav'

for i in range(1,21):

    cmd = 'ffmpeg -i '+str(i)+'.wav -ac 1 -ar 16000 ' +str(i)+'_.wav'
    print(cmd)



