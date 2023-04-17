import cv2
import skimage
import os
import numpy as np
from scipy.signal import lfilter, butter
from PIL import Image
import scipy.fftpack as fftpack




print('1. INITIALS AND COLOR TRANSFORMATION')

def read_video(data_dir):
    video = cv2.VideoCapture(data_dir)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = np.zeros((num_frames, H, W, 3), dtype=np.float32)
    for i in range(num_frames):
        _, frame = video.read()
        frames[i] = frame

    return frames


baby_dir = './data/baby2.mp4'
baby_video = read_video(baby_dir)

baby_YIQ = skimage.color.rgb2yiq(baby_video)

face_dir = './data/face.mp4'
face_video = read_video(face_dir)

face_YIQ = skimage.color.rgb2yiq(face_video)


print('2. LAPLACIAN PYRAMID')

def make_laplacian_pyr(img, pyramid_lv):
    tmp = img.copy()
    gaussian = [tmp]

    for i in range(pyramid_lv):
        tmp = cv2.pyrDown(tmp)
        gaussian.append(tmp)

    out = []
    for i in range(pyramid_lv, 0, -1):
        blurred = cv2.pyrUp(gaussian[i])
        diff = cv2.subtract(gaussian[i-1], blurred)
        out.append(diff)
    return out



def make_laplacian_vid(video, pyramid_lv, name=''):
    out = []
    os.makedirs('./Q2', exist_ok=True)

    for i in range(0, video.shape[0]):
        frame = video[i]
        laplacian_pyr = make_laplacian_pyr(frame, pyramid_lv)
        if i == 0:
            for k in range(pyramid_lv):
                out.append(np.zeros((video.shape[0], laplacian_pyr[k].shape[0], laplacian_pyr[k].shape[1], 3)))
        
        for n in range(pyramid_lv):
            out[n][i] = laplacian_pyr[n]
            if i == 0:
                cv2.imwrite('./Q2/{}_laplacian_frame_level{}.png'.format(name, str(n)), out[n][i]*255)

    return out


baby_laplacian = make_laplacian_vid(baby_YIQ, pyramid_lv=4, name='baby')
face_laplacian = make_laplacian_vid(face_YIQ, pyramid_lv=4, name='face')


print('3. TEMPORAL FILTERING & 4. EXTRACTING THE FREQUENCY BAND OF INTEREST')

def temporal_ideal_filter(frames, pyramid_lv, low, high, fps, axis=0):
    out = []
    for i in range(pyramid_lv):
        fft = fftpack.fft(frames[i], axis=axis)
        frequencies = fftpack.fftfreq(frames[i].shape[0], d=1.0 / fps)

        bound_l = (np.abs(frequencies - low)).argmin()
        bound_h = (np.abs(frequencies - high)).argmin()

        fft[:bound_l] = 0
        fft[bound_h:-bound_h] = 0
        fft[-bound_l:] = 0
        iff = fftpack.ifft(fft, axis=axis)
        out.append(np.abs(iff))

        plt.plot(len(baby_temporal), abs(baby_temporal))      
        plt.grid()
        plt.xlabel("Frequency(Hz)")
        plt.ylabel("Magnitude")
        plt.show()
        
    
    return out

baby_temporal = temporal_ideal_filter(baby_laplacian, 4, 1, 30, 150)

face_temporal = temporal_ideal_filter(face_laplacian, 4, 1, 30, 150)


print('5. IMAGE RECONSTRUCTION')

def inverse_lap(laps, pyramid_lv):
    ret = np.zeros(laps[-1].shape)
    for i in range(laps[0].shape[0]):
        a = 100.
        laps[0][i][:, :, 1:] *= a 
        up = laps[0][i]
        for j in range(pyramid_lv-1):
            laps[j+1][i][:, :, 1:] *= a
            up = cv2.pyrUp(up) + laps[j + 1][i]
            if j > (pyramid_lv//2 - 3):
                a/=2.0
        ret[i] = up
    return ret


def save_video(video, name):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [H,W]=video[0].shape[0:2]
    writer = cv2.VideoWriter(name + ".avi", fourcc, 30, (W, H), 1)
    
    for i in range(0,video.shape[0]):
        writer.write(cv2.convertScaleAbs(video[i]))
    writer.release()


baby_inv = inverse_lap(baby_temporal, pyramid_lv=4)
face_inv = inverse_lap(face_temporal, pyramid_lv=4)

baby_inv = baby_inv + baby_YIQ
baby_recon = skimage.color.yiq2rgb(baby_inv)

face_inv = face_inv + face_YIQ
face_recon = skimage.color.yiq2rgb(face_inv)

save_video(baby_recon, 'baby_recon')
save_video(face_recon, 'face_recon')
