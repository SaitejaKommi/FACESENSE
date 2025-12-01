import cv2
import numpy as np

def is_spoof(frame):
    if frame is None:
        return True

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (120, 120))

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = 20 * np.log(np.abs(fshift) + 1e-9)

    mean_val = np.mean(magnitude)
    var_val = np.var(magnitude)

    # If too smooth → printed photo
    if mean_val < 5 or var_val < 20:
        return True
    return False
