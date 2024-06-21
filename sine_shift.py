# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:53:21 2024

@author: 20212238
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_2dft(input_image):
    ft = np.fft.ifftshift(input_image)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    return ft

x = np.arange(-500, 501, 1)
X, Y = np.meshgrid(x, x)

phase = 0 # Adjust value of the phase

wavelength_1 = 200
angle_1 = np.pi/3

grating_1 = np.sin(
    2*np.pi*(X*np.cos(angle_1) + Y*np.sin(angle_1) + phase) / wavelength_1
)
wavelength_2 = 100
angle_2 = np.pi/4

grating_2 = np.sin(
    2*np.pi*(X*np.cos(angle_2) + Y*np.sin(angle_2)  + phase) / wavelength_2
)
grating = grating_1 + grating_2
#grating = grating_1
grating_fr = calculate_2dft(grating)

fig, ax = plt.subplots(1,3,figsize=(10,30))
ax[0].imshow(grating, cmap='gray')
ax[0].set_title("Simulated image")
            
ax[1].imshow(abs(grating_fr), cmap='gray')
ax[1].set_title("Fourier Transform")
ax[1].set_xlim([480, 520])
ax[1].set_ylim([520, 480])

ax[2].imshow(np.log10(abs(grating_fr)), cmap='gray')
ax[2].set_title("Log. Fourier Transform")
ax[2].set_xlim([480, 520])
ax[2].set_ylim([520, 480])
plt.show()