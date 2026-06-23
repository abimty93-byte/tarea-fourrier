import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Fondo negro estilo moderno
plt.style.use("dark_background")

# Tiempo y muestreo
fs = 500
t = np.linspace(-1, 1, fs)

# Señales base
rect = np.where(np.abs(t) < 0.2, 1, 0)
escalon = np.where(t >= 0, 1, 0)
seno = np.sin(2 * np.pi * 5 * t)

# Figura con dos gráficas: señal y FFT
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

# Línea de la señal
line_signal, = ax1.plot(t, seno, color="cyan", linewidth=2)
ax1.set_ylim(-2, 2)
ax1.set_title("Señal animada", color="white")

# Línea de la FFT
freqs = np.fft.fftfreq(len(t), 1/fs)
line_fft, = ax2.plot(freqs[:fs//2], np.abs(np.fft.fft(seno))[:fs//2],
                     color="magenta", linewidth=2)
ax2.set_title("FFT animada", color="white")
ax2.set_xlim(0, 50)

modo = 0  # 0=seno, 1=rect, 2=escalón

def update(frame):
    global modo

    # Cambiar de señal cada cierto tiempo
    if frame % 200 == 0:
        modo = (modo + 1) % 3

    # Selección de señal animada
    if modo == 0:
        y = np.sin(2 * np.pi * 5 * t + frame * 0.1)
        color = "cyan"
        ax1.set_title("Señal Senoidal (neón)", color="white")
    elif modo == 1:
        y = np.where(np.abs(t + 0.3*np.sin(frame*0.05)) < 0.2, 1, 0)
        color = "lime"
        ax1.set_title("Pulso Rectangular (neón)", color="white")
    else:
        y = np.where(t + 0.3*np.sin(frame*0.05) >= 0, 1, 0)
        color = "magenta"
        ax1.set_title("Función Escalón (neón)", color="white")

    # Actualizar señal
    line_signal.set_ydata(y)
    line_signal.set_color(color)

    # FFT animada
    Y = np.abs(np.fft.fft(y))[:fs//2]
    line_fft.set_ydata(Y)
    line_fft.set_color(color)

    return line_signal, line_fft

ani = FuncAnimation(fig, update, frames=1000, interval=30)
plt.tight_layout()
plt.show()
dir