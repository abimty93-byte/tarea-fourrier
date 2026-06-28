import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------------------------------------
# Fondo negro estilo moderno
plt.style.use("dark_background")

# Tiempo y muestreo
fs = 500                     # Frecuencia de muestreo
t = np.linspace(-1, 1, fs)   # Vector de tiempo

# ---------------------------------------------------------
# DEFINICIÓN DE SEÑALES BASE
# ---------------------------------------------------------
rect = np.where(np.abs(t) < 0.2, 1, 0)      # Pulso rectangular
escalon = np.where(t >= 0, 1, 0)            # Función escalón
seno = np.sin(2 * np.pi * 5 * t)            # Seno de 5 Hz

# Señal combinada (propiedad de linealidad)
lineal = seno + rect

# Señal escalada en frecuencia (propiedad de escalamiento)
seno_10hz = np.sin(2 * np.pi * 10 * t)

# ---------------------------------------------------------
# FIGURA PRINCIPAL
# ---------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))

# Gráfica de la señal
line_signal, = ax1.plot(t, seno, color="cyan", linewidth=2)
ax1.set_ylim(-2, 2)
ax1.set_title("Señal en el dominio del tiempo", color="white")

# Gráfica de la magnitud de la FFT
freqs = np.fft.fftfreq(len(t), 1/fs)
line_fft_mag, = ax2.plot(freqs[:fs//2], np.abs(np.fft.fft(seno))[:fs//2],
                         color="magenta", linewidth=2)
ax2.set_title("Magnitud de la FFT", color="white")
ax2.set_xlim(0, 50)

# Gráfica de la fase de la FFT
line_fft_phase, = ax3.plot(freqs[:fs//2], np.angle(np.fft.fft(seno))[:fs//2],
                           color="yellow", linewidth=2)
ax3.set_title("Fase de la FFT", color="white")
ax3.set_xlim(0, 50)

# ---------------------------------------------------------
# MODOS DE SEÑAL
# ---------------------------------------------------------
# 0 = seno 5 Hz
# 1 = pulso rectangular
# 2 = escalón
# 3 = linealidad (seno + pulso)
# 4 = escalamiento (seno 10 Hz)
modo = 0

def update(frame):
    global modo

    # Cambiar de señal cada cierto tiempo
    if frame % 200 == 0:
        modo = (modo + 1) % 5

    # -----------------------------------------------------
    # SELECCIÓN DE SEÑAL SEGÚN EL MODO
    # -----------------------------------------------------
    if modo == 0:
        y = np.sin(2 * np.pi * 5 * t + frame * 0.1)
        color = "cyan"
        ax1.set_title("Señal Senoidal (5 Hz)", color="white")

    elif modo == 1:
        y = np.where(np.abs(t + 0.3*np.sin(frame*0.05)) < 0.2, 1, 0)
        color = "lime"
        ax1.set_title("Pulso Rectangular (desplazamiento en el tiempo)", color="white")

    elif modo == 2:
        y = np.where(t + 0.3*np.sin(frame*0.05) >= 0, 1, 0)
        color = "magenta"
        ax1.set_title("Función Escalón (desplazamiento en el tiempo)", color="white")

    elif modo == 3:
        y = lineal
        color = "orange"
        ax1.set_title("Linealidad: Seno + Pulso", color="white")

    elif modo == 4:
        y = seno_10hz
        color = "red"
        ax1.set_title("Escalamiento en frecuencia: Seno de 10 Hz", color="white")

    # -----------------------------------------------------
    # ACTUALIZAR SEÑAL EN TIEMPO
    # -----------------------------------------------------
    line_signal.set_ydata(y)
    line_signal.set_color(color)

    # -----------------------------------------------------
    # FFT: MAGNITUD Y FASE
    # -----------------------------------------------------
    Y = np.fft.fft(y)

    # Magnitud
    line_fft_mag.set_ydata(np.abs(Y)[:fs//2])
    line_fft_mag.set_color(color)

    # Fase
    line_fft_phase.set_ydata(np.angle(Y)[:fs//2])
    line_fft_phase.set_color(color)

    return line_signal, line_fft_mag, line_fft_phase

# ---------------------------------------------------------
# ANIMACIÓN
# ---------------------------------------------------------
ani = FuncAnimation(fig, update, frames=1000, interval=30)
plt.tight_layout()
plt.show()
