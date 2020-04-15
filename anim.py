import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as pltanim

class Cercle:
    def __init__(self, ax, rayon=1, immersion=0, nb_pales=8, omega=0.2):
        ax.axis("equal")
        self.rayon = rayon
        self.immersion = immersion
        self.nb_pales = nb_pales
        self.omega = omega
        ax.plot([-rayon-1,rayon+1],[-(rayon-immersion),-(rayon-immersion)],'b', lw=5)
        self.lines, self.pales = [], []
        for i in range(nb_pales):
            line, = ax.plot([], [], 'r-', lw=2)
            self.lines.append(line)
            line, = ax.plot([], [], 'go-', lw=2)
            self.pales.append(line)
        # self.line, = ax.plot([], [], 'bo-', lw=2, label="line")
        ax.set_xlim(-2-rayon, 2+rayon)
        ax.set_ylim(-2-rayon, 2+rayon)
        self.timetext = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        theta = np.linspace(-np.pi, np.pi, 1000)
        ax.plot(rayon*np.cos(theta), rayon*np.sin(theta))

    def __call__(self, t):
        nb_pales, rayon, omega = self.nb_pales, self.rayon, self.omega
        theta = np.linspace(-np.pi,np.pi,nb_pales+1)
        xp = rayon * np.cos(theta + omega * t)
        yp = rayon * np.sin(theta + omega * t)
        # r = 1 + np.cos(t)
        # plt.plot([0, xp[i]], [0, yp[i]])
        a = 2/3
        for i in range(nb_pales):
            self.lines[i].set_data([0, a*xp[i]], [0, a*yp[i]])
            self.pales[i].set_data([a*xp[i], xp[i]], [a*yp[i], yp[i]])
        self.timetext.set_text(f"time = {t:.2f}s")
        return tuple(self.lines)+tuple(self.pales)+(self.timetext,)

fig = plt.figure()
np_pales = [6, 7, 8, 9, 10, 11, 12]
for np_pale in np_pales:
    plt.cla()
    cercle = Cercle(fig.gca(), nb_pales=np_pale, rayon=3, omega=0.64, immersion=1.1)
    t = np.linspace(0, 2*np.pi, 111)
    anim = pltanim.FuncAnimation(fig, cercle, t, interval=60, blit=True, repeat=False)
    anim.save(f"pales_{np_pale}.gif", writer='imagemagick', fps=60)
plt.show()