import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

bips = [27.18, 28.34, 122.211, 92.53, 16.69, 8, 382.8, 89.08, 72.49, 257.144, 18.2]

Gesamthandel = np.array([
    [0, 62869641, 236278458, 2617827455, 256359452, 391545338, 255931660, 2727614359, 1332965878, 189842340, 10409493],      # Albanien 
    [62869641, 0, 236278458, 2617827455, 256359452, 391545338, 255931660, 2727614359, 1332965878, 189842340, 10409493],      # Bosnien
    [329062549, 156706037, 0, 690811814, 1386488206, 27230582, 7125013542, 2374836843, 518251913, 4164373274, 332559004],    # Bulgarien
    [141700076, 4221016872, 790582633, 0, 359201092, 415713482, 998304881, 2691886435, 8220096009, 634991517, 22396827],     # Kroatien
    [214098518, 248724059, 1109725659, 253848342, 0, 64006207, 356663189, 1730372019, 325649838, 1351213639, 13943769],      # Nordmazedonien
    [123756098, 285198084, 25464614, 259840836, 59046876, 0, 42081397, 960593627, 103614668, 263814369, 2124952],            # Montenegro 
    [146198525, 146198525, 10526148114, 903520518, 402937957, 32474866, 0, 2546098776, 1716098020, 4063371753, 3671060032],  # Rumänien
    [368265709, 3145884205, 1868645351, 2043177500, 1508871560, 1572171916, 2415969590, 0, 2701924418, 993512263, 75964909], # Serbien
    [122363731, 1911830218, 614498797, 7274315286, 485771763, 153625357, 1382954138, 2508387553, 0, 629481311, 42709860],    # Slowenien
    [1295576799, 93280008, 5795775168, 554921154, 1897685806, 259994754, 3086150018, 1037693739, 778560032, 0, 122198050],   # Griechenland
    [9375447, 10113286, 267703808, 16981288, 12471916, 2346218, 2689597324, 73450158, 38622193, 133470695, 0]                # Moldawien
], dtype ='float64')

Distanz = np.array([
    [0, 304, 327, 587, 153, 131, 617, 388, 677, 500, 958], 
    [304, 0, 418, 290, 321, 173, 617, 193, 393, 791, 887], 
    [327, 418, 0, 679, 174, 334, 295, 327, 793, 525, 648],   
    [587, 290, 679, 0, 608, 458, 808, 368, 117, 1080, 995],
    [153, 321, 174, 608, 0, 185, 465, 320, 714, 487, 811], 
    [131, 173, 334, 458, 185, 0, 595, 279, 553, 623, 913], 
    [617, 617, 295, 808, 465, 595, 0, 449, 925, 743, 358], 
    [388, 193, 327, 368, 320, 279, 449, 0, 485, 804, 695], 
    [677, 393, 793, 117, 714, 553, 925, 485, 0, 1176, 1102],
    [500, 791, 525, 1080, 487, 623, 743, 804, 1176, 0, 1088],
    [958, 887, 648, 995, 811, 913, 358, 695, 1102, 1088, 0]
], dtype ='float64')

# Ländernamen für Legende 
laender = ['Albanien', 'Bosnien', 'Bulgarien', 'Kroatien', 'Nordmazedonien', 
           'Montenegro', 'Rumänien', 'Serbien', 'Slowenien', 'Griechenland', 
           'Moldawien']

x_all = []
y_all = []

# Plot erstellen
plt.figure(figsize=(14, 8))

for i in range(len(laender)): 

    x = Distanz[i]
    y = Gesamthandel[i]

    for k in range(0, len(laender)):
        y[k] /= bips[k] / bips[i] # relative Vergleichbarkeit trotz Größenunterschiede 

    m_val = max(y)

    for k in range(0, len(laender)): 
        y[k] /= m_val # Normierung der einzelnen Gesamthandelsdaten --> realtive Vergleichbarkeit 

    sort_idx = np.argsort(x) # Sortierung der Werte nach dem Index - so bleiben x und y zusammen 
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    x_all.append(x_sorted)
    y_all.append(y_sorted)

    mask_1 = x_sorted > 0 
    x_sorted_1 = x_sorted[mask_1]
    y_sorted_1 = y_sorted[mask_1]

    plt.plot(x_sorted_1, y_sorted_1, marker='o', label=laender[i], linewidth=2, markersize=4) # hier schon plotten, damit jedes Land eine eigene Kurve bekommt 

plt.xlabel('Distanz (km)', fontsize=12)
plt.ylabel('Gesamthandel (Millionen und Milliarden)', fontsize=12)
plt.title('Handelsvolumen in Abhängigkeit der Distanz', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10) #bbox_to_anchor - platziert Legende oben links (1.05 -> 5% vom Rand entfernt oder so )
plt.grid(True, alpha=0.3)
plt.tight_layout() # berechnet eigenständig die Abstände
plt.show()



x_all = np.concatenate(x_all)  
y_all = np.concatenate(y_all) # man braucht ein 1-Dimensionalen Array, um eine Funktion zu fitten

mask = x_all > 0
x_all = x_all[mask]
y_all = y_all[mask]

def exp_function(x, A, k): 
    return A * np.exp(k*x)

p0 = [1, 0.001]

parameter, abweichung = curve_fit(exp_function, x_all, y_all, p0=p0)

fit_A = parameter[0]
fit_k = parameter[1]

print(f"Parameter A: {fit_A}")
print(f"Parameter k: {fit_k}")

fit_y = exp_function(x_all, fit_A, fit_k)

plt.figure(figsize=(12, 6))

# Originaldatenpunkte
#plt.scatter(x_all, y_all, color='blue', alpha=0.5, s=30, label='Normierte Daten')

country_idx = []
for i in range(len(laender)):
    n = len(Distanz[i])
    idx = np.argsort(Distanz[i])
    mask = Distanz[i][idx] > 0
    country_idx.extend([i] * np.sum(mask))
country_idx = np.array(country_idx)

cmap = plt.get_cmap('tab20')

for i in range(len(laender)):
    mask_i = country_idx == i
    plt.scatter(
        x_all[mask_i],
        y_all[mask_i],
        alpha=1,
        s=30,
        color=cmap(i % 100),
        label=laender[i]
    )

# Gefittete Kurve
x_fit = np.linspace(min(x_all), max(x_all), 200)
y_fit = exp_function(x_fit, *parameter)

plt.plot(x_fit, y_fit, color='red', linewidth=3)
plt.xlabel("Distanz [km]", fontsize = 14 )
plt.ylabel("relativer Gesamthandel (Milliarden)", fontsize = 14)
plt.title("Durchschnittliches Handelsvolumen in Abhängigkeit der Distanz (Realdaten)", fontsize = 14, fontweight = "bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()



