import numpy as np
import random 
import math 
import matplotlib.pyplot as plt
from scipy.integrate import RK45 
from scipy.optimize import curve_fit
import pandas as pd 


random.seed(48) 
np.random.seed(48)

#mögliche Koordinaten festlegen - aus denen wird dann zufällig eine Koordinate für ein Land erstellt. 
possible_positions = set()
for i in range (10):
    cord_x = int(random.uniform(0,1000))
    cord_y = int(random.uniform(0,1000))    
    possible_positions.add((cord_x,cord_y))

class country(): #1 --> alle Angaben für Produkt 1
    def __init__(self, name, waste1 = None, prodcution1_m = None, S01 = None, waste2 = None, production2_m = None, S02 = None, max1 = None, max2 = None, position_x = None, position_y = None, position = None):
        self.name = name 
        self.waste1 = waste1 or round(random.uniform(0.2, 0.4), 2)
        self.production1_m = prodcution1_m or round(random.uniform(0, 200), 1) #50
        self.S01 = S01 or round(random.uniform(0, 0), 1)
        self.max1 = max1 or round(random.uniform(50, 100000), 2) #60

        # Positionen
        if position_x is None or position_y is None:
            if possible_positions:
                pos = possible_positions.pop()  # Nimmt eine Position raus
                self.position_x = pos[0]
                self.position_y = pos[1]
            else:
                print("Warnung: Keine Positionen mehr verfügbar!")
                self.position_x = 0
                self.position_y = 0
        else:
            self.position_x = position_x
            self.position_y = position_y
        
        self.position = (position_x, position_y) or (0,0)

print("1. Waachstumsrate, 2. Verbrauch, 3. anfänglicher Lagerstand") 
countryA = country("A")
print("Land A: ")
print("Produkt 1 - Konditionen", countryA.production1_m, countryA.waste1, countryA.S01, countryA.max1) # theorethischer Lagerstand des Produktes 1 bei 10 schritten
print("Land A Position: ", countryA.position_x)

countryB = country("B")
print("Land B: ")
print("Produkt 1 - Konditionen", countryB.production1_m, countryB.waste1, countryB.S01) 

countryC = country("C")
print("Land C: ")
print("Produkt 1 - Konditionen", countryC.production1_m, countryC.waste1, countryC.S01)

countryD = country("D")
print("Land D: ")
print("Produkt 1 - Konditionen", countryD.production1_m, countryD.waste1, countryD.S01)

countryE = country("E")
print("Land E: ")
print("Produkt 1 - Konditionen", countryE.production1_m, countryE.waste1, countryE.S01)

countryF = country("F")
print("Land F: ")
print("Produkt 1 - Konditionen", countryF.production1_m, countryF.waste1, countryF.S01)

countries = [countryA, countryB, countryC, countryD, countryE, countryF]

# Distanz berechnen: 
distances = np.zeros((6,6))
for a, La in enumerate(countries): 
    for b, L in enumerate(countries): #enumerate: loop durch Liste 
        Land1 = La
        Land2 = L

        d = math.sqrt((Land2.position_x - Land1.position_x)**2 + (Land2.position_y - Land1.position_y)**2)
        distances[a, b] = d 
        print(f" Die Distanz zwischen {Land1.name} und {Land2.name} ist: ", d)

# Bedingungen festlegen: 
n = 1000
t_end = 100 #Zeitspanne 0 - 60
t0 = 0  
delta_t = t_end / n # Zeitschritt 

# die Produkte durch unterschiedliche Lager drennen: 
stock_1 = np.zeros((n+1, 6))
stock_2 = np.zeros((n+1, 6))

#k = Zölle, Umwelt etc. für Pr. 1
k = np.zeros((6, 6))
for i in range(6):
    for j in range(i + 1, 6):
        value = round(random.uniform(0, 0.01), 4)  # Viel kleinere Werte! 
        k[i, j] = value * (1 / distances[i,j]**2)
        k[j, i] = value * (1 / distances[i,j]**2) # k[1,0]

# Produktionen der einzelnen Länder speichern - damit man diese auf 0 setzen und wieder herstellen kann. 
productions_original = np.array([[countryA.production1_m],
                       [countryB.production1_m], 
                       [countryC.production1_m],
                       [countryD.production1_m], 
                       [countryE.production1_m], 
                       [countryF.production1_m]])
productions_custom = productions_original.copy()

rk_y = np.array([countryA.S01, countryB.S01, countryC.S01, countryD.S01, countryE.S01, countryF.S01]) # 1Pro.             # Definition durch des Startwertes der jeweiligen Länder rk_y wird als y verstanden  # 2 Pro. 

# Differenitalgleischung für die Wachstumsrate eines Landes - hier Land A: 
def fun(t,y): 
    SA, SB, SC, SD, SE, SF = y 

    dSAdt = (productions_custom[0,0] - countryA.waste1 * SA + 
         (k[1, 0] * SB * (countryA.max1 - SA) - k[0, 1] * SA * (countryB.max1 - SB)) + 
         (k[2, 0] * SC * (countryA.max1 - SA) - k[0, 2] * SA * (countryC.max1 - SC)) + 
         (k[3, 0] * SD * (countryA.max1 - SA) - k[0, 3] * SA * (countryD.max1 - SD)) + 
         (k[4, 0] * SE * (countryA.max1 - SA) - k[0, 4] * SA * (countryE.max1 - SE)) + 
         (k[5, 0] * SF * (countryA.max1 - SA) - k[0, 5] * SA * (countryF.max1 - SF)))

    dSBdt = (productions_custom[1,0] - countryB.waste1 * SB + 
         (k[0, 1] * SA * (countryB.max1 - SB) - k[1, 0] * SB * (countryA.max1 - SA)) + # einzeln rausschreiben ! 
         (k[2, 1] * SC * (countryB.max1 - SB) - k[1, 2] * SB * (countryC.max1 - SC)) + 
         (k[3, 1] * SD * (countryB.max1 - SB) - k[1, 3] * SB * (countryD.max1 - SD)) + 
         (k[4, 1] * SE * (countryB.max1 - SB) - k[1, 4] * SB * (countryE.max1 - SE)) + 
         (k[5, 1] * SF * (countryB.max1 - SB) - k[1, 5] * SB * (countryF.max1 - SF)))

    dSCdt = (productions_custom[2,0] - countryC.waste1 * SC + 
         (k[0, 2] * SA * (countryC.max1 - SC) - k[2, 0] * SC * (countryA.max1 - SA)) + 
         (k[1, 2] * SB * (countryC.max1 - SC) - k[2, 1] * SC * (countryB.max1 - SB)) + 
         (k[3, 2] * SD * (countryC.max1 - SC) - k[2, 3] * SC * (countryD.max1 - SD)) + 
         (k[4, 2] * SE * (countryC.max1 - SC) - k[2, 4] * SC * (countryE.max1 - SE)) + 
         (k[5, 2] * SF * (countryC.max1 - SC) - k[2, 5] * SC * (countryF.max1 - SF)))

    dSDdt = (productions_custom[3,0] - countryD.waste1 * SD + 
         (k[0, 3] * SA * (countryD.max1 - SD) - k[3, 0] * SD * (countryA.max1 - SA)) + 
         (k[1, 3] * SB * (countryD.max1 - SD) - k[3, 1] * SD * (countryB.max1 - SB)) + 
         (k[2, 3] * SC * (countryD.max1 - SD) - k[3, 2] * SD * (countryC.max1 - SC)) + 
         (k[4, 3] * SE * (countryD.max1 - SD) - k[3, 4] * SD * (countryE.max1 - SE)) + 
         (k[5, 3] * SF * (countryD.max1 - SD) - k[3, 5] * SD * (countryF.max1 - SF)))

    dSEdt = (productions_custom[4,0] - countryE.waste1 * SE + 
         (k[0, 4] * SA * (countryE.max1 - SE) - k[4, 0] * SE * (countryA.max1 - SA)) + 
         (k[1, 4] * SB * (countryE.max1 - SE) - k[4, 1] * SE * (countryB.max1 - SB)) + 
         (k[2, 4] * SC * (countryE.max1 - SE) - k[4, 2] * SE * (countryC.max1 - SC)) + 
         (k[3, 4] * SD * (countryE.max1 - SE) - k[4, 3] * SE * (countryD.max1 - SD)) + 
         (k[5, 4] * SF * (countryE.max1 - SE) - k[4, 5] * SE * (countryF.max1 - SF)))

    dSFdt = (productions_custom[5,0] - countryF.waste1 * SF + 
         (k[0, 5] * SA * (countryF.max1 - SF) - k[5, 0] * SF * (countryA.max1 - SA)) + 
         (k[1, 5] * SB * (countryF.max1 - SF) - k[5, 1] * SF * (countryB.max1 - SB)) + 
         (k[2, 5] * SC * (countryF.max1 - SF) - k[5, 2] * SF * (countryC.max1 - SC)) + 
         (k[3, 5] * SD * (countryF.max1 - SF) - k[5, 3] * SF * (countryD.max1 - SD)) + 
         (k[4, 5] * SE * (countryF.max1 - SF) - k[5, 4] * SF * (countryE.max1 - SE)))
    

    return np.array([dSAdt, dSBdt, dSCdt, dSDdt, dSEdt, dSFdt])

# für ein Land k Rechnungen extern rausschreiben + plotten - Export / Import abbilden --> ein Land 
# dSXdt plott? 

# RK45-Solver für jeden Zeitschritt
my_RK45 = RK45(fun, t0, rk_y, t_end)

print("Zeitschritt: " + str(delta_t))

# Erstellung der Arrays für Speicherung der Werte: 
f_res = np.zeros((n+1, 6)) # Achtung: keine eckigen Klammer - heißt man deffiniert nur das Aussehen des Array - quasi wie Matrix: n+1 Spalten, 2 Zeilen = 2 Länder 
t_res = np.zeros(n+1) 
f_res[0, :] = rk_y

#Gesamthandelsdaten für den realen Vergleich (Rheinfloge: A,B,C,D,E,F)
Gesamthandelsdaten_solo = np.zeros((6,6))
Gesamthandelsdaten_whole = np.zeros(6)

for i in range(0,n):
    my_RK45.t_bound = (i+1) * delta_t #quasi die Zeit 
    my_RK45.status = "running"
    while my_RK45.status != "finished":
        
        y_stock = my_RK45.y

        stock_1 = y_stock[:6]
        stock_2 = y_stock[6:]  


        # for a in range(6):
        #     for b in range(6):
        #         if a != b:
        #             import_ab = k[b, a] * y_stock[b] * (countries[a].max1 - y_stock[a]) # verallgemeinerung der Formel oben 
        #             export_ab = k[a, b] * y_stock[a] * (countries[b].max1 - y_stock[b])
        #             trade_ab = abs(import_ab) + abs(export_ab) #abs verhindert negative Zahlen - zur Berechnung des Gesamthandel Import + Export 
        #             Gesamthandelsdaten_solo[a,b] += trade_ab * delta_t 
        #             Gesamthandelsdaten_whole[a] += trade_ab * delta_t # 
                

        my_RK45.step()
    f_res[i+1, :] = my_RK45.y # speichert aktuellen Stand y (Lager-----wachstum??)
    t_res[i+1] = my_RK45.t # speichert aktuellen Stand t (Zeit)

VH = np.zeros((6,6))
VL = np.zeros((6))

for ti in range(n):  # n Schritte, delta_t konstant
    y = f_res[ti]  # Lagerstände zum Zeitpunkt ti
    for a in range(6):
        for b in range(6):  
            if a != b: 
                F_ab = k[a,b] * y[a] * (countries[b].max1 - y[b])
                F_ba = k[b,a] * y[b] * (countries[a].max1 - y[a])
                VH[a,b] += (F_ab + F_ba) * delta_t
                VL[a] += (F_ab + F_ba) * delta_t
print(f"Das ist A-B: {VH[0,1]} und B-A: {VH[1,0]}")
print(f"Das ist VL von A: {VL[0]}")
print("VH:")
print(VH)
print("----------------------------------------")
max_vals = np.max(VH, axis=1)

# Index des Partners b
partner_idx_max = np.argmax(VH, axis=1)
partner_idx_min = np.argmin(VH>0, axis=1)

for a in range(6):
    b = partner_idx_max[a]
    print(f"Land {countries[a].name} stärkster Partner: {countries[b].name} mit VH={max_vals[a]}")
print("----------------------------------------")
#print(f"Das größte Handelsvolumen ist: {max(VH)}")
# Erstellung der Ausgangssituationen 
baseline_f_res = f_res.copy() 
baseline_Gesamdhandelsdaten_solo = VH.copy()
baseline_Gesamdhandelsdaten_whole = VL.copy()
k_baseline = k.copy()
print("----------------------------------------")
print("Zur Identifizierung des stärksten Akteurs:")
print(f"Gesamthandelsdaten A: {baseline_Gesamdhandelsdaten_whole[0]}")
print(f"Gesamthandelsdaten B: {baseline_Gesamdhandelsdaten_whole[1]}")
print(f"Gesamthandelsdaten C: {baseline_Gesamdhandelsdaten_whole[2]}")
print(f"Gesamthandelsdaten D: {baseline_Gesamdhandelsdaten_whole[3]}")
print(f"Gesamthandelsdaten E: {baseline_Gesamdhandelsdaten_whole[4]}")
print(f"Gesamthandelsdaten F: {baseline_Gesamdhandelsdaten_whole[5]}")
print("----------------------------------------")
print(f"größtes Volumen: {max(baseline_Gesamdhandelsdaten_whole)}")
print("----------------------------------------")
print("Zur Identifizierung des stärksten Handelspartner für Land A:")
print(f"Gesamthandelsdaten A-B: {baseline_Gesamdhandelsdaten_solo[0,1]}")
print(f"Gesamthandelsdaten A-C: {baseline_Gesamdhandelsdaten_solo[0,2]}")
print(f"Gesamthandelsdaten A-D: {baseline_Gesamdhandelsdaten_solo[0,3]}")
print(f"Gesamthandelsdaten A-E: {baseline_Gesamdhandelsdaten_solo[0,4]}")
print(f"Gesamthandelsdaten A-F: {baseline_Gesamdhandelsdaten_solo[0,5]}")

#Handelsströme 
k_shock = k.copy()
def handelsstrom(y, k_matrix): 
    strom = np.zeros((6,6))
    for a in range(6): 
        for b in range(6): 
            if a != b: 
                strom[a,b] = (
                    k_matrix[a,b] * y[a] * (countries[b].max1 - y[b])
                  - k_matrix[b,a] * y[b] * (countries[a].max1 - y[a])
                )
    return strom 

Handlesstrom_Daten = np.zeros((len(t_res), 6, 6))
trade_AB = np.zeros(len(t_res))
trade_BC = np.zeros(len(t_res))
trade_CD = np.zeros(len(t_res))
trade_DE = np.zeros(len(t_res))
trade_EF = np.zeros(len(t_res))
trade_BF = np.zeros(len(t_res))

trade_AC = np.zeros(len(t_res))
trade_AD = np.zeros(len(t_res))
trade_AE = np.zeros(len(t_res))
trade_AF = np.zeros(len(t_res))

for i in range(len(t_res)):
    strom_spez = handelsstrom(f_res[i], k_baseline)      # 6×6 Momentaufnahme - fres liefert quasi die Lagerstände 
    Handlesstrom_Daten[i, :, :] = strom_spez # speichern: Zeit, Land 1, Land 2 
    trade_AB[i] = abs(strom_spez[0,1])
    trade_BC[i] = abs(strom_spez[1,2]) 
    trade_CD[i] = abs(strom_spez[2,3])           # A -> B; Test  
    trade_DE[i] = abs(strom_spez[3,4])           # A -> B; Test  
    trade_EF[i] = abs(strom_spez[4,5]) 

    trade_BF[i] = abs(strom_spez[1,5]) 
    
    trade_AC[i] = strom_spez[0,2]
    trade_AD[i] = strom_spez[0,3]
    trade_AE[i] = strom_spez[0,4]
    trade_AF[i] = strom_spez[0,5]          # A -> B; Test  

print("Zur Analyse des wichtigsten Handelstroms:")
print(f"A - B: {trade_AB}")
print(f"B - C: {trade_BC}")
print(f"C - D: {trade_CD}")
print(f"D - E: {trade_DE}")
print(f"E - F: {trade_EF}")


baseline_Handlesstrom_Daten = Handlesstrom_Daten.copy()
'''
# Handelsströme vor dem Schock plot
plt.plot(t_res, trade_AB, label = "trade_AB", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_BC, label = "trade_BC", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_CD, label = "trade_CD", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_DE, label = "trade_DE", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_EF, label = "trade_EF", linewidth = 2, markersize = 4)
plt.xlabel("Zeit", fontsize = 12)
plt.ylabel("Handelsstrom", fontsize = 12)
plt.title("Handelsströme vor dem Schock")
plt.grid(True, alpha = 0.3)
plt.legend()
plt.show()
'''
'''
# Land A einzelne Handelsströme, um Bilateralen-Schaden zu ermitteln: 
plt.plot(t_res, trade_AB, label = "trade_AB", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_AC, label = "trade_AC", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_AD, label = "trade_AD", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_AE, label = "trade_AE", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_AF, label = "trade_AF", linewidth = 2, markersize = 4)
plt.xlabel("Zeit", fontsize = 12)
plt.ylabel("Handelsstrom", fontsize = 12)
plt.title("Handelsströme ausgehen von Land A vor dem Schock")
plt.grid(True, alpha = 0.3)
plt.legend()
plt.show()
'''
# Distanz und Gesamthandel
x_all = []
y_all = []

for z, land in enumerate(countries):
    y_Handelsdaten = baseline_Gesamdhandelsdaten_solo[z, :]
    x_Distanz = distances[z, :]

    m_val = max(y_Handelsdaten)
    for l in range(0, len(countries)): 
        y_Handelsdaten[l] /= m_val

    sort_idx = np.argsort(x_Distanz)
    x_Distanz_sorted = x_Distanz[sort_idx]
    y_Handelsdaten_sorted = y_Handelsdaten[sort_idx]


    x_all.append(x_Distanz_sorted)
    y_all.append(y_Handelsdaten_sorted)

    print(f"Die Handelsdaten von {land.name} sind {y_Handelsdaten}")

    #plt.plot(x_Distanz_sorted, y_Handelsdaten_sorted, marker='o', label=land.name, linewidth=2, markersize=4)
'''
plt.xlabel("Distanz [km]", fontsize = 12 )
plt.ylabel(" Gesamthandel ", fontsize = 12)
plt.title("durchschnittliches Handelsvolumen in Abhängigkeit der Distanz", fontsize = 14, fontweight = "bold")
plt.grid(True, alpha=0.3)
plt.legend()
'''

# Funktion fitten 
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

'''
plt.figure(figsize=(12, 7))

# Originaldatenpunkte
plt.scatter(x_all, y_all, color='blue', alpha=0.5, s=30, label='Normierte Daten')

# Gefittete Kurve
x_fit = np.linspace(min(x_all), max(x_all), 200)
y_fit = exp_function(x_fit, *parameter)

plt.plot(x_fit, y_fit, color='red', linewidth=3, label = "Durchschnitsdatenpunkte (eigener Handel [0] ausgeschlossen)" )
plt.xlabel("Distanz [km]", fontsize = 12 )
plt.ylabel("raltiver Gesamthandel (Milliarden)", fontsize = 12)
plt.title("durchschnittliches Handelsvolumen in Abhängigkeit der Distanz", fontsize = 14, fontweight = "bold")
plt.legend()
plt.show()
'''
'''
# Position anzeigen: 
for  land in (countries):
    plt.scatter(land.position_x, land.position_y, s=150, alpha=0.6) # scatter: Werte sind mit einem Punkt gekennzeichnet 
    plt.annotate(f"Land {land.name}", #Beschriftung 
                 (land.position_x, land.position_y), #damit der richtige Name zum richtigen Punkt gesetzt wird 
                 xytext=(-2, 8), # Verschiebung in Pixeln sonst wären es Koordinaten !
                 textcoords='offset points',
                 fontsize=10)

plt.xlabel('X-Koordinate')
plt.ylabel('Y-Koordinate')
plt.title('Positionen der Länder')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1050)  # Etwas Rand um deine 0-1000 Koordinaten
plt.ylim(0, 1050)
plt.show()
'''

'''
#Lagerstände plotten - Baseline: 
plt.plot(t_res, f_res[:, 0], label="Land A-1")
plt.plot(t_res, f_res[:, 1], label="Land B-1")
plt.plot(t_res, f_res[:, 2], label="Land C-1")
plt.plot(t_res, f_res[:, 3], label="Land D-1")
plt.plot(t_res, f_res[:, 4], label="Land E-1")
plt.plot(t_res, f_res[:, 5], label="Land F-1")
plt.title("Lagerstände ohne Schock")
plt.xlabel("Zeit")
plt.ylabel("Lagerstand")
plt.legend() # zeigt Legende im Graph an 
plt.grid()
plt.show()
'''

Gesamthandelsdaten_solo_copy = Gesamthandelsdaten_solo.copy()
Gesamthandelsdaten_whole_copy = Gesamthandelsdaten_whole.copy()


# Funktion um eine Handelsbeziehung zwischen zwei Ländern zu stören
def shock_simulation(
        k_00 = None,
        k_10 = None,
        k_value0 = None,
        shock_start = 40.0,
        shock_end = 60.0,
        shock_edges = None, 
        alpha_edges = None, 
        specific_production = None,
        specific_production_value = None,  
        whole_production = None
    ): 

    production_base = productions_custom.copy()
    k_base = k.copy()
    k_shock = k.copy()
    rk_y_local = np.array([countryA.S01, countryB.S01, countryC.S01, countryD.S01, countryE.S01, countryF.S01]) 
    my_RK45 = RK45(fun, t0, rk_y_local, t_end)

    # Erstellung der Arrays für Speicherung der Werte: 
    f_res_simul = np.zeros((n+1, 6)) # Achtung: keine eckigen Klammer - heißt man deffiniert nur das Aussehen des Array - quasi wie Matrix: n+1 Spalten, 2 Zeilen = 2 Länder 
    t_res_simul = np.zeros(n+1) 
    f_res_simul[0, :] = rk_y_local

    #Gesamthandelsdaten für den realen Vergleich (Rheinfloge: A,B,C,D,E,F)
    Gesamthandelsdaten_solo_copy = np.zeros((6, 6))
    Gesamthandelsdaten_whole_copy = np.zeros(6)
    for i in range(0,n):
        my_RK45.t_bound = (i+1) * delta_t #quasi die Zeit 
        my_RK45.status = "running"
        while my_RK45.status != "finished":
            t_now = my_RK45.t
            y_stock = my_RK45.y

            if t_now >= shock_start and t_now <= shock_end:
                shock_situation = "active"
            else: 
                shock_situation = "unactiv"
            
            k[:,:] = k_base
            if shock_situation == "active" and (shock_edges is not None) and (alpha_edges is not None): 
                for (a, b) in shock_edges: 
                    a = int(a)
                    b = int(b)
                    k[a, b] = k_base[a,b] * float(alpha_edges) #* (1 / distances[k_00, k_10]**2)
                    k[b, a] = k_base[b,a] * float(alpha_edges)#* (1 / distances[k_00, k_10]**2)
            else: 
                k[:, :] = k_base

            # for a in range(6):
            #     for b in range(6):
            #         if a != b:
            #             import_ab = k[b, a] * y_stock[b] * (countries[a].max1 - y_stock[a]) # verallgemeinerung der Formel oben 
            #             export_ab = k[a, b] * y_stock[a] * (countries[b].max1 - y_stock[b])
            #             trade_ab = abs(import_ab) + abs(export_ab) #abs verhindert negative Zahlen 
            #             Gesamthandelsdaten_solo_copy[a,b] += trade_ab * delta_t 
            #             Gesamthandelsdaten_whole_copy[a] += trade_ab * delta_t                    
            my_RK45.step()
        f_res_simul[i+1, :] = my_RK45.y # speichert aktuellen Stand y  - hier muss Änderung von dSB etc. berechnet werden!! 
        t_res_simul[i+1] = my_RK45.t # speichert aktuellen Stand t (Zeit)
    k_shock_matrix = k_base.copy()
    if (shock_edges is not None): 
        for (a,b) in shock_edges: 
            a = int(a)
            b = int(b)

            k_shock_matrix[a, b] = k_base[a,b] * float(alpha_edges) # * (1 / distances[k_00, k_10]**2)
            k_shock_matrix[b, a] = k_base[b,a] * float(alpha_edges) # * (1 / distances[k_00, k_10]**2)
    
    VH_Schock = np.zeros((6,6))
    VL_Schock = np.zeros((6))
    # Handelsvolumen entlang der simulierten Trajektorie integrieren (zeitabhängiges k)
    for ti in range(n):  # 0..n-1 passt zu f_res_simul[ti]
        t_now = t_res_simul[ti]
        y = f_res_simul[ti]

        # pro Zeitschritt von der Baseline ausgehen
        k_Volumen = k_base.copy()
        if (
            (t_now >= shock_start)
            and (t_now <= shock_end)
            and(shock_edges is not None)
        ):
            for (a,b) in shock_edges: 
                a = int(a)
                b = int(b)
                k_Volumen[a, b] = k_base[a,b] * float(alpha_edges)
                k_Volumen[b, a] = k_base[b,a] * float(alpha_edges)

        y = f_res_simul[ti]  # Lagerstände zum Zeitpunkt ti
        for o in range(6):
            for i in range(6):  
                if o != i: 
                    F_ab = k_Volumen[o,i] * y[o] * (countries[i].max1 - y[i])
                    F_ba = k_Volumen[i,o] * y[i] * (countries[o].max1 - y[o])
                    VH_Schock[o,i] += (F_ab + F_ba) * delta_t
                    VL_Schock[o] += (F_ab + F_ba) * delta_t

    k[:, :] = k_base
    return f_res_simul, t_res_simul, VH_Schock, VL_Schock, k_shock_matrix
#f_res_simul, t_res_simul, Gesamthandelsdaten_solo_copy, Gesamthandelsdaten_whole_copy = shock_simulation(k_00=int(0), k_10=int(1), k_value0=float(0), specific_production=int(0), specific_production_value=float(0))

'''
print(f"Das ist f_res: {f_res_simul}")
print(f"Das ist t_res: {t_res_simul}")
print(f"Das ist Gesamthandelsdaten_solo: {Gesamthandelsdaten_solo_copy}")
print(f"Das ist Gesamthandelsdaten_whole: {Gesamthandelsdaten_whole_copy}")
'''

#bilateraler Schock 
shock_pairs = [(1,0),(1,2),(1,3),(1,4),(1,5)]
#shock_values = [100]
alphas = [2.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
results = []
for i,j in shock_pairs:
    for alpha in alphas:
        f_res_sim, t_res_sim, solo, whole, k_shock_mat = shock_simulation(shock_edges=[(1,0),(1,2),(1,3),(1,4),(1,5)], alpha_edges=alpha, shock_start=40, shock_end=60) #davor 80
        results.append({
            "shock_pair": (i,j),
            "alpha":alpha,
            "k_value": k_baseline[i, j] * alpha,
            "k_shock": k_shock_mat,
            "f_res": f_res_sim,       # Lagerstände
            "t_res": t_res_sim,       # Zeitpunkte
            "total_trade": whole,
            "trade_paths": solo
        })

print(f"Das ist k_value: {results[0]['alpha']}")
# print(f"Das ist total_trade: {results[0]['total_trade']}")
# print(f"Das ist trade_paths: {results[0]['trade_paths']}")

land_names = ["A", "B", "C", "D", "E", "F"]

# Wähle ersten Eintrag

Handlesstrom_Daten_Schock = np.zeros((len(t_res), 6, 6))
trade_AB_Schock = np.zeros(len(t_res))
trade_BC_Schock = np.zeros(len(t_res))
trade_CD_Schock = np.zeros(len(t_res))
trade_DE_Schock = np.zeros(len(t_res))
trade_EF_Schock = np.zeros(len(t_res))
trade_BF_Schock = np.zeros(len(t_res))

trade_AC_Schock = np.zeros(len(t_res))
trade_AD_Schock = np.zeros(len(t_res))
trade_AE_Schock = np.zeros(len(t_res))
trade_AF_Schock = np.zeros(len(t_res))

# Schockzeit markieren
shock_start = 40
shock_end = 60

print(f"Das ist k[0,4]: {k[0,4]}")

print(f"Das ist k_value: {results[0]['k_value']}")
print(f"Das ist total_trade: {results[0]['total_trade']}")
print(f"Das ist trade_paths: {results[0]['trade_paths']}")
# print(f"Das ist trade_AB: f{trade_AB_Schock}")
# print(f"Das ist Handelstrom_Daten: {Handlesstrom_Daten_Schock}")

# Wähle ersten Eintrag
run = results[6]
f_res = run["f_res"] # Daten-Zugriff 
t_res = run["t_res"]
k_shock_global = run["k_shock"]

for i in range(len(t_res)):
    if t_res[i] >= shock_start and t_res[i] <= shock_end: 
        k_now = k_shock_global
    else: 
        k_now = k_baseline
    
    strom_spez = handelsstrom(f_res[i], k_now)      # 6×6 Momentaufnahme      # 6×6 Momentaufnahme - fres liefert quasi die Lagerstände 
    Handlesstrom_Daten_Schock[i, :, :] = strom_spez # speichern: Zeit, Land 1, Land 2 
    trade_AB_Schock[i] = abs(strom_spez[0,1])
    trade_BC_Schock[i] = abs(strom_spez[1,2])# um es vergleichbar zu machen 
    trade_CD_Schock[i] = abs(strom_spez[2,3])
    trade_DE_Schock[i] = abs(strom_spez[3,4])
    trade_EF_Schock[i] = abs(strom_spez[4,5])
    trade_BF_Schock[i] = abs(strom_spez[1,5])

    trade_AC_Schock[i] = abs(strom_spez[0,2])
    trade_AD_Schock[i] = abs(strom_spez[0,3])
    trade_AE_Schock[i] = abs(strom_spez[0,4])
    trade_AF_Schock[i] = abs(strom_spez[0,5])


plt.figure(figsize=(12,6))
for i in range(6):
    plt.plot(t_res, f_res[:, i], label=f"Land {land_names[i]}")

# Schockzeit markieren
shock_start = 40
shock_end = 60

plt.axvline(x=shock_start, color='red', linestyle='-.', label='Schock')
plt.axvline(x=shock_end, color='red', linestyle='-.', label='Schock')
plt.xlabel("Zeit")
plt.ylabel("Lagerstand")
plt.title(f"Lagerstände der Länder über Zeit für Schock {run['shock_pair']} mit k_value={run['k_value']}")
plt.legend()
plt.grid(True)
plt.show()


'''
plt.plot(t_res, trade_AB_Schock, label = "trade_AB_Schock", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_BC_Schock, label = "trade_BC_Schock", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_CD_Schock, label = "trade_CD_Schock", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_DE_Schock, label = "trade_DE_Schock", linewidth = 2, markersize = 4)
plt.plot(t_res, trade_EF_Schock, label = "trade_CD_Schock", linewidth = 2, markersize = 4)
plt.xlabel("Zeit", fontsize = 12)
plt.ylabel("Handelsstrom", fontsize = 12)
plt.title("Handelsstrom der direkten Partner nach Schock")
plt.axvline(x=shock_start, color='red', linestyle='-.', label='Schock')
plt.axvline(x=shock_end, color='red', linestyle='-.', label='Schock')
plt.grid(True, alpha = 0.3)
plt.legend()
plt.show()
'''

print("k_base A->C,D,E,F:", k[0,2], k[0,3], k[0,4], k[0,5])


# Diagnose: Effekt auf B–C ist oft klein -> Delta-Plot gegen Baseline
# Baseline B–C (aus baseline_Handlesstrom_Daten)
trade_AB_base = np.abs(baseline_Handlesstrom_Daten[:, 0, 1])
trade_BC_base = np.abs(baseline_Handlesstrom_Daten[:, 1, 2])
trade_CD_base = np.abs(baseline_Handlesstrom_Daten[:, 2, 3])
trade_DE_base = np.abs(baseline_Handlesstrom_Daten[:, 3, 4])
trade_EF_base = np.abs(baseline_Handlesstrom_Daten[:, 4, 5])

trade_AE_base = np.abs(baseline_Handlesstrom_Daten[:, 0, 4])
trade_BF_base = np.abs(baseline_Handlesstrom_Daten[:, 4, 5])


trade_BF_base = np.abs(baseline_Handlesstrom_Daten[:, 1, 5])

plt.figure(figsize=(12,4))
plt.plot(t_res, (trade_BC_base - trade_BC_Schock), label="Δ trade_BC (Schock - Baseline)")
plt.plot(t_res, (trade_AE_base - trade_AE_Schock), label="Δ trade_AE (Schock - Baseline)")
plt.plot(t_res, (trade_BF_base - trade_BF_Schock), label="Δ trade_BF (Schock - Baseline)")
plt.plot(t_res, (trade_DE_base - trade_DE_Schock), label="Δ trade_DE (schwächste Route) (Schock - Baseline)")
plt.axvline(x=shock_start, color='red', linestyle='-.', label='Schock')
plt.axvline(x=shock_end, color='red', linestyle='-.', label='Schock')
plt.xlabel("Zeit")
plt.ylabel("Δ Handelsstrom")
plt.title("Differenz im Handelsströme (Schock - Baseline) (nur dominante Routen - und die Schwächste)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
#plt.plot(t_res, (trade_BC_base - trade_BC_Schock), label="Δ trade_BC (Schock - Baseline)")
plt.plot(t_res, (trade_AE_base - trade_AE_Schock), label="Δ trade_AE (Schock - Baseline)")
plt.plot(t_res, (trade_BF_base - trade_BF_Schock), label="Δ trade_BF (Schock - Baseline)")
plt.plot(t_res, (trade_DE_base - trade_DE_Schock), label="Δ trade_DE (schwächste Route) (Schock - Baseline)")
plt.axvline(x=shock_start, color='red', linestyle='-.', label='Schock')
plt.axvline(x=shock_end, color='red', linestyle='-.', label='Schock')
plt.xlabel("Zeit")
plt.ylabel("Δ Handelsstrom")
plt.title("Differenz im Handelsströme (Schock - Baseline) (nur dominante Routen - und die Schwächste); ohne Schockparteien")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

'''
plt.figure(figsize=(12, 4))
plt.plot(t_res, trade_AB_Schock - trade_AB_base, label="Δ trade_Ab (Schock - Baseline)")
plt.plot(t_res, trade_BC_Schock - trade_BC_base, label="Δ trade_BC (Schock - Baseline)")
plt.plot(t_res, trade_CD_Schock - trade_CD_base, label="Δ trade_CD (Schock - Baseline)")
plt.plot(t_res, trade_DE_Schock - trade_DE_base, label="Δ trade_DE (Schock - Baseline)")
plt.plot(t_res, trade_EF_Schock - trade_EF_base, label="Δ trade_EF (Schock - Baseline)")
plt.plot(t_res, trade_BF_Schock - trade_BF_base, label="Δ trade_BF (Schock - Baseline)")
plt.axvline(x=shock_start, color='red', linestyle='-.', label='Schock')
plt.axvline(x=shock_end, color='red', linestyle='-.', label='Schock')
plt.xlabel("Zeit")
plt.ylabel("Δ Handelsstrom")
plt.title("Differenz im Handelsströme (Schock - Baseline) (nur dominante Routen)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
'''
plt.figure(figsize=(12, 4))
plt.plot(t_res, trade_AB_Schock - trade_AB_base, label="Δ trade_Ab (Schock - Baseline)")
#plt.plot(t_res, trade_BC_Schock - trade_BC_base, label="Δ trade_BC (Schock - Baseline)")
plt.plot(t_res, trade_CD_Schock - trade_CD_base, label="Δ trade_CD (Schock - Baseline)")
plt.plot(t_res, trade_DE_Schock - trade_DE_base, label="Δ trade_DE (Schock - Baseline)")
plt.plot(t_res, trade_EF_Schock - trade_EF_base, label="Δ trade_EF (Schock - Baseline)")
plt.axvline(x=shock_start, color='red', linestyle='-.', label='Schock')
plt.axvline(x=shock_end, color='red', linestyle='-.', label='Schock')
plt.xlabel("Zeit")
plt.ylabel("Δ Handelsstrom")
plt.title("Differenz der direkten Handelsströme, einseitig (Schock - Baseline) ohne die Schockparteien  ")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


rows = []
for idx, alpha in enumerate(alphas):
    delta = baseline_Gesamdhandelsdaten_whole - results[idx]["total_trade"]
    rows.append({
        "alpha": alpha,
        "delta_trade_sum": float(np.sum(delta)),
        "delta_trade_A": float(delta[0]),
        "delta_trade_B": float(delta[1]),
        "delta_trade_C": float(delta[2]),
        "delta_trade_D": float(delta[3]),
        "delta_trade_E": float(delta[4]),
        "delta_trade_F": float(delta[5]),
    })

df = pd.DataFrame(rows)
# Optionen setzen, um alle Zeilen und Spalten anzuzeigen
pd.set_option('display.max_rows', None) # Zeigt alle Zeilen
pd.set_option('display.max_columns', None) # Zeigt alle Spalten
pd.set_option('display.width', None) # Verhindert Zeilenumbrüche (optional)
pd.set_option('display.max_colwidth', None) # Zeigt vollständigen Inhalt der Spalten an (optional)

#Tabelle 
# Optional: nach alpha sortieren (für logischere Darstellung)
df_show = df.sort_values("alpha", ascending=False).reset_index(drop=True)

print(df_show)

# 3) Plots aus der Tabelle
# (a) Gesamtdelta über alpha
plt.figure(figsize=(10, 5))
plt.plot(df_show["alpha"], df_show["delta_trade_sum"], marker="o")
plt.xlabel("alpha")
plt.ylabel("Δ Gesamthandel (Summe)")
plt.title("Gesamthandelsdifferenz des Systems (baseline - shock) vs. Schockstärke (alpha)")
plt.grid(True, alpha=0.3)
plt.show()

#Gesamthandel pro Land 
plt.figure(figsize=(10, 5))
plt.plot(df_show["alpha"], df_show["delta_trade_A"], label = "delta_A", marker="o")
plt.plot(df_show["alpha"], df_show["delta_trade_B"], label = "delta_B", marker="o")
plt.plot(df_show["alpha"], df_show["delta_trade_C"], label = "delta_c", marker="o")
plt.plot(df_show["alpha"], df_show["delta_trade_D"], label = "delta_D", marker="o")
plt.plot(df_show["alpha"], df_show["delta_trade_E"], label = "delta_E", marker="o")
plt.plot(df_show["alpha"], df_show["delta_trade_F"], label = "delta_F", marker="o")
plt.xlabel("alpha")
plt.ylabel("Δ Gesamthandel (Summe)")
plt.legend()
plt.title("Gesamthandelsdifferenz der Länder (baseline - shock) vs. Schockstärke (alpha)")
plt.grid(True, alpha=0.3)
plt.show()

#Gesamthandel pro Land ohne Schockparteien 
plt.figure(figsize=(10, 5))
plt.plot(df_show["alpha"], df_show["delta_trade_A"], label = "delta_A", marker="o")
#plt.plot(df_show["alpha"], df_show["delta_trade_B"], label = "delta_B", marker="o")
#plt.plot(df_show["alpha"], df_show["delta_trade_C"], label = "delta_c", marker="o")
plt.plot(df_show["alpha"], df_show["delta_trade_D"], label = "delta_D", marker="o")
plt.plot(df_show["alpha"], df_show["delta_trade_E"], label = "delta_E", marker="o")
plt.plot(df_show["alpha"], df_show["delta_trade_F"], label = "delta_F", marker="o")
plt.xlabel("alpha")
plt.ylabel("Δ Gesamthandel (Summe)")
plt.legend()
plt.title("Gesamthandelsdifferenz (baseline - shock) vs. Schockstärke (alpha) – ohne Schockparteien B und C")
plt.grid(True, alpha=0.3)
plt.show()

print(str(k[0,1]) + " :A - B")
print(str(k[0,2]) + " :A - C")
print(str(k[0,3]) + " :A - D")
print(str(k[0,4]) + " :A - E")
print(str(k[0,5]) + " :A - F")
print("-------------------------------------")
print(str(k[5,0]) + " :F - A")
print(str(k[5,1]) + " :F - B")
print(str(k[5,2]) + " :F - C")
print(str(k[5,3]) + " :F - D")
print(str(k[5,4]) + " :F - E")