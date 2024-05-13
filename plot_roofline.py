import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

device_config = {"RTXA5500": (34.10/2, 0.768), 
                 "A100":     (155.92/2, 19, 1.5)}

def roofline_gflops(arithmetic_intensity, device_name):
    peak_flops, peak_sram_bw, peak_hbm_bw = device_config[device_name]
    return np.minimum(peak_flops, arithmetic_intensity * peak_sram_bw), np.minimum(peak_flops, arithmetic_intensity * peak_hbm_bw)

# max_ell = 0
arithmetic_intensities_ell_0 = [3.216182572614108, 13.421757770632368, 22.684222957014644, 35.81620026651437, 51.02846475303144]
tflops_ell_0 = np.array([0.2135307080722496, 21.37959384364892, 77.21768937217183, 283.23091208408914, 860.5104463001686]) * 1e-3

# max_ell = 1  
arithmetic_intensities_ell_1 = [8.724259166248116, 26.173599443832916, 42.748071505148474, 65.06677052803771, 89.3056774145511]  
tflops_ell_1 = np.array([1.6807429206779663, 83.211707974611, 282.4851033529378, 989.9553554793678, 3357.0163419412816]) * 1e-3

# max_ell = 2
arithmetic_intensities_ell_2 = [16.40463384360778, 43.71854461483806, 68.53882103224845, 100.09014618046113, 132.11940035439898]
tflops_ell_2 = np.array([5.091672449814504, 217.15741441001725, 707.0359149291564, 2263.4454714763137, 5754.370706473561]) * 1e-3

# max_ell = 3
arithmetic_intensities_ell_3 = [26.325181758096498, 65.81041043579435, 99.67226232187673, 140.206931536806, 178.94496410218355]
tflops_ell_3 = np.array([10.634564947835502, 398.97848895258437, 1229.2835714883222, 3527.43144650245, 7661.100817686353]) * 1e-3

# max_ell = 4
arithmetic_intensities_ell_4 = [38.28053774799936, 91.78513161473771, 135.21374393730522, 183.69445662841454, 227.66858313031238]
tflops_ell_4 = np.array([18.528934863781945, 643.6990976144284, 1821.288644932002, 4473.656011127326, 8892.088195744553]) * 1e-3

# max_ell = 5
arithmetic_intensities_ell_5 = [52.372318339100346, 121.23631889673409, 174.3175145045837, 230.12158302676536, 278.16931816550215]
tflops_ell_5 = np.array([30.253499180935446, 892.8946798103932, 2366.894094617055, 4958.605190902782, 7975.694118976293]) * 1e-3

# max_ell = 6
arithmetic_intensities_ell_6 = [68.47332523591517, 153.84935585074427, 216.5886931396865, 278.887771400702, 329.81426706269326]
tflops_ell_6 = np.array([46.93594855232594, 1081.1277417669055, 2632.825246312816, 4486.355921563159, 6963.49126023697]) * 1e-3
cm = mpl.colormaps['YlGnBu'](np.linspace(0, 1, 7))

# Plotting
plt.figure(figsize=(10, 6))

# Plot arithmetic_intensities vs flops
plt.plot(arithmetic_intensities_ell_0, tflops_ell_0, 'o--', color=cm[0], label="L=1")
labels = ['C=1', 'C=32', 'C=64', 'C=128', 'C=256']
for i, label in enumerate(labels):
        plt.annotate(label, (arithmetic_intensities_ell_0[i], tflops_ell_0[i]), textcoords="offset points", xytext=(0,10), ha='center')


plt.plot(arithmetic_intensities_ell_1, tflops_ell_1, 'o--', color=cm[1], label="L=2")
plt.plot(arithmetic_intensities_ell_2, tflops_ell_2, 'o--', color=cm[2], label="L=3")
plt.plot(arithmetic_intensities_ell_3, tflops_ell_3, 'o--', color=cm[3], label="L=4")
plt.plot(arithmetic_intensities_ell_4, tflops_ell_4, 'o--', color=cm[4], label="L=5")
plt.plot(arithmetic_intensities_ell_5, tflops_ell_5, 'o--', color=cm[5], label="L=6")
plt.plot(arithmetic_intensities_ell_6, tflops_ell_6, 'o--', color=cm[6], label="L=7")

arithmetic_intensities_roofline = np.logspace(0, 3, 100)

roofline_sram_gflops, roofline_hbm_gflops = roofline_gflops(arithmetic_intensities_roofline, "A100")
# Plot roofline_gflops as a straight line
plt.plot(arithmetic_intensities_roofline[:60],
        roofline_sram_gflops[:60],
        color='red')
plt.plot(arithmetic_intensities_roofline,
        roofline_hbm_gflops,
        color='black')
plt.text(2, 60, "SRAM 19 TB/s", rotation=13, fontsize='large', va='center', ha='center')
plt.text(2, 5, "HBM 1.5 TB/s", rotation=13, fontsize='large', va='center', ha='center')
plt.text(175, 145, "77.96 TFLOPS/s", rotation=0, fontsize='large', va='center', ha='center')
plt.text(20, 175, "A100", rotation=0, fontsize='xx-large', va='center', ha='center')


plt.xlabel('Operational Intensity: FLOPS/Bytes (log scale)')
plt.ylabel('TFLOPS/s (log scale)')
plt.title('Roofline for einsum(fi,gj,fgh,ijk->hk) $\in$ $(l_{1},l_{2},l_{out})$ \n CHANNELx(0e + 1o + 2e + ....) \otimes 0e + 1o + 2e')
plt.legend()
plt.xlim(10**0, 400)
plt.ylim(10**-4, 10**3)
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.savefig("tp_roofline.png")