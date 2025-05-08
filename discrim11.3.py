import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib.patches import Ellipse

# 1) Dữ liệu từ Table 11.8 (30 non-carriers, 45 obligatory carriers)
normals = np.array([
    [-0.0056, -0.1657],
    [-0.1698, -0.1585],
    [-0.3469, -0.1879],
    [-0.0894,  0.0064],
    [-0.1679,  0.0713],
    [-0.0836,  0.0106],
    [-0.1979, -0.0005],
    [-0.0762,  0.0392],
    [-0.1913, -0.2123],
    [-0.1092, -0.1190],
    [-0.5268, -0.4773],
    [-0.0842,  0.0248],
    [-0.0225, -0.0580],
    [ 0.0084,  0.0782],
    [-0.1827, -0.1138],
    [ 0.1237,  0.2140],
    [-0.4702, -0.3099],
    [-0.1519, -0.0686],
    [ 0.0006, -0.1153],
    [-0.2015, -0.0498],
    [-0.1932, -0.2293],
    [ 0.1507,  0.0933],
    [-0.1259, -0.0669],
    [-0.1551, -0.1232],
    [-0.1952, -0.1007],
    [ 0.0291,  0.0442],
    [-0.2228, -0.1710],
    [-0.0997, -0.0733],
    [-0.1972, -0.0607],
    [-0.0867, -0.0560]
])

carriers = np.array([
    [-0.3478,  0.1151],
    [-0.3618, -0.2008],
    [-0.4986, -0.0860],
    [-0.5015, -0.2984],
    [-0.1326,  0.0997],
    [-0.6911, -0.3390],
    [-0.3608,  0.1237],
    [-0.4535, -0.1682],
    [-0.3479, -0.1721],
    [-0.3539,  0.0722],
    [-0.4719, -0.1079],
    [-0.3610, -0.0399],
    [-0.3226,  0.1670],
    [-0.4319, -0.0687],
    [-0.2734, -0.0020],
    [-0.5573,  0.0548],
    [-0.3755, -0.1865],
    [-0.4950, -0.0153],
    [-0.5107, -0.2483],
    [-0.1652,  0.2132],
    [-0.2447, -0.0407],
    [-0.4232, -0.0998],
    [-0.2375,  0.2876],
    [-0.2205,  0.0046],
    [-0.2154, -0.0219],
    [-0.3447,  0.0097],
    [-0.2540, -0.0573],
    [-0.3778, -0.2682],
    [-0.4046, -0.1162],
    [-0.0639,  0.1569],
    [-0.3351, -0.1368],
    [-0.0149,  0.1539],
    [-0.0132,  0.1400],
    [-0.1740, -0.0776],
    [-0.1416,  0.1642],
    [-0.1508,  0.1137],
    [-0.0964,  0.0531],
    [-0.2642,  0.0867],
    [-0.0234,  0.0804],
    [-0.3352,  0.0875],
    [-0.1878,  0.2510],
    [-0.1744,  0.1892],
    [-0.4055, -0.2418],
    [-0.2444,  0.1614],
    [-0.4784,  0.0282]
])

# 2) Tính sample means và pooled covariance
x1_bar = normals.mean(axis=0)
x2_bar = carriers.mean(axis=0)

n1, n2 = len(normals), len(carriers)
S1 = np.cov(normals, rowvar=False)
S2 = np.cov(carriers, rowvar=False)
S_pooled = ((n1 - 1)*S1 + (n2 - 1)*S2) / (n1 + n2 - 2)

print(n1, n2)
print(x1_bar)
print(x2_bar)
print(S1) 
print(S2)
print(S_pooled)
# Tính ma trận nghịch đảo của S_pooled
S_pooled_inv = np.linalg.inv(S_pooled)
print("Ma trận nghịch đảo của S_pooled:")
print(S_pooled_inv)
# Tính hiệu vector trung bình của 2 nhóm
print("Hiệu vector trung bình (x1_bar - x2_bar):")
print(x1_bar - x2_bar)
# Tính (x1_bar - x2_bar)^T * S_pooled_inv
print("Tích (x1_bar - x2_bar)^T * S_pooled_inv:")
result = np.dot((x1_bar - x2_bar).T, S_pooled_inv)
print(result)
# Tính tích vô hướng của result với x1_bar và x2_bar
print("Tích vô hướng của result với x1_bar:")
print(np.dot(result, x1_bar))
print("Tích vô hướng của result với x2_bar:")
print(np.dot(result, x2_bar))

# 3) Hàm vẽ ellipse contour p
def plot_cov_ellipse(mean, cov, p, ax, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    chi2_val = chi2.ppf(p, df=2)
    widths = np.sqrt(vals * chi2_val)
    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    ell = Ellipse(
        xy=mean,
        width=2*widths[0],
        height=2*widths[1],
        angle=angle,
        facecolor='none',  # Không tô
        **kwargs
    )
    ax.add_patch(ell)


# 4) Vẽ scatter + ellipse 50% & 95%
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(normals[:,0], normals[:,1], c='black', marker='o', label='Normals (π₁)')
ax.scatter(carriers[:,0], carriers[:,1], facecolors='none', edgecolors='black', marker='o', label='Carriers (π₂)')

ax.scatter(-0.21, -0.44, c='green', marker='*', s=100, label='Điểm (-0.21, -0.44)')

for mean, col in [(x1_bar,'blue'), (x2_bar,'red')]:
    plot_cov_ellipse(mean, S_pooled, 0.50, ax, edgecolor=col, ls='--', lw=1.5)
    plot_cov_ellipse(mean, S_pooled, 0.95, ax, edgecolor=col, ls='-',  lw=2.0)

ax.scatter(*x1_bar, c='blue', marker='x', s=80, label='Mean π₁')
ax.scatter(*x2_bar, c='red',  marker='x', s=80, label='Mean π₂')

ax.set_xlabel(r'$x_1=\log_{10}(\mathrm{AHF\ activity})$')
ax.set_ylabel(r'$x_2=\log_{10}(\mathrm{AHF\ like\ antigen})$')
ax.legend(loc='upper left')
ax.set_title('Hemophilia Data: Scatter & 50%/95% Ellipse Contours')
ax.grid(True)
plt.tight_layout()
plt.show()

