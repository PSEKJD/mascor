import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
save_path = os.path.join('./dataset/', '{country}/{region}/oracle_dataset_{option}_sample_{sample}/checkpoint_transformer_des_{des}_z_{z}_z_type_{z_type}'.format(country = 'France', 
                                                                                                                                              region = 'Dunkirk',
                                                                                                                                              option = 'c_fax_fix',
                                                                                                                                              sample = 50000,
                                                                                                                                              des = False,
                                                                                                                                              z = False, z_type = 'mv'))
with open(os.path.join(save_path, 'loss_{epoch}.pkl'.format(epoch = 50)), 'rb') as f:
    ST_loss = pickle.load(f)
    
save_path = os.path.join('./dataset/', '{country}/{region}/oracle_dataset_{option}_sample_{sample}/checkpoint_transformer_des_{des}_z_{z}_z_type_{z_type}'.format(country = 'France', 
                                                                                                                                              region = 'Dunkirk',
                                                                                                                                              option = 'c_fax_fix',
                                                                                                                                              sample = 50000,
                                                                                                                                              des = True,
                                                                                                                                              z = False, z_type = 'mv'))
with open(os.path.join(save_path, 'loss_{epoch}.pkl'.format(epoch = 50)), 'rb') as f:
    ST_des_loss = pickle.load(f)
    
save_path = os.path.join('./dataset/', '{country}/{region}/oracle_dataset_{option}_sample_{sample}/checkpoint_transformer_des_{des}_z_{z}_z_type_{z_type}'.format(country = 'France', 
                                                                                                                                              region = 'Dunkirk',
                                                                                                                                              option = 'c_fax_fix',
                                                                                                                                              sample = 50000,
                                                                                                                                              des = True,
                                                                                                                                              z = True, z_type = 'mv'))
with open(os.path.join(save_path, 'loss_{epoch}.pkl'.format(epoch = 50)), 'rb') as f:
    PT_des_loss = pickle.load(f)
    
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

#%%
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Font 설정 (추가)
# =========================
plt.rcParams['font.family'] = 'Arial'
#plt.rcParams['mathtext.fontset'] = 'dejavusans'  # Arial + math 호환

# 색상 정의
colors = {
    'No-token baseline': '#43A047',
    r'$D$-token only': '#1E88E5',
    r'$D{+}E$ tokens (Proposed)': '#6A1E9C'
}

# 이름과 데이터 매핑
models = {
    'No-token baseline': ST_loss,
    r'$D$-token only': ST_des_loss,
    r'$D{+}E$ tokens (Proposed)': PT_des_loss
}

metrics = ['action', 'rtg', 'ctg']

# =========================
# y-axis 수식 라벨 (추가)
# =========================
ylabels = [
    r'$J_{\pi}(\theta)$',
    r'$J_{Q_{\mathrm{RTG}}}(\theta^\prime)$',
    r'$J_{Q_{\mathrm{CTG}}}(\theta^\prime)$'
]

panel_labels = ['a', 'b', 'c']

fig, axes = plt.subplots(3, 1, figsize=(6, 12))

total_epoch = 50
max_step = 4000
step_per_epoch = max_step / total_epoch

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    for name, loss_dict in models.items():
        raw_loss = np.array(loss_dict[metric][:max_step])
        filtered_loss = np.where(raw_loss > 1.0, 0.0, raw_loss)
        smoothed = moving_average(filtered_loss, window_size=100)
        epoch_x = np.arange(len(smoothed)) / step_per_epoch

        ax.plot(epoch_x, smoothed, label=name, color=colors[name])

    # =========================
    # y-label 설정 (추가)
    # =========================
    ax.set_ylabel(ylabels[idx], fontsize=12)

    # =========================
    # 패널 라벨 (a)(b)(c) 추가
    # =========================
    ax.text(
    -0.20, 1.05, panel_labels[idx],
    transform=ax.transAxes,
    fontsize=20,
    fontweight='bold',
    va='top',
    ha='left'
)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(labelsize=13)

    # =========================
    # x축 Epoch: 마지막 plot만
    # =========================
    if idx < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Epoch', fontsize=13)

    if idx == 0:
        ax.legend(fontsize=13)

plt.tight_layout()
plt.savefig("loss-curve.pdf", dpi=400)
plt.show()
plt.close()

#%%
# 색상 정의
colors = {
    'No-token baseline': '#43A047',
    'D-token only': '#1E88E5',
    'D+E tokens (Proposed)': '#6A1E9C'
}

# 이름과 데이터 매핑
models = {
    'No-token baseline': ST_loss,
    'D-token only': ST_des_loss,
    'D+E tokens (Proposed)': PT_des_loss
}

metrics = ['action', 'rtg', 'ctg']
titles = ['Action Loss', 'RTG Loss', 'CTG Loss']

fig, axes = plt.subplots(3, 1, figsize=(6, 12))

total_epoch = 50
max_step = 4000
step_per_epoch = max_step / total_epoch

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    for name, loss_dict in models.items():
        raw_loss = np.array(loss_dict[metric][:max_step])
        filtered_loss = np.where(raw_loss > 1.0, 0.0, raw_loss)  # outlier 처리
        smoothed = moving_average(filtered_loss, window_size=100)
        epoch_x = np.arange(len(smoothed)) / step_per_epoch

        ax.plot(epoch_x, smoothed, label=name, color=colors[name])

    #ax.set_title(titles[idx], fontsize=13)
    ax.set_xlabel([])
    #ax.set_ylabel('Loss', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(labelsize=10)

    if idx == 0:
        ax.legend(fontsize=10)  # 첫 번째 subplot에만 legend 추가

ax.set_xlabel('Epoch', fontsize=11)
plt.tight_layout()
plt.savefig("loss-curve.png", dpi = 400)
plt.show()
plt.close()