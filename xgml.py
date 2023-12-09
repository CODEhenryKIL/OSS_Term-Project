import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from src.plot_utils import draw_pitch

#이벤트 데이터 불러오기
competitions = [x for x in os.listdir('data/refined_events') if not x.startswith('.')]
match_events_list = []

for competition_name in competitions:
    match_df = pd.read_csv(f'data/refined_events/{competition_name}/matches.csv', index_col=0, encoding='utf-8-sig')

    for match_id in tqdm(match_df.index, desc=f"{competition_name + ' ':10s}"):
        match_events = pd.read_pickle(f'data/refined_events/{competition_name}/{match_id}.pkl')
        match_events['competition_name'] = competition_name
        match_events_list.append(match_events)

events = pd.concat(match_events_list, ignore_index=True)

#슈팅 데이터 필터링
shots = events[
    (events['event_type'] == 'Shot') | (events['sub_event_type'].isin(['Free kick shot', 'Penalty']))
].reset_index(drop=True)

#슈팅 위치, 각도, 거리 계산
shot_features = pd.DataFrame(index=shots.index)
shot_features['x'] = 104 - shots['start_x']
shot_features['y'] = shots['start_y'] - 34
shot_features['distance'] = shot_features[['x', 'y']].apply(np.linalg.norm, axis=1)

x = shot_features['x']
y = shot_features['y']
goal_width = 7.32
angles = np.arctan((goal_width * x) / (x ** 2 + y ** 2 - (goal_width / 2) ** 2)) * 180 / np.pi
shot_features['angle'] = np.where(angles >= 0, angles, angles + 180)

shot_features['freekick'] = (shots['event_type'] == 'Free kick').astype(int)
shot_features['header'] = shots['tags'].apply(lambda x: 'Head/body' in x).astype(int)
shot_features['goal'] = shots['tags'].apply(lambda x: 'Goal' in x).astype(int)

#슈팅 데이터 연결 및 저장
shots = pd.concat([shots[['competition_name'] + shots.columns[:-5].tolist()], shot_features], axis=1)
shots.to_pickle('data/shots.pkl')

# 모델 학습을 위한 데이터 세팅
shots = pd.read_pickle('data/shots.pkl')
feature_cols = ['x', 'y', 'distance', 'angle', 'freekick', 'header', 'goal']
shot_features = shots[feature_cols]

#xG 모델 학습
#슈팅 각도별 골 확률
bins = np.arange(0, 180, 3)
angle_cats = pd.cut(shot_features['angle'], bins, right=False)
probs_per_angle = shot_features.groupby(angle_cats)['goal'].mean()

#슈팅 각도와 득점 여부 간 로지스틱 회귀 모델 학습
xg_angle_fit = smf.glm(formula='goal ~ angle', data=shot_features, family=sm.families.Binomial()).fit()
xg_angle_fit.params

#슈팅 각도별 득점 확률 시각화
plt.figure(figsize=(10, 5))
plt.scatter(bins[:-1] + 1.5, probs_per_angle.values, s=50)

a, b = xg_angle_fit.params
x = np.arange(0, 150)
y = 1 / (1 + np.exp(-a - b * x))
plt.plot(x, y, c='black')

plt.xlabel("Shot angle (degrees)")
plt.ylabel('Probability of a shot scored')
plt.show()

#거리별 골 확률
bins = np.arange(0, 50, 1) + 1
dist_cats = pd.cut(shot_features['distance'], bins, right=False)
probs_per_dist = shot_features.groupby(dist_cats)['goal'].mean()


#슈팅 거리와 득점 사이의 로지스틱 회귀 모델 학습
xg_dist_fit = smf.glm(formula='goal ~ distance', data=shot_features, family=sm.families.Binomial()).fit()
xg_dist_fit.params

# 슈팅 거리 구간별 득점 확률
plt.figure(figsize=(10, 5))
plt.scatter(bins[:-1], probs_per_dist.values, s=50)

a, b = xg_dist_fit.params
x = np.arange(0, 50, 1)
y = 1 / (1 + np.exp(-a - b * x))
plt.plot(x, y, c='black')

plt.xlabel("Shot distance (m)")
plt.ylabel('Probability of a shot scored')
plt.show()

# 로지스틱 회귀 모델 학습
formula = 'goal ~ x + y + distance + angle + freekick + header'
xg_fit = smf.glm(formula=formula, data=shot_features, family=sm.families.Binomial()).fit()
print(xg_fit.summary())

#학습된 모델 기반 슈팅별 xG 계산
sum = xg_fit.params[0]
for i, c in enumerate(shot_features.columns[:6]):
    sum += xg_fit.params[i+1] * shot_features[c]

shot_features['xg'] = 1 / (1 + np.exp(-sum))

shots = pd.concat([shots, shot_features[['xg']]], axis=1)
shots.to_pickle('data/shots.pkl')

#구역별 xG 산출
feature_cols
shot_grid = []
m = 50
n = 64

for x_idx in range(m):
    for y_idx in range(n):
        x = (x_idx + 0.5) * 52 / m
        y = (y_idx + 0.5) * 68 / n - 34
        shot_grid.append({'x_idx': x_idx, 'y_idx': y_idx, 'x': x, 'y': y})

shot_grid = pd.DataFrame(shot_grid)
shot_grid['distance'] = shot_grid[['x', 'y']].apply(np.linalg.norm, axis=1)

x = shot_grid['x']
y = shot_grid['y']
goal_width = 7.32
angles = np.arctan((goal_width * x) / (x ** 2 + y ** 2 - (goal_width / 2) ** 2))
shot_grid['angle'] = np.where(angles >= 0, angles, angles + np.pi) * 180 / np.pi

shot_grid['freekick'] = 0
shot_grid['header'] = 0

sum = xg_fit.params[0]
for i, c in enumerate(feature_cols[:-1]):
    sum += xg_fit.params[i+1] * shot_grid[c]

shot_grid['xg'] = 1 / (1 + np.exp(-sum))

xg_heatmap = np.zeros((m, n))

for i in shot_grid.index:
    x_idx = shot_grid.at[i, 'x_idx']
    y_idx = shot_grid.at[i, 'y_idx']
    xg_heatmap[x_idx, y_idx] = shot_grid.at[i, 'xg']

    # x = int(shot_grid.at[i, 'x'] - 0.5)
    # y = int(shot_grid.at[i, 'y'] + 33.5)
    # xg_heatmap[x, y] = shot_grid.at[i, 'xg']

#xG 히트맵 시각화
fig, ax = draw_pitch(pitch='white', line='black', orientation='v', view='h')

img = ax.imshow(xg_heatmap, extent=[0, 68, 52, 104], vmin=0, vmax=0.3, cmap='jet', alpha=0.7)
fig.colorbar(img, ax=ax)
plt.title('Goal probability (xG)', fontdict={'size': 25})

plt.show()
pd.DataFrame(xg_heatmap).to_csv('data/xg_heatmap.csv', index=False)