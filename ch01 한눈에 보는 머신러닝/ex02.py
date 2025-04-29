"""
    선형회귀 모델
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# 데이터 다운로드
data_root = "https://raw.githubusercontent.com/ageron/data/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# 그래프 그리기
lifesat.plot(kind = 'scatter', grid = True,
            x = "GDP per capita (USD)", y = "Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# 선형 모델
model = KNeighborsRegressor()

# 모델 훈련
model.fit(X, y)

X_new = [[37_655.2]]
print(model.predict(X_new))
