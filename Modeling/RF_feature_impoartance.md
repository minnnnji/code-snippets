### RF 변수 중요도 

```py
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = RandomForestClassifier(random_state=42, min_samples_split=5)
model.fit(X, y)
y_pred = model.predict(X)

# 변수 중요도 추출

importance_df = pd.DataFrame({'Feature': X.columns, 
                              'Importance' : model.feature_importances_})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 시각화
plt.figure(figsize=(5, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title('Feature Importances')
plt.show()
```