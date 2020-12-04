# Ensemble Learning

Ensemble learning이란 이름은 거창하게 들리겠지만, 간단히 말해서 하나의 모델을 사용하는 게 아닌 여러 모델을 잘 조합하여서 하나의 모델만 사용할 때보다 성능을 높이자는 것이다. 즉, 어떠한 task를 수행하기 위한 모델을 학습하고 사용하는 데에 있어서, 하나의 모델을 사용하는 것이 아닌 여러 모델을 사용하자는 취지에서 나온 이론이며, 이러한 여러 모델을 어떻게 조합할지에 대해서 다루고자 한다. 

## 1. Simple Enbsemble methods

### 1.1 Max voting

Max voting은 classification 문제에서 일반적으로 사용되는 정말 간단한 방법 중 하나로서, 여러 모델은 각자의 output을 생성하고, 이러한 output 중 다수결로 하여 가장 많이 생성된 output을 최종 결과로서 사용하자는 것이다. 즉, 가장 많이 나오는 output을 최종 결과로서 사용하자는 것으로, 모든 과정이 서로 독립적인 관계로 고려된다. 

그리고 이는 sklearn에서 다음과 같이 module을 제공한다.

```python
from sklearn.ensemble import VotingClassifier

model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)

model = VotingClassifier(estimators=[('LR', model1), ('DT', model2)], voting='hard')
model.fit(x_train,y_train)
model.score(x_test,y_test)
```

### 1.2 Averaging

Averaging은 주로 regression 문제에서 사용될 수 있으며, 여러 모델의 예측 결과의 평균을 최종 결과로서 사용하자는 것이다.(당연히 classification 문제에서는 일반적으로 사용되기 힘들다. classification의 예측 결과는 index 개념이기 때문이다.) 

```python
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)

pred1 = model1.predict_proba(x_test)
pred2 = model2.predict_proba(x_test)

pred = (pred1 + pred2)/2
```

### 1.3 Weighted Average

이는 앞선 Averaging에 대해서 확장된 것으로 모델들의 평균을 구하되, 각각의 모델이 예측한 결과 자체에 중요도에 따른 차별을 주자는 것이다. 즉, 여러 모델들의 예측 결과에 각각의 모델이 갖는 중용도를 weight로서 곱한 값들의 평균으로 최종 결과를 정하자는 것이다. 

```python
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)

pred1 = model1.predict_proba(x_test)
pred2 = model2.predict_proba(x_test)

pred = (0.2*pred1 + 0.8*pred2)/2
```

## 2. Advanced Ensemble methods

### 2.1 Stacking

Stacking은 기존의 여러 모델에 대한 결과를 바탕으로 새로운 모델을 만들어서 최종 결론을 내는 방법이다. 예를 들어서, 다음과 같이 두가지의 classfier model이 있다. 

* DecisionTree
* KNN

1. 각각의 classifier는 train dataset을 10등분하여, 1 ~ 9 등분에 대해서 학습을 시켜서 결과를 예측하고, 마지막 10 등분에 대하여서는 학습을 진행하지 않고 결과를 도출한다. 이렇게 되면 각각의 classifier에 대해 train dataset의 크기만큼의 예측결과가 만들어진다. 

2. 그리고 각각의 classifier는 test dataset에 대해서 결과를 예측한다. (물론, test dataset이므로 학습은 진행하지 않는다.)

3. 앞선 1번에서의 예측결과를 train dataset의 

### 2.1 Blending


## References

* https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/