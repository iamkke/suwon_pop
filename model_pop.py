from pmdarima import auto_arima
import pandas as pd

#데이터 불러오기
data=pd.read_csv('데이터 경로')

#남자/여자 분류
man=data[(data.성별=='남')]
woman=data[(data.성별=='여')]

#input dataset
man.인구수.groupby([man.나이, man.년, man.월]).sum()
woman.인구수.groupby([woman.나이, woman.년, woman.월]).sum()

#분석모델
def model_fit(train):
  model_arima=auto_arima(train,trace=True, error_action='ignore', start_p=1,start_q=1,max_p=5,max_q=5, suppress_warnings=True, stepwise=True, seasonal=True)
  preds, conf_int = model_arima.predict(n_periods=60, return_conf_int=True)
  return preds

#예측
def predict(data, age):
    result=model_fit(data[(data.나이==age)].인구수.values)
    result=result.astype('int')
    return list(result)

ages=data.나이
pop_man=[]
pop_woman=[]
for age in ages: #1세 계급별로 남자/여자 따로 모델링
  pop_man.append(predict(man, age))
  pop_woman.append(predict(woman, age))

#남자/여자 예측값 취합
result=pd.concat([pop_man,pop_woman])