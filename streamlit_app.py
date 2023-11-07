import os
import xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

plt.rc('font', family='Malgun Gothic')

from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import missingno as msno
import streamlit as st
import streamlit_toggle as tog

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import missingno as msno

from itertools import combinations
from tqdm import tqdm
import xgboost



st.set_page_config(layout="wide")




def AL_RandomForest(trainX, trainY, testX, testY):
    rf_clf = RandomForestRegressor(n_estimators=500)
    rf_clf.fit(trainX, trainY)
    #rf_clf.fit(trainX, trainY)

    # relation_square = rf_clf.score(trainX, trainY)
    # print('RandomForest 학습 결정계수 : ', relation_square)

    y_pred1 = rf_clf.predict(trainX)
    y_pred2 = rf_clf.predict(testX)

    return y_pred2

def AL_GradientBoosting(trainX, trainY, testX, testY):
    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns = pd.RangeIndex(testX.shape[1])

    gbr_model = GradientBoostingRegressor(n_estimators=500, learning_rate = 0.05)
    gbr_model.fit(trainX, trainY)

    y_pred = gbr_model.predict(trainX)
    y_pred2 = gbr_model.predict(testX)

    return y_pred2

def AL_SVR(trainX, trainY, testX, testY):

    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns = pd.RangeIndex(testX.shape[1])

    sv_regressor = SVR(kernel='linear', C=3, epsilon=0.03)
    sv_regressor.fit(trainX, trainY)

    y_pred = sv_regressor.predict(trainX)
    y_pred2 = sv_regressor.predict(testX)

    return y_pred2

# 비교값 표현 차트
def basic_chart(obsY, preY, str_part):
    if str_part == 'line':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(obsY)), obsY, '-', label="Original Y")
        ax.plot(range(len(preY)), preY, '-x', label="predict Y")

    plt.legend(loc='upper right')

def Performance_index(obs, pre, mod_str):
    if mod_str == 'R2':
        pf_index = r2_score(obs, pre)
    elif mod_str == 'RMSE':
        s1 = mean_squared_error(obs, pre)
        pf_index = np.sqrt(s1)
    elif mod_str == 'MSE':
        pf_index = mean_squared_error(obs, pre)
    elif mod_str == 'MAE':
        pf_index = mean_absolute_error(obs, pre)

    return pf_index

def AL_XGBoosting(trainX, trainY, testX, testY):
    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns = pd.RangeIndex(testX.shape[1])

    gbr_model = xgboost.XGBRegressor(n_estimators=500, learning_rate = 0.05, max_depth=7)
    gbr_model.fit(trainX, trainY)

    y_pred = gbr_model.predict(trainX)
    y_pred2 = gbr_model.predict(testX)

    return y_pred2


# 데이터 전처리
def preprocess_data(data):
    data = data.drop(["Name"], axis=1)  # 불필요한 열 제거
    data["Age"] = data["Age"].fillna(data["Age"].mean())  # 결측치 처리
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})  # 범주형 데이터 숫자로 변환
    return data

# 다중 파일 업로드 및 데이터프레임 반환 함수
def get_multifile_uploaded_df():
    uploaded_files = st.file_uploader("여러 개의 CSV 파일을 선택하세요.", type=["csv"], accept_multiple_files=True)
    dfs = []
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            dfs.append(df)
        except Exception as e:
            st.warning(f"{uploaded_file.name} 파일을 읽을 수 없습니다.")
            st.write(e)
    return pd.concat(dfs, ignore_index=True)

# 개별 그래프
def basic_chart_set(df1, str_part):
    fig, ax = plt.subplots()

    if str_part == 'boxplot':
        ax.boxplot(df1)
        ax.set_xlabel('Data Type')
        ax.set_ylabel('Value')
    elif str_part == 'line':
        ax.plot(range(len(df1)), df1, '-', label="Original Y")
    elif str_part == 'distplot':
        dist = sns.distplot(df1, kde=False)  # kde=False를 넣어보자
        dist.set_xlabel("Chlorophyll")
        dist.set_ylabel("개체수")
    elif str_part == 'heatmap':
        y_corrmat = df1.corr()
        sns.heatmap(y_corrmat, vmax=.8, annot=True, fmt='.1f', square=True)
    elif str_part == 'countbar':
        df1.value_counts(ascending=True).plot(kind='bar')

    plt.legend(loc='upper right')
    plt.show()


def set_date(rawframe):
    rawframe['set_date'] = pd.to_datetime(rawframe[rawframe.columns[0]], format='%Y%m%d %H:%M', errors='coerce')
    rawframe = rawframe.drop(rawframe.columns[0], axis=1)
    return rawframe


# 각 지점의 알맞는 학습 배치 데이터 생성 함수
def buildDataSet(timeSeries, seqLength, target):
    xdata = []
    ydata = []
    for i in range(0, len(timeSeries) - seqLength - target):
        tx = timeSeries[i:i + seqLength]
        # print(tx)
        ty = timeSeries.iloc[i + seqLength + target, -1]
        # print(ty)
        xdata.append(tx)
        ydata.append(ty)
    return np.array(xdata), np.array(ydata)


# 각 지점의 알맞는 학습 배치 데이터 생성 함수
def buildDataSet2(timeSeries, seqLength, target):
    xdata = pd.DataFrame()
    # ydata = pd.DataFrame()
    for s in range(0, seqLength):
        tx = timeSeries.iloc[(seqLength - s):(len(timeSeries) - s - target), :].reset_index()
        xdata = pd.concat([xdata, tx], axis=1, ignore_index=False)

    xdata = xdata.drop(['index', xdata.columns[-1]], axis='columns')
    ydata = pd.DataFrame(timeSeries.iloc[seqLength + target:, -1])
    return xdata, ydata


def main():
    # 헤더
    st.container()  # 헤더 컨테이너
    st.image("header_img.jpg", use_column_width=True)
    # st.markdown('<img src="header_img.jpg" width="100%" style="pointer-events:none">', unsafe_allow_html=True)

    st.markdown("# 망간 수질 데이터 예측")


    # 본문
    st.container()  # 본문 컨테이너
    # 토글 버튼으로 데이터 로드 여부 결정
    load_data = st.checkbox("기본 예제 데이터(주암댐)", value=False)

    # 데이터 로드 여부에 따라 실행
    if not load_data:
        file = st.file_uploader("Upload CSV", type=["csv"])
    else:
        # 데이터 로드하지 않을 경우 안내 메시지 출력
        # st.write("기본 예제 데이터(주암댐)")
        file = "./dataset_cleaned.csv"


    # 데이터 불러오기
    if file is not None:
        data = pd.read_csv(file)

        # 데이터 열 선택
        columns_list = st.multiselect('데이터 열 선택', data.columns)
        st.dataframe(data.iloc[0:1][columns_list])

        # 날짜 데이터 선택
        set_date = st.selectbox('날짜 데이터 선택', columns_list)

        print(type(set_date))
        print(set_date)

        if not set_date == None:

            # 날짜 전처리
            data = data.rename(columns={set_date: "set_date"})
            data['set_date'] = pd.to_datetime(data['set_date'])
            data['month'] = data['set_date'].dt.month.astype(float)
            st.dataframe(data)

            # 종속 데이터 선택
            y_var = st.selectbox('종속 변수 선택', columns_list)

            # data = preprocess_data(data)
            data = data.dropna()

            scaler = MinMaxScaler()
            _train_data = data.drop(['set_date'], axis=1)
            scaler.fit(_train_data)
            train_data_ = scaler.transform(_train_data)
            train_data = pd.DataFrame(data, columns=data.columns)
            train_data[_train_data.columns] = train_data_

            st.dataframe(train_data)

            # set_date 삭제
            train_data.drop(['set_date'], axis=1, inplace=True)

            # 환경 변수
            model_list = ["GBM", "RF", "XGB"]  # 분석 모델 리스트 설정 : LSTM, GBM, RF, SVR
            performance_list = ["RMSE", "R2", "MSE", "MAE"]  # 분석 성능평가 리스트 설정 : RMSE, R2, MSE, MAE
            var_list = [columns_list]  # 최저기온=TMn_HS_old
            temp_list_name = ["Mn"]

            trainSize_rate = 0.75  # 학습 및 예측 셋 구분


            # 초매개변수 설정
            n_estimators = st.sidebar.slider('n_estimators', min_value=10, max_value=1000, value=100, step=10)
            max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=100, value=10, step=1)
            min_samples_split = st.sidebar.slider('min_samples_split', min_value=2, max_value=20, value=2, step=1)

            selected_model = st.sidebar.radio('Select an option:', model_list)

            # 데이터 분할
            X = train_data.drop([y_var], axis=1)
            y = train_data[y_var]
            trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

            print(len(trainX))

            # 모델 학습
            if selected_model == "GBM":
                predict = AL_GradientBoosting(trainX, trainY, testX, testY)
            elif selected_model == "RF":
                predict = AL_RandomForest(trainX, trainY, testX, testY)
            elif selected_model == "SVR":
                predict = AL_SVR(trainX, trainY, testX, testY)
            elif selected_model == "XGB":
                predict = AL_XGBoosting(trainX, trainY, testX, testY)

            # 예측
            yhat = predict
            actual = testY

            # 정확도 출력
            # 성과지표 표출 부분 : 적용 항목은 confing > performance_list[] 참조
            st.markdown(f'### {selected_model}')
            for pi in performance_list:
                rmse = Performance_index(actual, yhat, pi)
                # print(temp_list_name[count] + " " + md + ' 예측 ' + pi + ' : ', rmse)
                st.write(f'{pi} : {rmse}')

            # 예측값과 실측값 비교 그래프
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(np.arange(len(actual)), actual, 'go-', label='Actual')
            ax.plot(np.arange(len(yhat)), yhat, 'ro-', label='Predicted')
            ax.set_xlabel('Samples')
            ax.set_ylabel('Mn(%)')
            ax.legend()
            st.pyplot(fig)


if __name__ == '__main__':
    main()
