import streamlit as st
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import itertools


def main():
    html_temp="""
    <div style="background-color:cyan;padding:10px">
    <h2 style="color:black;text-align:center;">ARIMA SALES FORECASTING</h2>
    </div>
    """


    #response = requests.get('https://www.google.co.in/')
    st.markdown(html_temp, unsafe_allow_html=True)

    data = pd.read_csv(r"C:\Users\LTP-5\OneDrive\Documents\sales analysis dataset\sample dataset\car.csv", engine='python', skipfooter=3)
    pl = list(data['Sales'])
    e = pl[72:84]
    #st.write(data)

    data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m-%d')
    data.set_index(['Month'], inplace=True)



    q = d = range(0, 2)
    p = range(0, 4)

    pdq = list(itertools.product(p, d, q))

    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    #st.write('Examples of parameter combinations for Seasonal ARIMA...')
    #st.write('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    #st.write('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    #st.write('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    #st.write('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    train_data = data['1960-01-01':'1965-12-01']
    test_data = data['1966-01-01':'1968-09-01']

    #st.write(train_data)
    #st.write(test_data)

    warnings.filterwarnings("ignore")
    AIC = []
    SARIMAX_model = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(train_data,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()

                print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
                AIC.append(results.aic)
                SARIMAX_model.append([param, param_seasonal])
            except:
                continue

    mod = sm.tsa.statespace.SARIMAX(train_data,
                                    order=SARIMAX_model[AIC.index(min(AIC))][0],
                                    seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()

    pred0 = results.get_prediction(start='1964-01-01', dynamic=False)
    pred0_ci = pred0.conf_int()

    pred1 = results.get_prediction(start='1964-01-01', dynamic=True)
    pred1_ci = pred1.conf_int()

    pred2 = results.get_forecast('1966-12-01')
    pred2_ci = pred2.conf_int()
    p = pred2.predicted_mean['1966-01-01':'1966-12-01']
    #st.write(p)

    s = pd.DataFrame({'Predicted': p, 'Actual': e})
    st.write(s)

if __name__ == "__main__":
    main()