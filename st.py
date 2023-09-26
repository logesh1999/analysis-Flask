import pickle
import streamlit as st

model = pickle.load(open('model(pkl).pkl', 'rb'))

def main():

    st.title('ARIMA time series analysis')

    start_date =  st.text_input('start_date')
    end_date = st.text_input('end_date')

    if st.button('predict'):

        prediction = model.get_predict([[start_date, end_date]])
        output = round(prediction[0],2)

if __name__ == '__main__':
    main()