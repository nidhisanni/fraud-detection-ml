import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

st.set_page_config(page_title='Fraud Detection System', page_icon='', layout='wide')

@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('xgb_model.joblib')
        scaler = joblib.load('scaler.joblib')
        with open('feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
        return model, scaler, feature_columns
    except FileNotFoundError as e:
        st.error(f'Model files not found! Error: {e}')
        return None, None, None

def preprocess_input(data, scaler, feature_columns):
    data['amount_scaled'] = np.log1p(data['amount'])
    data['balance_diff_orig'] = data['newbalanceOrig'] - data['oldbalanceOrg']
    data['balance_diff_dest'] = data['newbalanceDest'] - data['oldbalanceDest']
    data['balance_zero_orig'] = (data['newbalanceOrig'] == 0).astype(int)
    data['balance_zero_dest'] = (data['newbalanceDest'] == 0).astype(int)
    
    transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    for t_type in transaction_types:
        data[t_type] = (data['type'] == t_type).astype(int)
    
    features_df = data[feature_columns]
    features_scaled = scaler.transform(features_df)
    return features_scaled

def main():
    st.title(' Fraud Detection System')
    st.markdown('### Detect fraudulent transactions using machine learning')
    
    model, scaler, feature_columns = load_model_artifacts()
    
    if model is None:
        st.error('Please run the model saving cell in your notebook first!')
        return
    
    st.sidebar.header(' Model Information')
    st.sidebar.info('**Model**: XGBoost Classifier\n**ROC-AUC Score**: 99.93%\n**Features**: 17 engineered features\n**Dataset**: 6M+ transactions')
    
    st.header(' Transaction Details')
    
    col1, col2 = st.columns(2)
    
    with col1:
        step = st.number_input('Time Step (hours)', min_value=1, max_value=744, value=1)
        transaction_type = st.selectbox('Transaction Type', ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
        amount = st.number_input('Transaction Amount', min_value=0.0, value=1000.0, step=100.0)
        oldbalance_orig = st.number_input('Sender Original Balance', min_value=0.0, value=5000.0)
        newbalance_orig = st.number_input('Sender New Balance', min_value=0.0, value=4000.0)
    
    with col2:
        oldbalance_dest = st.number_input('Receiver Original Balance', min_value=0.0, value=0.0)
        newbalance_dest = st.number_input('Receiver New Balance', min_value=0.0, value=1000.0)
        orig_is_merchant = st.checkbox('Sender is Merchant')
        dest_is_merchant = st.checkbox('Receiver is Merchant')
    
    if st.button(' Analyze Transaction', type='primary'):
        input_data = pd.DataFrame({
            'step': [step],
            'type': [transaction_type],
            'amount': [amount],
            'oldbalanceOrg': [oldbalance_orig],
            'newbalanceOrig': [newbalance_orig],
            'oldbalanceDest': [oldbalance_dest],
            'newbalanceDest': [newbalance_dest],
            'orig_is_merchant': [int(orig_is_merchant)],
            'dest_is_merchant': [int(dest_is_merchant)]
        })
        
        try:
            processed_features = preprocess_input(input_data, scaler, feature_columns)
            fraud_probability = model.predict_proba(processed_features)[0][1]
            is_fraud = model.predict(processed_features)[0]
            
            st.header(' Analysis Results')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if is_fraud:
                    st.error(' FRAUD DETECTED')
                else:
                    st.success(' LEGITIMATE')
                st.metric('Fraud Probability', f'{fraud_probability:.1%}')
            
            with col2:
                st.metric('Transaction Amount', f'')
                st.metric('Balance Change', f'')
            
            with col3:
                risk_level = 'HIGH' if fraud_probability > 0.7 else 'MEDIUM' if fraud_probability > 0.3 else 'LOW'
                st.metric('Risk Level', risk_level)
                st.metric('Transaction Type', transaction_type)
            
            fig = go.Figure(go.Indicator(
                mode='gauge+number',
                value=fraud_probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': 'Fraud Probability (%)'},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': 'darkblue'},
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgreen'},
                        {'range': [30, 70], 'color': 'yellow'},
                        {'range': [70, 100], 'color': 'red'}
                    ]
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader(' Risk Analysis')
            
            risk_factors = []
            if fraud_probability > 0.5:
                if transaction_type in ['TRANSFER', 'CASH_OUT']:
                    risk_factors.append('Transaction type commonly used for fraud')
                if amount > 100000:
                    risk_factors.append('High transaction amount')
                if newbalance_orig == 0:
                    risk_factors.append('Sender account drained to zero')
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(f' {factor}')
            else:
                st.info(' No significant risk factors detected')
                
        except Exception as e:
            st.error(f'Error during prediction: {str(e)}')
    
    st.header(' Model Performance')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('ROC-AUC Score', '99.93%')
    with col2:
        st.metric('Precision', 'High')
    with col3:
        st.metric('Recall', 'High')

if __name__ == '__main__':
    main()
