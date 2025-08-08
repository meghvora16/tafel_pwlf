# Install required libraries first!
# pip install streamlit pandas matplotlib pwlf

import streamlit as st
import pandas as pd
import numpy as np
import pwlf
import matplotlib.pyplot as plt

st.title("Automated Tafel Plot Analyzer")

uploaded_file = st.file_uploader("Upload your CSV/Excel data", type=["csv", "xlsx"])
if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    # Select relevant columns
    st.write("Data preview:")
    st.write(df.head())
    potential_col = st.selectbox("Select Potential column", df.columns)
    current_col = st.selectbox("Select Current column", df.columns)
    
    # Preprocess
    potential = np.array(df[potential_col])
    current = np.array(df[current_col])
    
    # Remove zeros and negatives for log calculation
    mask = np.abs(current) > 1e-12
    potential = potential[mask]
    current = current[mask]
    
    log_current = np.log10(np.abs(current))
    
    st.write("Tafel plot (log(Current) vs Potential):")
    fig, ax = plt.subplots()
    ax.scatter(potential, log_current, s=10)
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("log(Current) (A)")
    st.pyplot(fig)
    
    st.markdown("### Piecewise Linear Fit (automatic)")
    segments = st.slider("Choose number of segments", min_value=2, max_value=5, value=2)
    
    # Automated segmentation/fitting
    my_pwlf = pwlf.PiecewiseLinFit(potential, log_current)
    breakpoints = my_pwlf.fit(segments)
    slopes = my_pwlf.slopes
    intercepts = my_pwlf.intercepts
    
    # Plot fit
    x_hat = np.linspace(potential.min(), potential.max(), 500)
    y_hat = my_pwlf.predict(x_hat)
    fig, ax = plt.subplots()
    ax.scatter(potential, log_current, s=10, label="Data")
    ax.plot(x_hat, y_hat, color='red', label='Piecewise Fit')
    for bp in breakpoints[1:-1]:
        ax.axvline(bp, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("log(Current) (A)")
    ax.legend()
    st.pyplot(fig)
    
    # Show fit results
    results_df = pd.DataFrame({
        'Segment': np.arange(1, segments+1),
        'Slope (dec/V)': slopes,
        'Intercept': intercepts,
        'Start Potential': breakpoints[:-1],
        'End Potential': breakpoints[1:]
    })
    st.write("Fit results (no human bias!):")
    st.write(results_df)
    
    # Fit quality
    residuals = log_current - my_pwlf.predict(potential)
    st.write(f"Fit RMSE: {np.sqrt(np.mean(residuals**2)):.3f}")
    st.write(f"Recommended: Compare slopes to theoretical values for mechanism insight.")

st.markdown("""
**How it works:**  
- Uses objective piecewise linear regression for region identification  
- Slope and intercept of each segment are extracted automatically  
- No manual selection or bias!
""")
