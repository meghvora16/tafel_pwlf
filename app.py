import streamlit as st
import pandas as pd
import numpy as np
import pwlf
import matplotlib.pyplot as plt

st.title("Automated Tafel Plot & Corrosion Parameters")

uploaded_file = st.file_uploader("Upload your CSV/Excel data", type=["csv", "xlsx"])
if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    st.write("Data preview:")
    st.write(df.head())
    potential_col = st.selectbox("Potential column", df.columns)
    current_col = st.selectbox("Current column", df.columns)
    
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
    
    st.markdown("### Piecewise Linear Fit (automatic for Tafel regions)")
    segments = st.slider("Choose number of segments", min_value=2, max_value=5, value=5)
    
    # Automated segmentation/fitting
    my_pwlf = pwlf.PiecewiseLinFit(potential, log_current)
    breakpoints = my_pwlf.fit(segments)
    slopes = my_pwlf.slopes
    intercepts = my_pwlf.intercepts
    
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
    
    results_df = pd.DataFrame({
        'Segment': np.arange(1, segments+1),
        'Slope (dec/V)': slopes,
        'Intercept': intercepts,
        'Start Potential': breakpoints[:-1],
        'End Potential': breakpoints[1:]
    })
    st.write("Fit results (no human bias!):")
    st.write(results_df)
    
    # --------- Tafel parameter extraction ---------
    # Get cathodic branch: Most negative slope
    idx_cathodic = np.argmin(slopes)
    # Get anodic branch: Most positive slope
    idx_anodic = np.argmax(slopes)
    
    slope_cathodic = slopes[idx_cathodic]
    intercept_cathodic = intercepts[idx_cathodic]
    slope_anodic = slopes[idx_anodic]
    intercept_anodic = intercepts[idx_anodic]
    
    # Calculate Ecorr (intersection point)
    # log(i)_anodic = log(i)_cathodic at Ecorr
    # intercept_a + slope_a*E = intercept_c + slope_c*E
    Ecorr = (intercept_cathodic - intercept_anodic) / (slope_anodic - slope_cathodic)
    log_icorr = intercept_anodic + slope_anodic * Ecorr
    icorr_A = 10**log_icorr
    icorr_uAcm2 = icorr_A * 1e6 # 1 cm^2 area
    
    st.markdown("### Corrosion Parameters from Non-biased Tafel Fit")
    st.write(f"**Ecorr:** {Ecorr:.3f} V")
    st.write(f"**log(Icorr):** {log_icorr:.2f}")
    st.write(f"**Icorr:** {icorr_A:.2e} A/cm² or {icorr_uAcm2:.3f} μA/cm² (area = 1 cm²)")
    
    st.markdown("### Input Metal Properties (default: Iron)")
    EW = st.number_input("Equivalent Weight (g/equiv)", value=27.92)
    density = st.number_input("Density (g/cm³)", value=7.87)
    
    # Calculation
    # K = 3.27e-3 for mm/year, icorr in μA/cm²
    K = 3.27e-3
    corrosion_rate_mmperyr = (K * icorr_uAcm2 * EW) / density
    
    st.write(f"**Corrosion Rate:** {corrosion_rate_mmperyr:.4f} mm/year (for area = 1 cm²)")

    st.markdown("""
    **Formula used:**  
    Corrosion rate [mm/year] = `(K × icorr × EW) / density`  
    Where K = 3.27e-3, icorr in μA/cm²
    """)
    
    st.markdown(f"""
    | Parameter      | Value             |
    |----------------|-------------------|
    | Ecorr (V)      | {Ecorr:.3f}       |
    | Icorr (A/cm²)  | {icorr_A:.2e}     |
    | Icorr (μA/cm²) | {icorr_uAcm2:.3f} |
    | Corr. Rate     | {corrosion_rate_mmperyr:.4f} mm/year |
    """)

    st.info("If using a different metal, update EW and density above.")

st.markdown("""
#### Notes
- Fit selects Tafel regions by extreme slopes automatically (no human bias)
- Calculations valid for 1 cm² working electrode
""")
