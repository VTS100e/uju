import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VECM, VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
import io
import zipfile
import traceback

def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

def df_to_csv_bytes(df, filename="data.csv"):
    try:
        return df.to_csv(index=True).encode('utf-8')
    except Exception as e:
        st.error(f"Error converting DataFrame {filename} to CSV: {e}")
        return None

# ==============================================================
# Analysis Function
# ==============================================================
# ---  ADF Test Function---
def run_adf_test(series, name=''):
    try:
        result = adfuller(series.dropna())
        output = f'--- ADF Test Results for {name} ---\n'
        output += f'ADF Statistic: {result[0]:.4f}\n'
        output += f'p-value: {result[1]:.4f}\n'
        output += 'Critical Values:\n'
        for key, value in result[4].items():
            output += f'\t{key}: {value:.4f}\n'
        if result[1] <= 0.05:
            output += "Conclusion: Reject H0 - Series is likely Stationary\n"
        else:
            output += "Conclusion: Fail to Reject H0 - Series is likely Non-Stationary\n"
        return output
    except Exception as e:
        return f"Error during ADF test for {name}: {e}\n"

# ---  Lag VAR Function ---
def find_optimal_lags(data, maxlags=10, criterion='aic'):
    model_var = VAR(data)
    criterion_lower = criterion.lower()
    try:
        lag_selection_results = model_var.select_order(maxlags=maxlags)
        selected_lags = getattr(lag_selection_results, criterion_lower)
        summary = lag_selection_results.summary().as_text()
        return int(selected_lags), summary, criterion
    except Exception as e:
        st.warning(f"Error during lag selection using {criterion}: {e}. Defaulting VAR lag to 2.")
        return 2, f"Error during lag selection: {e}.", criterion

# --- Johansen Function Test ---
def run_johansen_test(data, vecm_lag_order, sig_level=0.05):
    det_order_johansen = 0
    if not isinstance(vecm_lag_order, int) or vecm_lag_order < 0:
        st.warning(f"Invalid VECM lag order ({vecm_lag_order}) for Johansen test. Setting p=0.")
        vecm_lag_order = 0
    crit_idx = {0.10: 0, 0.05: 1, 0.01: 2}.get(sig_level, 1)
    try:
        if data.empty: raise ValueError("Input data for Johansen test is empty.")

        johansen_test = coint_johansen(data, det_order=det_order_johansen, k_ar_diff=vecm_lag_order)
        r = 0
        lr1 = johansen_test.lr1; cvt = johansen_test.cvt; trace_crit_val = cvt[:, crit_idx]
        for i in range(len(lr1)):
            if lr1[i] > trace_crit_val[i]: r = i + 1
            else: break
        output = f"--- Johansen Cointegration Test Results (Significance Level: {sig_level*100}%) ---\n"
        output += f"Data shape: {data.shape}, VECM Lag (p): {vecm_lag_order}, Det. Order: {det_order_johansen}\n"
        output += f"Trace Statistic: {johansen_test.lr1}\nTrace Critical Values (90%, 95%, 99%):\n{johansen_test.cvt}\n\n"
        output += f"Max-Eigen Statistic: {johansen_test.lr2}\nMax-Eigen Critical Values (90%, 95%, 99%):\n{johansen_test.cvm}\n\n"
        output += f"Determined Cointegrating Rank (r) based on Trace at {sig_level*100}% level: {r}\n"
        return int(r), output, sig_level
    except ValueError as ve:
         st.error(f"Error during Johansen test setup: {ve}")
         return 0, f"Error during Johansen test setup: {ve}. Setting rank (r) to 0.", sig_level
    except Exception as e:
        st.error(f"Error during Johansen test execution: {e}\nTraceback:\n{traceback.format_exc()}")
        return 0, f"Error during Johansen test execution: {e}. Setting rank (r) to 0.", sig_level

# ---  VECM Estimation Function ---
def estimate_vecm(data, vecm_lag_order, r, selected_lags, deterministic_choice='co'):
    try:
        rank_r = int(r)
    except (ValueError, TypeError):
        st.error(f"Invalid rank '{r}' provided to estimate_vecm. Setting rank to 0.")
        rank_r = 0
    if rank_r <= 0:
        return None, None, f"No cointegration (r={rank_r}) or Johansen failed. VECM not estimated.", deterministic_choice

    if not isinstance(selected_lags, int):
        st.error(f"Invalid selected_lags '{selected_lags}' for VECM residual calculation. Cannot proceed.")
        return None, None, f"VECM estimation failed due to invalid selected_lags.", deterministic_choice

    try:
        if data.empty: raise ValueError("Input data for VECM estimation is empty.")
        if not isinstance(vecm_lag_order, int) or vecm_lag_order < 0:
            raise ValueError(f"VECM lag order must be a non-negative integer: {vecm_lag_order}")

        model_vecm = VECM(data, k_ar_diff=vecm_lag_order, coint_rank=rank_r, deterministic=deterministic_choice)
        vecm_results = model_vecm.fit()
        summary_text = vecm_results.summary().as_text()

        residuals_np = vecm_results.resid
        residual_start_index = selected_lags
        if len(data.index) > residual_start_index:
             residuals = pd.DataFrame(residuals_np, index=data.index[residual_start_index:], columns=data.columns)
        else:
             st.warning(f"Not enough data points ({len(data.index)}) to align residuals after VAR lags ({selected_lags}). Residual index may not match original dates.")
             residual_index = pd.RangeIndex(start=0, stop=len(residuals_np), step=1)
             residuals = pd.DataFrame(residuals_np, index=residual_index, columns=data.columns)
        return vecm_results, residuals, summary_text, deterministic_choice

    except ValueError as ve:
        st.error(f"Error during VECM estimation setup: {ve}")
        return None, None, f"VECM estimation failed: {ve}", deterministic_choice
    except Exception as e:
        st.error(f"Error during VECM estimation execution with deterministic='{deterministic_choice}', rank={rank_r}: {e}\nTraceback:\n{traceback.format_exc()}")
        return None, None, f"Error during VECM estimation execution: {e}", deterministic_choice


# --- VECM Diagnostics Residuals Function ---
def run_vecm_diagnostics(residuals):
    diagnostics_text = {}
    if residuals is None or not isinstance(residuals, pd.DataFrame) or residuals.empty:
        return {}, "Residuals not available for VECM diagnostics."
    n_obs = len(residuals)
    lags_lb = 0
    if n_obs // 4 > 0: lags_lb = min(10, n_obs // 4)
    elif n_obs > 1: lags_lb = min(10, n_obs - 1)

    lags_arch = 0
    if n_obs // 4 > 1: lags_arch = min(5, n_obs // 4 - 1)
    elif n_obs > 2: lags_arch = min(5, n_obs - 2)

    if lags_lb <= 0:
        return {}, "Not enough observations for reliable Ljung-Box test in VECM diagnostics."

    log_message = ""

    for col in residuals.columns:
        resid_series = residuals[col].dropna()
        output = f"--- VECM Residual Diagnostics for: {col} ---\n"
        min_points_for_tests = 5 
        if resid_series.empty or len(resid_series) < min_points_for_tests:
            diagnostics_text[col] = output + f"No valid data or too few points ({len(resid_series)}) for diagnostics."
            log_message += f"Skipped VECM diagnostics for {col} (too few points).\n"
            continue

        # Ljung-Box Test
        try:
            current_lags_lb = min(lags_lb, len(resid_series) // 2 - 1)
            if current_lags_lb > 0:
                lb_test = acorr_ljungbox(resid_series, lags=[current_lags_lb], return_df=True, boxpierce=False)
                p_val = lb_test.loc[current_lags_lb, 'lb_pvalue']
                output += f"\nLjung-Box Test (Autocorr, lags={current_lags_lb}):\n{lb_test.to_string()}\n"
                output += "Conclusion: " + ("No sig. autocorrelation (p > 0.05).\n" if p_val > 0.05 else "Significant autocorrelation detected (p <= 0.05).\n")
            else:
                output += f"\nLjung-Box Test: Skipped (too few data points after lag consideration: {len(resid_series)}).\n"
        except Exception as e_lb:
            output += f"\nError Ljung-Box for {col}: {e_lb}\n"
            log_message += f"Error Ljung-Box for {col}: {e_lb}\n"

        # ARCH-LM Test
        try:
            if lags_arch > 0:
                current_lags_arch = min(lags_arch, len(resid_series) // 2 - 1)
                if current_lags_arch > 0:
                    # <nobs - 1
                    if current_lags_arch < len(resid_series):
                         arch_test_result = het_arch(resid_series, nlags=current_lags_arch)
                         output += f"\nARCH-LM Test (Heterosk, lags={current_lags_arch}):\n"
                         output += f"  LM Stat: {arch_test_result[0]:.4f}, p-value: {arch_test_result[1]:.4f}\n"
                         output += "Conclusion: " + ("Reject H0 - ARCH effects likely present (p <= 0.05).\n" if arch_test_result[1] <= 0.05 else "Fail to Reject H0 - No sig. ARCH effects (p > 0.05).\n")
                    else:
                        output += f"\nARCH-LM Test: Skipped (lags >= # points: {current_lags_arch} >= {len(resid_series)}).\n"
                else:
                    output += f"\nARCH-LM Test: Skipped (too few data points after lag consideration: {len(resid_series)}).\n"
            else:
                output += f"\nARCH-LM Test: Skipped (not enough initial observations: {n_obs}).\n"
        except Exception as e_arch:
             # LinAlgError
            if "numpy.linalg.LinAlgError" in str(type(e_arch)):
                 output += f"\nARCH-LM Test: Skipped for {col} due to numerical instability (possibly singular matrix).\n"
                 log_message += f"Skipped ARCH-LM for {col} due to LinAlgError.\n"
            else:
                 output += f"\nError ARCH-LM for {col}: {e_arch}\n"
                 log_message += f"Error ARCH-LM for {col}: {e_arch}\n"

        # Jarque-Bera Test
        try:
            if len(resid_series) > 2:
                jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(resid_series)
                output += f"\nJarque-Bera Test (Normality):\n"
                output += f"  Stat: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}\n  Skew: {skew:.4f}, Kurtosis: {kurtosis:.4f}\n"
                output += "Conclusion: " + ("Reject H0 - Residuals likely not normal (p <= 0.05).\n" if jb_pvalue <= 0.05 else "Fail to Reject H0 - Residuals likely normal (p > 0.05).\n")
            else:
                output += "\nJarque-Bera Test: Skipped (too few data points).\n"
        except Exception as e_jb:
            output += f"\nError Jarque-Bera for {col}: {e_jb}\n"
            log_message += f"Error Jarque-Bera for {col}: {e_jb}\n"

        diagnostics_text[col] = output
    return diagnostics_text, log_message

# ---  GARCH Analysis Function ---
def run_garch_analysis(residuals, garch_model_type="Standard GARCH", garch_p=1, garch_q=1, garch_dist='normal'):
    garch_summaries = {}
    garch_plots = {}
    garch_diagnostics_text = {}

    # Input Check
    if residuals is None or not isinstance(residuals, pd.DataFrame) or residuals.empty:
        return {}, {}, {}, "Residuals not available for GARCH analysis."

    output_log = f"--- GARCH Settings Used: Model={garch_model_type}, p={garch_p}, q={garch_q}, dist='{garch_dist}' ---"
    o_lag = 0  # o lag asymmetry (Standard GARCH=0)
    if garch_model_type == "GJR-GARCH":
        output_log += " (o=1 implicit)"; o_lag = 1
    elif garch_model_type == "EGARCH":
        output_log += " (o=1 implicit)"; o_lag = 1

    # Lag for GARCH
    n_obs = len(residuals)
    lags_lb_garch = 0; lags_arch_garch = 0
    if n_obs > 1: lags_lb_garch = min(10, max(1, n_obs // 4)) # Lag for Ljung-Box (min 1)
    if n_obs > 2: lags_arch_garch = min(5, max(1, n_obs // 4 - 1)) # Lag for ARCH test (min 1)
    if lags_lb_garch <= 0: output_log += "\nWarning: Not enough observations for reliable GARCH Ljung-Box diagnostics."

    # Loop for columns in residuals
    for col in residuals.columns:
        resid_series = residuals[col].dropna()
        diag_text = f"--- GARCH Diagnostics for: {col} ({garch_model_type}) ---\n"

        # --- GARCH Requariments Check ---
        min_obs_needed = max(garch_p, o_lag, garch_q) + 5
        if resid_series.empty or len(resid_series) < min_obs_needed:
            skip_msg = f"Skipped: Too few valid points ({len(resid_series)}), needed ~{min_obs_needed}."
            output_log += f"\nSkipping GARCH for {col}: {skip_msg}"; garch_summaries[col] = skip_msg; garch_diagnostics_text[col] = diag_text + skip_msg; continue

        series_var = np.var(resid_series)
        if series_var < 1e-15:
            skip_msg = "Skipped: Near zero variance detected."
            output_log += f"\nSkipping GARCH for {col}: {skip_msg}"; garch_summaries[col] = skip_msg; garch_diagnostics_text[col] = diag_text + skip_msg; continue

        # --- Data Residual Scaling (Manual) ---
        scale = 1.0
        target_scale_std = 1.0
        if series_var > 1e-12:
             current_scale_std = np.sqrt(series_var)
             if current_scale_std > 1e-9:
                 scale = target_scale_std / current_scale_std
                 scale = max(0.001, min(1000.0, scale))
                 if abs(scale - 1.0) > 0.05:
                      output_log += f"\nNote: Rescaled '{col}' residuals by factor {scale:.3f} for GARCH stability."
        rescaled_resid_series = resid_series * scale
        # ----------------------------------------

        # --- Input Plot  ---
        fig_resid = None
        try:
            fig_resid, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            plot_index = resid_series.index if isinstance(resid_series.index, pd.DatetimeIndex) else np.arange(len(resid_series))
            axs[0].plot(plot_index, resid_series) 
            axs[0].set_title(f'VECM Residuals for {col} (Original Scale - Input)')
            axs[1].plot(plot_index, resid_series**2)
            axs[1].set_title(f'Squared VECM Residuals for {col}')
            plt.tight_layout()
            garch_plots[f'{col}_residuals_input'] = fig_resid
        except Exception as e_plot_in:
            output_log += f"\nWarning: Could not plot input residuals for {col}: {e_plot_in}"
            garch_plots[f'{col}_residuals_input'] = None
            if fig_resid: plt.close(fig_resid)

        # ---  GARCH Model ---
        garch_model_obj = None; garch_results = None; std_resid = None; model_desc = "N/A"
        try:

            if garch_model_type=="EGARCH":   model_obj=arch_model(rescaled_resid_series, vol='EGARCH', p=garch_p, o=1, q=garch_q, dist=garch_dist); model_desc=f"EGARCH(p={garch_p},o=1,q={garch_q})"
            elif ( garch_model_type=="GJR-GARCH"): model_obj=arch_model(rescaled_resid_series, vol='GARCH',  p=garch_p, o=1, q=garch_q, dist=garch_dist); model_desc=f"GJR-GARCH(p={garch_p},o=1,q={garch_q})"
            else:
                model_obj=arch_model(rescaled_resid_series, vol='GARCH',  p=garch_p, o=0, q=garch_q, dist=garch_dist); model_desc=f"Std GARCH(p={garch_p},q={garch_q})"


            garch_results = model_obj.fit(disp='off' )
            garch_summaries[col] = garch_results.summary().as_text()
            diag_text += f"\n{model_desc} Model Estimated Successfully.\n"


            std_resid = (garch_results.resid / garch_results.conditional_volatility).dropna()

            # ---  GARCH Diagnostics ( if std_resid valid) ---
            if std_resid is not None and not std_resid.empty and len(std_resid) > 2:
                fig_std_resid = None
                try:
                    fig_std_resid, axs_std = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                    plot_index_std = std_resid.index if isinstance(std_resid.index, pd.DatetimeIndex) else np.arange(len(std_resid))
                    axs_std[0].plot(plot_index_std, std_resid); axs_std[0].set_title(f'Standardized Residuals ({col} - {model_desc})')
                    axs_std[1].plot(plot_index_std, std_resid**2); axs_std[1].set_title(f'Squared Standardized Residuals ({col} - {model_desc})')
                    plt.tight_layout(); garch_plots[f'{col}_std_residuals'] = fig_std_resid
                except Exception as e_plot_std:
                    output_log += f"\nWarning: Could not plot std residuals for {col}: {e_plot_std}"
                    garch_plots[f'{col}_std_residuals'] = None
                    if fig_std_resid: plt.close(fig_std_resid)

                # 1. Ljung-Box for Standarization Residuals  (Autocorelations)
                try:
                    lags = min(lags_lb_garch, len(std_resid)//2 - 1)
                    if lags > 0:
                        lb_test = acorr_ljungbox(std_resid, lags=[lags], return_df=True, boxpierce=False)
                        p_val = lb_test.loc[lags, 'lb_pvalue']
                        diag_text += f"\nLjung-Box Test (Std.Resids, lags={lags}):\n{lb_test.to_string()}\n"
                        diag_text += "Conclusion: " + ("No significant serial correlation remains (p > 0.05).\n" if p_val > 0.05 else "Warning: Significant serial correlation may remain (p <= 0.05).\n")
                    else: diag_text += f"\nLjung-Box Test (Std.Resids): Skipped (too few points for lags).\n"
                except Exception as e: diag_text += f"\nError Ljung-Box (Std.Resids): {e}\n"

                # 2. Ljung-Box for squared Standarization Residual  (left over ARCH Effects)
                try:
                    lags = min(lags_arch_garch, len(std_resid)//2 - 1)
                    if lags > 0 and lags < len(std_resid):
                        lb_test_sq = acorr_ljungbox(std_resid**2, lags=[lags], return_df=True, boxpierce=False)
                        p_val_sq = lb_test_sq.loc[lags, 'lb_pvalue']
                        diag_text += f"\nLjung-Box ARCH Test (Sq.Std.Resids, lags={lags}):\n{lb_test_sq.to_string()}\n"
                        diag_text += "Conclusion: " + ("No significant remaining ARCH effects (p > 0.05).\n" if p_val_sq > 0.05 else "Warning: Significant ARCH effects may remain (p <= 0.05).\n")
                    else: diag_text += f"\nLjung-Box ARCH Test (Sq.Std.Resids): Skipped (too few points for lags).\n"
                except Exception as e: diag_text += f"\nError ARCH Test (Sq.Std.Resids): {e}\n"

                # 3. Normality Test on Standardized Residuals
                try:
                    jb_stat, jb_pvalue, skew, kurt = jarque_bera(std_resid)
                    diag_text += f"\nJarque-Bera Test (Std.Resids Normality vs {garch_dist}):\n"
                    diag_text += f"  Statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}\n  Skewness: {skew:.4f}, Kurtosis: {kurt:.4f}\n"
                    conclusion = f"Fail to Reject H0 - Std. Residuals consistent with assumed '{garch_dist}' distribution (p > 0.05).\n"
                    if jb_pvalue <= 0.05:
                        conclusion = f"Reject H0 - Std. Residuals likely NOT {garch_dist} distributed (p <= 0.05)."
                        if garch_dist == 'normal': conclusion += " Consider using 't' or 'skewt' distribution.\n"
                        else: conclusion += "\n"
                    diag_text += f"Conclusion: {conclusion}"
                except Exception as e: diag_text += f"\nError Jarque-Bera (Std.Resids): {e}\n"
            else:
                diag_text += "\nCould not perform GARCH diagnostics: Standardized residuals not available or too few.\n"
                # erase plot if failed
                if f'{col}_std_residuals' in garch_plots and garch_plots[f'{col}_std_residuals']:
                    plt.close(garch_plots[f'{col}_std_residuals']); garch_plots[f'{col}_std_residuals'] = None

        except Exception as e_garch:
            # Handling error
            error_msg = f"Error during GARCH ({model_desc}) estimation/diagnostics for {col}: {e_garch}"
            output_log += f"\n{error_msg}\nTraceback:\n{traceback.format_exc()}"
            garch_summaries[col] = error_msg # error message
            diag_text += f"\n{error_msg}\n"
            # erase plot if its error
            if f'{col}_residuals_input' in garch_plots and garch_plots[f'{col}_residuals_input']: plt.close(garch_plots[f'{col}_residuals_input']); garch_plots[f'{col}_residuals_input'] = None
            if f'{col}_std_residuals' in garch_plots and garch_plots[f'{col}_std_residuals']: plt.close(garch_plots[f'{col}_std_residuals']); garch_plots[f'{col}_std_residuals'] = None


        garch_diagnostics_text[col] = diag_text


    plt.close('all')
    return garch_diagnostics_text, garch_summaries, garch_plots, output_log


# ==============================================================
# Streamlit App Layout
# ==============================================================

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("VECM-GARCH Analysis (Alpha) ")
st.markdown("""This application performs VECM, IRF, and GARCH analysis as a default on uploaded time series data.
**GARCH modeling includes options for Standard GARCH, EGARCH, and GJR-GARCH.**
**You can download all text results as a ZIP file after running the analysis.**
For a detailed explanation of the analytical steps and methods used, please scroll down.""")


# --- Sidebar ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file (Index: Date)", type=["csv"])

st.sidebar.header("2. Model Parameters")
# VECM
st.sidebar.subheader("VECM Settings")
use_manual_lag = st.sidebar.checkbox("Manually select lag", value=False, key="use_manual_lag")
if use_manual_lag:
    manual_lag = st.sidebar.number_input("Manual VAR Lag (k_ar)", min_value=1, max_value=20, value=2, key="manual_lag")
    lag_criterion = "Manual"
    max_lags_var = manual_lag  # Set max_lags_var to manual_lag for consistency
    st.sidebar.markdown("*Using manually selected lag*")
else:
    lag_criterion = st.sidebar.selectbox("VAR Lag Criterion", ['AIC', 'BIC', 'HQIC', 'FPE'], index=0, key="lag_crit")
    max_lags_var = st.sidebar.slider(f"Max Lags for {lag_criterion}", 1, 20, 10, key="max_lags")
johansen_sig = st.sidebar.selectbox("Johansen Test Sig. Level", [0.10, 0.05, 0.01], format_func=lambda x: f"{int(x*100)}%", index=1, key="johan_sig")
vecm_deterministic = st.sidebar.selectbox("VECM Deterministic Term", ['co', 'ci', 'const', 'lo', 'li', 'none'], index=0, key="vecm_det")

# IRF
st.sidebar.subheader("IRF Settings")
run_irf_analysis = st.sidebar.checkbox("Run Impulse Response Function (IRF) Analysis", value=True, key="run_irf")
irf_periods = st.sidebar.number_input("IRF Periods (Steps)", min_value=5, max_value=50, value=10, key="irf_steps", disabled=not run_irf_analysis)

# GARCH
st.sidebar.subheader("GARCH Settings")
garch_model_type_selection = st.sidebar.selectbox(
    "GARCH Model Type",
    ["Standard GARCH", "EGARCH", "GJR-GARCH"], index=0, key="garch_type_select",
    help="Standard GARCH: Symmetric volatility response. EGARCH/GJR-GARCH: Allow asymmetric response to shocks (leverage effect)."
)
garch_p_order = st.sidebar.number_input("GARCH p order", min_value=0, max_value=5, value=1, key="garch_p", help="Order of ARCH/GARCH terms (past shocks/symmetric variance).")
garch_q_order = st.sidebar.number_input("GARCH q order", min_value=0, max_value=5, value=1, key="garch_q", help="Order of GARCH/Variance lags.")
garch_distribution = st.sidebar.selectbox("GARCH Distribution", ['normal', 't', 'skewt'], index=0, key="garch_dist")


# ==============================================================
# Main Analysis Area
# ==============================================================

if uploaded_file is not None:
    # This is the main try-except block for the whole application workflow
    try:
        data_initial = None
        # --- NEW, MORE ROBUST DATA LOADING LOGIC ---
        st.caption("Loading data...")
        uploaded_file.seek(0)
        
        # Step 1: Read the CSV as a plain DataFrame first.
        # This helps isolate file reading errors from date parsing errors.
        try:
            df_temp = pd.read_csv(uploaded_file, sep=None, engine='python')
            if 'Date' not in df_temp.columns:
                 # Try again with semicolon if 'Date' is not in columns
                 uploaded_file.seek(0)
                 df_temp = pd.read_csv(uploaded_file, sep=';', engine='python')
                 if 'Date' not in df_temp.columns:
                     raise ValueError("A 'Date' column was not found in the CSV file with comma or semicolon separator.")
            
            df_temp.set_index('Date', inplace=True)

        except Exception as e:
            st.error(f"Could not read the basic structure of the CSV file. Please ensure it is a valid CSV. Error: {e}")
            st.stop()
        
        # Step 2: Now, try to convert the text index into proper dates using multiple strategies.
        try:
            # Strategy 1: Standard date conversion (for YYYY-MM-DD, etc.)
            st.caption("Attempting to parse dates (standard format)...")
            df_temp.index = pd.to_datetime(df_temp.index)
            data_initial = df_temp
            st.caption("Date parsing successful (standard).")
        except (ValueError, TypeError):
            # Strategy 2: Day-first date conversion (for DD/MM/YYYY)
            st.caption("Standard date parsing failed. Trying day-first format...")
            try:
                df_temp.index = pd.to_datetime(df_temp.index, dayfirst=True)
                data_initial = df_temp
                st.caption("Date parsing successful (day-first).")
            except (ValueError, TypeError) as e:
                st.error(f"Could not convert the 'Date' column to dates using any method. Please check the date format in your file. Error: {e}")
                st.stop()
        
        # --- Continue with processing after successful data load ---
        if data_initial is None:
            st.error("Data loaded but could not be parsed into a final DataFrame.")
            st.stop()

        if not isinstance(data_initial.index, pd.DatetimeIndex):
             st.error("The 'Date' column was loaded but could not be converted to a DatetimeIndex.")
             st.stop()

        data = data_initial.sort_index()
        numeric_cols = data.select_dtypes(include=np.number).columns
        if len(numeric_cols) < data.shape[1]:
            non_numeric_cols = data.select_dtypes(exclude=np.number).columns
            st.warning(f"Ignored non-numeric columns: {list(non_numeric_cols)}. Analysis proceeds with numeric columns.")
            data = data[numeric_cols]
        if data.empty: st.error("No numeric data found or remaining after selection."); st.stop()

        data_original_shape = data.shape
        data = data.dropna()
        if data.empty:
            st.error(f"Data is empty after dropping rows with missing values (NaN). Original numeric shape was {data_original_shape}. Check data for NaNs.")
            st.stop()
        if data.shape[1] < 2:
            st.error(f"VECM requires at least two numeric variables without missing values. Found {data.shape[1]}: {list(data.columns)}")
            st.stop()

        # --- Frequency Inference ---
        st.subheader("Data Frequency Check")
        inferred_freq = None
        try:
            if data.index.is_unique:
                inferred_freq = pd.infer_freq(data.index)
                if inferred_freq:
                    st.success(f"Inferred frequency: **{inferred_freq}**")
                    try: data = data.asfreq(inferred_freq); st.caption("Frequency set.")
                    except ValueError as e_freq_set: st.warning(f"Could not set frequency '{inferred_freq}' (gaps/irregularities?). Continuing without enforcing frequency. Reason: {e_freq_set}")
                else: st.warning("Could not infer regular frequency. Continuing.")
            else: st.warning("Duplicate dates found. Cannot infer frequency.")
        except Exception as e_freq: st.warning(f"Frequency inference error: {e_freq}. Continuing.")
        st.write("")

        st.header("Uploaded Data Preview (Numeric, Non-Missing)")
        st.dataframe(data.head())
        st.caption(f"Original file shape: {data_initial.shape}, Processed shape: {data.shape}")


        # --- Run Analysis Button ---
        if st.button("Run Full Analysis"):
            if use_manual_lag:
                info_msg = (f"Running analysis: Manual Lag={manual_lag}, "
                            f"Johansen Sig={johansen_sig*100}%, VECM Det={vecm_deterministic}, "
                            f"IRF={'Yes (' + str(irf_periods) + ' steps)' if run_irf_analysis else 'No'}, "
                            f"GARCH={garch_model_type_selection}(p={garch_p_order},q={garch_q_order}), Dist={garch_distribution}")
            else:
                info_msg = (f"Running analysis: Max Lags={max_lags_var}, Crit={lag_criterion.upper()}, "
                            f"Johansen Sig={johansen_sig*100}%, VECM Det={vecm_deterministic}, "
                            f"IRF={'Yes (' + str(irf_periods) + ' steps)' if run_irf_analysis else 'No'}, "
                            f"GARCH={garch_model_type_selection}(p={garch_p_order},q={garch_q_order}), Dist={garch_distribution}")
            st.info(info_msg)

            # Initialize the result variable
            adf_results_text_dict = {}
            selected_lags = None; lag_summary = "Lag selection not performed."; used_criterion = lag_criterion
            vecm_lag_order = None; r = 0; johansen_summary = "Johansen test not performed."; used_sig = johansen_sig
            vecm_results_model = None; residuals = None; vecm_summary = "VECM not estimated."; used_det = vecm_deterministic
            vecm_diagnostics_results = {}; vecm_diagnostics_log = ""
            irf_results = None; irf_summary_text = "IRF not calculated."
            irf_vals = None; irf_stderr = None
            garch_diags_text = {}; garch_summaries = {}; garch_plots = {}; garch_log = ""
            analysis_successful = True

            try:
                with st.spinner ("Performing analysis... This may take a while."):
                    # --- 1. ADF Test ---
                    adf_all_stationary_diff = True
                    for name, series in data.items():
                        level_res = run_adf_test(series, name + ' (Level)')
                        diff_res = run_adf_test(series.diff().dropna(), name + ' (1st Difference)')
                        adf_results_text_dict[name] = level_res + "\n" + diff_res
                        if "Fail to Reject H0 - Series is likely Non-Stationary" in diff_res:
                            adf_all_stationary_diff = False
                    if not adf_all_stationary_diff:
                        st.warning("Warning: Not all series appear stationary at 1st difference (ADF p>0.05). VECM assumptions may be violated.")

                    # --- 2. VECM Estimation Steps ---
                    if use_manual_lag:
                        selected_lags = manual_lag
                        lag_summary = f"Manual lag selection: VAR lag (k_ar) = {manual_lag}"
                        used_criterion = "Manual"
                    else:
                        selected_lags, lag_summary, used_criterion = find_optimal_lags(data, maxlags=max_lags_var, criterion=lag_criterion)
                    vecm_lag_order = selected_lags - 1
                    if vecm_lag_order < 0: st.warning(f"VAR lag ({selected_lags}) => VECM lag < 0. Setting VECM p=0."); vecm_lag_order = 0
                    r, johansen_summary, used_sig = run_johansen_test(data, vecm_lag_order, sig_level=johansen_sig)
                    vecm_results_model, residuals, vecm_summary, used_det = estimate_vecm(data, vecm_lag_order, r, selected_lags, deterministic_choice=vecm_deterministic)
                    returned_values = estimate_vecm(data, vecm_lag_order, r, selected_lags, deterministic_choice=vecm_deterministic)
                    if len(returned_values) == 5: 
                        vecm_results_model, residuals, vecm_summary, vecm_message, used_det_from_func = returned_values
                        if vecm_message: 
                            vecm_summary = vecm_message 
                        used_det = used_det_from_func 
                    else: 
                        vecm_results_model, residuals, vecm_summary, used_det = returned_values

                    # --- 3. IRF Calculation ---
                    irf_results = None; irf_summary_text = "IRF not calculated."; irf_vals = None; irf_stderr = None
                    if run_irf_analysis:
                        if vecm_results_model is not None and r > 0:
                            try:
                                irf_results = vecm_results_model.irf(periods=irf_periods)
                                irf_summary_text = f"IRF calculated for {irf_periods} periods."
                                try: irf_vals = irf_results.irfs
                                except Exception as e_irf_val: st.warning(f"Could not extract IRF values: {e_irf_val}"); irf_vals = None
                                try: irf_stderr = irf_results.stderr()
                                except AttributeError: irf_stderr = None
                                except Exception as e_irf_se: st.warning(f"Could not extract IRF stderr: {e_irf_se}"); irf_stderr = None
                            except Exception as e_irf: irf_summary_text = f"Error calculating IRF: {e_irf}"; st.error(irf_summary_text + f"\n{traceback.format_exc()}"); irf_results=None; irf_vals=None; irf_stderr=None
                        elif r == 0: irf_summary_text = "IRF skipped (r=0)."
                        else: irf_summary_text = f"IRF skipped (VECM failed: {vecm_summary})"
                    else: irf_summary_text = "IRF analysis disabled."

                    # --- 4. VECM Residual Diagnostics ---
                    if residuals is not None and not residuals.empty:
                         vecm_diagnostics_results, vecm_diagnostics_log = run_vecm_diagnostics(residuals)
                         if vecm_diagnostics_log: st.info(f"VECM Diagnostics Log: {vecm_diagnostics_log}")
                    else: vecm_diagnostics_results = {}; vecm_diagnostics_log = "Skipped VECM diagnostics (no residuals)."

                    # --- 5. GARCH Modeling ---
                    run_garch_modeling = False # Flag to check if GARCH should run
                    if residuals is not None and not residuals.empty:
                        # Check for ARCH effects
                        arch_detected = False
                        if isinstance(vecm_diagnostics_results, dict):
                             for var, diag_text in vecm_diagnostics_results.items():
                                 if "ARCH effects likely present" in diag_text:
                                     arch_detected = True
                                     st.info(f"ARCH effects detected in VECM residuals for '{var}'. Proceeding with GARCH.")
                                     break
                        if arch_detected:
                            run_garch_modeling = True
                        else:
                            st.info("No significant ARCH effects found in VECM residuals (ARCH-LM p>0.05). GARCH might not be necessary, but proceeding as requested.")
                            run_garch_modeling = True
                    else:
                        garch_log = "Skipped GARCH (no VECM residuals)."

                    if run_garch_modeling:
                        garch_diags_text, garch_summaries, garch_plots, garch_log = run_garch_analysis(residuals, garch_model_type_selection, garch_p_order, garch_q_order, garch_distribution)
                    elif not garch_log:
                        garch_log = "Skipped GARCH modeling as no significant ARCH effects were found and residuals were available."


            # --- Error Handling  ---
            except Exception as analysis_error:
                 st.error(f"Error during main analysis workflow: {analysis_error}")
                 st.exception(analysis_error)
                 analysis_successful = False

            # --- Display Results ---
            if analysis_successful:
                tab_list = [
                    "📊 Data & Stationarity", "📈 VECM Estimation",
                    "📉 VECM Diagnostics", "⚡ Impulse Responses", "💹 GARCH Modeling",
                    "💡 Interpretation Guide", "💾 Download Results"
                ]
                tabs = st.tabs(tab_list)

                # --- Tab 1: Data & Stationarity ---
                with tabs[0]:
                    st.header("1. Data Overview & Stationarity")
                    st.subheader("1.1 Data Info"); buffer = io.StringIO(); data.info(buf=buffer); st.text(buffer.getvalue())
                    st.subheader("1.2 Data Plot")
                    try:
                         fig_data, ax_data = plt.subplots(figsize=(12, 5)); ax_data.plot(data.index, data); ax_data.set_title('Time Series Data (Processed)'); ax_data.legend(data.columns); st.pyplot(fig_data); plt.close(fig_data)
                    except Exception as e_plot: st.warning(f"Could not plot data: {e_plot}")
                    st.subheader("1.3 Stationarity Test (ADF)"); st.caption("_Sig. Level: 5%_")
                    if adf_results_text_dict:
                        for var_name, results in adf_results_text_dict.items(): st.text_area(f"ADF Results: {var_name}", results, height=200, key=f"adf_{var_name}")
                    else: st.warning("ADF results not available.")

                # ========================================================================
                # --- Tab 2: VECM Estimation --
                # ========================================================================
                with tabs[1]:
                    st.header("2. VECM Estimation")
                    st.subheader(f"2.1 VAR Lag Order Selection"); 
                    if used_criterion.upper() == "MANUAL":
                        st.markdown(f"**Selection Method:** `Manual`, **Selected Lag:** `{selected_lags}`");
                    else:
                        st.markdown(f"**Crit:** `{used_criterion.upper()}`, **Max Lags:** `{max_lags_var}`");
                    st.text(f"Selected VAR lag (k_ar): {selected_lags if selected_lags is not None else 'N/A'}\nImplied VECM lag (p): {vecm_lag_order if vecm_lag_order is not None else 'N/A'}");
                    with st.expander("Show Lag Selection Summary"): st.text(lag_summary if lag_summary else "N/A")
                    st.divider()
                    st.subheader(f"2.2 Johansen Cointegration Test"); st.markdown(f"**Sig. Level:** `{used_sig*100}%`"); st.text(f"Determined Rank (r): {r}");
                    with st.expander("Show Johansen Test Output"): st.text(johansen_summary if johansen_summary else "N/A")
                    st.divider()
                    st.subheader(f"2.3 VECM Results"); st.markdown(f"**Det. Term:** `{used_det}`, **Rank (r):** `{r}`, **VECM Lag (p):** `{vecm_lag_order if vecm_lag_order is not None else 'N/A'}`");

                    if vecm_results_model is not None and r > 0:
                        with st.expander("Show VECM Estimation Summary"):
                            st.text(vecm_summary if vecm_summary else "Summary not available.")

                        # --- VECM Coefficients Download Button ---
                        st.markdown("**Download VECM Coefficients:**")
                        col1, col2, col3 = st.columns(3)

                        # Alpha (Adjustment)
                        try:
                            alpha_df = pd.DataFrame(vecm_results_model.alpha, index=data.columns, columns=[f'alpha_ect{i+1}' for i in range(r)])
                            csv_alpha = df_to_csv_bytes(alpha_df, "vecm_alpha.csv")
                            if csv_alpha:
                                col1.download_button(label="📥 Alpha", data=csv_alpha, file_name="vecm_alpha.csv", mime="text/csv", key="dl_alpha")
                        except Exception as e_alpha:
                            col1.error(f"Error Alpha CSV: {e_alpha}")

                        # Beta (Cointegrating Vectors)
                        try:
                            beta_df = pd.DataFrame(vecm_results_model.beta, index=data.columns, columns=[f'beta_ect{i+1}' for i in range(r)])
                            csv_beta = df_to_csv_bytes(beta_df, "vecm_beta.csv")
                            if csv_beta:
                                col2.download_button(label="📥 Beta", data=csv_beta, file_name="vecm_beta.csv", mime="text/csv", key="dl_beta")
                        except Exception as e_beta:
                            col2.error(f"Error Beta CSV: {e_beta}")

                        # Gamma (Short-run)
                        try:

                            gamma_cols = []
                            for lag in range(1, vecm_lag_order + 1):
                                for var in data.columns:
                                    gamma_cols.append(f'Gamma.L{lag}.{var}')

                            expected_gamma_shape = (data.shape[1], data.shape[1] * vecm_lag_order)
                            if hasattr(vecm_results_model, 'gamma') and vecm_results_model.gamma.shape == expected_gamma_shape:
                                gamma_df = pd.DataFrame(vecm_results_model.gamma, index=data.columns, columns=gamma_cols[:expected_gamma_shape[1]])
                                csv_gamma = df_to_csv_bytes(gamma_df, "vecm_gamma.csv")
                                if csv_gamma:
                                    col3.download_button(label="📥 Gamma", data=csv_gamma, file_name="vecm_gamma.csv", mime="text/csv", key="dl_gamma")
                            elif not hasattr(vecm_results_model, 'gamma'):
                                 col3.warning("Gamma coefficients not found in results.")
                            else:
                                col3.warning(f"Gamma shape mismatch ({vecm_results_model.gamma.shape} vs {expected_gamma_shape}). Cannot create CSV.")

                        except Exception as e_gamma:
                            col3.error(f"Error Gamma CSV: {e_gamma}")

                    elif r == 0:
                        st.warning("VECM not estimated (r=0). No coefficients to download.")
                        if vecm_summary != f"No cointegration (r=0) or Johansen failed. VECM not estimated.": st.text(f"Details: {vecm_summary}")
                    else:
                        st.warning("VECM model not estimated. No coefficients to download.")
                        st.text(f"Reason: {vecm_summary}")


                # --- Tab 3: VECM Diagnostics ---
                with tabs[2]:
                    st.header("3. VECM Residual Diagnostics")
                    if vecm_diagnostics_log: st.info(f"Diagnostics Log: {vecm_diagnostics_log}")
                    if isinstance(vecm_diagnostics_results, dict) and vecm_diagnostics_results:
                         for var_name, results_text in vecm_diagnostics_results.items():
                             st.subheader(f"Diagnostics for '{var_name}' Residuals"); st.text_area(f"VECM Diag Output {var_name}", results_text, height=350, key=f"vecm_diag_{var_name}"); st.divider()
                    elif not vecm_diagnostics_log: st.info("No VECM diagnostic results generated.")

                # --- Tab 4: Impulse Responses ---
                with tabs[3]:
                    st.header(f"4. Impulse Response Functions (IRF)")
                    if run_irf_analysis:
                        st.markdown(f"Displaying IRFs for **{irf_periods}** periods ahead."); st.markdown("_Note: Non-orthogonalized IRFs. 95% CI shown (if available)._")
                        if irf_results is not None:
                            try: fig_irf = irf_results.plot(orth=False, signif=0.05); st.pyplot(fig_irf); plt.close(fig_irf)
                            except Exception as e_plot_irf: st.error(f"Error plotting IRF: {e_plot_irf}\n{traceback.format_exc()}")
                        else: st.warning(f"Could not display IRF plots. Reason: {irf_summary_text}")
                    else: st.info("IRF analysis disabled.")

                # --- Tab 5: GARCH Modeling ---
                with tabs[4]:
                    st.header(f"5. GARCH Modeling ({garch_model_type_selection})")
                    st.markdown(f"**Model:** `{garch_model_type_selection}`, **Order:** `p={garch_p_order}, q={garch_q_order}`" + (", o=1" if garch_model_type_selection in ["EGARCH", "GJR-GARCH"] else ""))
                    st.markdown(f"**Distribution:** `{garch_distribution}`"); st.markdown("_Applied to VECM Residuals_")
                    if garch_log: st.info(garch_log); st.divider()
                    garch_was_run = bool(garch_summaries or garch_diags_text or ("Skipped GARCH" not in garch_log and "GARCH log is empty" not in garch_log))
                    if garch_was_run and residuals is not None and not residuals.empty:
                        for col_name in data.columns:
                            st.subheader(f"GARCH Analysis for '{col_name}' Residuals")
                            plot_key_in = f'{col_name}_residuals_input'; plot_key_std = f'{col_name}_std_residuals'
                            # Plot Input
                            if plot_key_in in garch_plots and garch_plots[plot_key_in]: st.pyplot(garch_plots[plot_key_in]); plt.close(garch_plots[plot_key_in])
                            else: st.caption(f"_Input plot for {col_name} not available._")
                            # Summary
                            if col_name in garch_summaries:
                                 with st.expander(f"Show {garch_model_type_selection}(p={garch_p_order}, q={garch_q_order}) Summary ({col_name})"): st.text(garch_summaries[col_name])
                            else: st.caption(f"_Summary for {col_name} not available._")
                            # Diagnostics Text
                            if col_name in garch_diags_text:
                                 st.text_area(f"GARCH Diagnostics ({col_name})", garch_diags_text[col_name], height=400, key=f"garch_diag_text_{col_name}")
                            else: st.caption(f"_Diagnostics text for {col_name} not available._")
                            # Diagnostics Plot
                            if plot_key_std in garch_plots and garch_plots[plot_key_std]: st.pyplot(garch_plots[plot_key_std]); plt.close(garch_plots[plot_key_std])
                            else: st.caption(f"_Std. residuals plot for {col_name} not available._")
                            st.divider()
                    elif not garch_was_run: st.info("GARCH modeling was skipped or not performed.")
                    else: st.warning("Cannot display GARCH results: VECM residuals were not available.")

                # --- Tab 6: Interpretation Guide ---
                with tabs[5]:
                    st.header("6. Quick Interpretation Guide")

                    st.markdown("""
                    This guide provides practical instructions for interpreting the results of your VECM-GARCH analysis. Understanding these outputs correctly is crucial for drawing valid conclusions from your time series data.
                    """)

                    st.subheader("VECM Results Interpretation")
                    st.markdown("""
                    ### Cointegrating Relationships (Beta Coefficients)

                    **What to look for:**
                    - **Signs and magnitudes** of coefficients in the cointegrating vector (β)
                    - **Normalization** - one variable is typically normalized to 1.0
                    - **Statistical significance** of each coefficient

                    **How to interpret:**
                    - These represent the **long-run equilibrium relationship** between variables
                    - Example: If Y₁ is normalized to 1.0 and the coefficient for Y₂ is -0.5, then the long-run relationship is: Y₁ = 0.5Y₂ + ... (+ other variables and constant)
                    - The relationship can be read as: "In the long run, a 1-unit increase in Y₂ is associated with a 0.5-unit increase in Y₁"

                    ### Adjustment Coefficients (Alpha)

                    **What to look for:**
                    - **Signs and magnitudes** of adjustment coefficients (α)
                    - **Statistical significance** (P>|z|)

                    **How to interpret:**
                    - These show the **speed of adjustment** back to equilibrium after a deviation
                    - **Negative values** for the variable being normalized in β indicate proper error correction
                    - **Magnitude** indicates adjustment speed (e.g., -0.2 means 20% of disequilibrium is corrected each period)
                    - **Non-significant** coefficients suggest that variable may be weakly exogenous

                    ### Short-run Dynamics (Gamma)

                    **What to look for:**
                    - **Patterns** in the short-run coefficients across lags
                    - **Statistical significance** of coefficients

                    **How to interpret:**
                    - These capture **immediate responses** between variables
                    - Useful for understanding **short-term causality**
                    - Can reveal **feedback relationships** not visible in the long-run model
                    """)

                    st.subheader("Impulse Response Function (IRF) Interpretation")
                    st.markdown("""
                    IRFs show how variables respond over time to a shock in one variable.

                    **Key aspects to analyze:**

                    1. **Direction (sign):**
                       - **Positive response**: The variable increases following the shock
                       - **Negative response**: The variable decreases following the shock

                    2. **Magnitude:**
                       - How large is the effect? Compare across different response variables

                    3. **Persistence:**
                       - How long does the effect last?
                       - Does it die out quickly or persist for many periods?
                       - Does it return to zero (temporary effect) or stabilize at a new level (permanent effect)?

                    4. **Statistical significance:**
                       - Check if confidence intervals include zero
                       - Effects are significant when confidence intervals don't cross the zero line

                    5. **Pattern:**
                       - Monotonic (consistently increasing/decreasing)
                       - Oscillatory (alternating positive/negative)
                       - Hump-shaped (initial increase then decrease)
                    """)

                    st.subheader("GARCH Model Interpretation")
                    st.markdown("""
                    ### Preliminary Check

                    **Always check diagnostics first!** A well-specified GARCH model should show:
                    - No significant autocorrelation in standardized residuals
                    - No remaining ARCH effects in squared standardized residuals
                    - Distribution of standardized residuals consistent with the assumed distribution

                    ### Standard GARCH Parameters

                    **Constant term (omega):**
                    - Represents the **long-run average variance** when divided by (1 - sum of alpha - sum of beta)
                    - Usually small in magnitude

                    **ARCH terms (alpha):**
                    - Measure the **reaction of conditional volatility to market shocks**
                    - Higher values indicate stronger reaction to recent shocks
                    - Range: Typically 0.05 to 0.2 for financial data

                    **GARCH terms (beta):**
                    - Measure the **persistence of volatility**
                    - Higher values indicate longer-lasting effects of shocks
                    - Range: Typically 0.7 to 0.9 for financial data

                    **Volatility persistence:**
                    - Sum of alpha + beta coefficients
                    - Values close to 1 indicate high persistence
                    - Values > 1 suggest non-stationarity in variance

                    ### Asymmetric GARCH Models

                    **EGARCH specific parameters:**
                    - **alpha**: Size effect (magnitude of shocks)
                    - **gamma**: Sign effect (asymmetry/leverage)
                      - Positive gamma: Negative shocks increase volatility more than positive shocks
                      - Negative gamma: Positive shocks increase volatility more than negative shocks

                    **GJR-GARCH specific parameters:**
                    - **gamma**: Direct measure of asymmetry
                      - Positive and significant gamma indicates leverage effect
                      - Interpretation: Negative shocks of the same magnitude as positive shocks have a larger impact of (alpha + gamma) on volatility

                    ### Distribution Parameters

                    - **Student's t (nu)**: Degrees of freedom parameter; smaller values indicate heavier tails
                    - **Skewed-t (lambda)**: Skewness parameter; negative values indicate negative skew
                    """)

                    st.subheader("Integrated Analysis Approach")
                    st.markdown("""
                    For comprehensive understanding of your time series system:

                    1. **First analyze VECM results** to understand:
                       - Long-run equilibrium relationships between variables
                       - Speed of adjustment to equilibrium
                       - Short-run dynamic interactions

                    2. **Use IRF analysis** to visualize:
                       - How shocks propagate through the system
                       - Which variables have the strongest influence on others
                       - How long effects persist after shocks

                    3. **Examine GARCH results** to understand:
                       - Volatility patterns in the unexplained components (residuals)
                       - Whether volatility shows clustering behavior
                       - If negative shocks have different impacts than positive ones
                       - How quickly volatility reverts to its long-run average

                    This multi-faceted approach provides insights into both the conditional mean dynamics (VECM) and conditional variance dynamics (GARCH) of your multivariate time series system.
                    """)

                # ========================================================================
                # --- Tab 7: Download Results --
                # ========================================================================

                with tabs[6]:
                    st.header("7. Download Analysis Results")


                    zip_buffer = io.BytesIO()
                    try:
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            file_counter = 0

                            # 1. Parameter
                            params_text = f"Analysis Parameters:\n"
                            params_text += f"Data File: {uploaded_file.name if uploaded_file else 'N/A'}\n"
                            params_text += f"Processed Data Shape: {data.shape if data is not None else 'N/A'}\n"
                            if used_criterion.upper() == "MANUAL":
                                params_text += f"Lag Selection: Manual\nSelected VAR Lag (k_ar): {selected_lags}\n"
                            else:
                                params_text += f"Lag Criterion: {used_criterion.upper()}\nMax Lags Tested: {max_lags_var}\n"
                            params_text += f"Johansen Sig Level: {used_sig * 100}%\nDetermined Rank (r): {r}\n"
                            params_text += f"VECM Deterministic Term: {used_det}\nVECM Lag (p): {vecm_lag_order if vecm_lag_order is not None else 'N/A'}\n"
                            params_text += f"Run IRF Analysis: {'Yes' if run_irf_analysis else 'No'}\n"
                            if run_irf_analysis: params_text += f"IRF Periods: {irf_periods}\n"
                            params_text += f"GARCH Model Type: {garch_model_type_selection}\nGARCH p: {garch_p_order}, q: {garch_q_order}\nGARCH Distribution: {garch_distribution}\n"
                            zip_file.writestr(f"{file_counter:02d}_Parameters_Used.txt", params_text); file_counter += 1

                            # 2. ADF Test Results
                            adf_full_text = "--- ADF Test Results ---\n(Significance Level: 5%)\n\n"
                            if adf_results_text_dict:
                                for var_name, results in adf_results_text_dict.items(): adf_full_text += f"--- Variable: {var_name} ---\n{results}\n--------------------\n"
                            else: adf_full_text += "ADF results dictionary is empty.\n"
                            zip_file.writestr(f"{file_counter:02d}_ADF_Tests.txt", adf_full_text); file_counter += 1

                            # 3. VAR Lag Selection Summary
                            lag_selection_text = f"--- VAR Lag Order Selection ---\nCriterion: {used_criterion.upper()}, Max Lags: {max_lags_var}\nSelected VAR Lag (k_ar): {selected_lags if selected_lags is not None else 'N/A'}, Implied VECM Lag (p): {vecm_lag_order if vecm_lag_order is not None else 'N/A'}\n\nSummary Output:\n{lag_summary if lag_summary else 'N/A'}"
                            zip_file.writestr(f"{file_counter:02d}_VAR_Lag_Selection.txt", lag_selection_text); file_counter += 1

                            # 4. Johansen Test Output
                            zip_file.writestr(f"{file_counter:02d}_Johansen_Cointegration_Test.txt", johansen_summary if johansen_summary else "N/A"); file_counter += 1

                            # --- VECM Results Files (Summary and Coefficients) ---
                            if vecm_results_model is not None and r > 0:
                                # 5. VECM Summary
                                zip_file.writestr(f"{file_counter:02d}_VECM_Estimation_Summary.txt", vecm_summary if vecm_summary else "N/A"); file_counter += 1

                                # 6. Alpha Coefficients
                                try:
                                    alpha_df_zip = pd.DataFrame(vecm_results_model.alpha, index=data.columns, columns=[f'alpha_ect{i+1}' for i in range(r)])
                                    alpha_csv_bytes = df_to_csv_bytes(alpha_df_zip, "vecm_alpha.csv")
                                    if alpha_csv_bytes: zip_file.writestr(f"{file_counter:02d}_VECM_Alpha_Coeffs.csv", alpha_csv_bytes)
                                    else: zip_file.writestr(f"{file_counter:02d}_VECM_Alpha_Coeffs_Error.txt", "Error converting Alpha to CSV.")
                                except Exception as e_alpha_zip: zip_file.writestr(f"{file_counter:02d}_VECM_Alpha_Coeffs_Error.txt", f"Error: {e_alpha_zip}")
                                file_counter += 1

                                # 7. Beta Coefficients
                                try:
                                    beta_df_zip = pd.DataFrame(vecm_results_model.beta, index=data.columns, columns=[f'beta_ect{i+1}' for i in range(r)])
                                    beta_csv_bytes = df_to_csv_bytes(beta_df_zip, "vecm_beta.csv")
                                    if beta_csv_bytes: zip_file.writestr(f"{file_counter:02d}_VECM_Beta_Coeffs.csv", beta_csv_bytes)
                                    else: zip_file.writestr(f"{file_counter:02d}_VECM_Beta_Coeffs_Error.txt", "Error converting Beta to CSV.")
                                except Exception as e_beta_zip: zip_file.writestr(f"{file_counter:02d}_VECM_Beta_Coeffs_Error.txt", f"Error: {e_beta_zip}")
                                file_counter += 1

                                # 8. Gamma Coefficients
                                try:
                                    gamma_cols_zip = []
                                    for lag_g in range(1, vecm_lag_order + 1):
                                        for var_g in data.columns: gamma_cols_zip.append(f'Gamma.L{lag_g}.{var_g}')
                                    expected_gamma_shape_zip = (data.shape[1], data.shape[1] * vecm_lag_order)
                                    if hasattr(vecm_results_model, 'gamma') and vecm_results_model.gamma.shape == expected_gamma_shape_zip:
                                        gamma_df_zip = pd.DataFrame(vecm_results_model.gamma, index=data.columns, columns=gamma_cols_zip[:expected_gamma_shape_zip[1]])
                                        gamma_csv_bytes = df_to_csv_bytes(gamma_df_zip, "vecm_gamma.csv")
                                        if gamma_csv_bytes: zip_file.writestr(f"{file_counter:02d}_VECM_Gamma_Coeffs.csv", gamma_csv_bytes)
                                        else: zip_file.writestr(f"{file_counter:02d}_VECM_Gamma_Coeffs_Error.txt", "Error converting Gamma to CSV.")
                                    elif not hasattr(vecm_results_model, 'gamma'): zip_file.writestr(f"{file_counter:02d}_VECM_Gamma_Coeffs_Error.txt", "Gamma attribute not found.")
                                    else: zip_file.writestr(f"{file_counter:02d}_VECM_Gamma_Coeffs_Error.txt", f"Gamma shape mismatch ({vecm_results_model.gamma.shape} vs {expected_gamma_shape_zip}).")
                                except Exception as e_gamma_zip: zip_file.writestr(f"{file_counter:02d}_VECM_Gamma_Coeffs_Error.txt", f"Error: {e_gamma_zip}")
                                file_counter += 1

                                # 9. Deterministic Coefficients 
                                if hasattr(vecm_results_model, 'det_coefs') and vecm_results_model.det_coefs is not None:
                                    try:
                                        det_cols = [f'det_term_{i+1}' for i in range(vecm_results_model.det_coefs.shape[1])]
                                        if used_det == 'co': det_cols = ['const_coint']
                                        elif used_det == 'ci': det_cols = ['const_level']
                                        elif used_det == 'const': det_cols = ['const_level'] + [f'const_ect{i+1}' for i in range(r)]


                                        det_df_zip = pd.DataFrame(vecm_results_model.det_coefs, index=data.columns, columns=det_cols[:vecm_results_model.det_coefs.shape[1]])
                                        det_csv_bytes = df_to_csv_bytes(det_df_zip, "vecm_det_coeffs.csv")
                                        if det_csv_bytes: zip_file.writestr(f"{file_counter:02d}_VECM_Deterministic_Coeffs.csv", det_csv_bytes)
                                        else: zip_file.writestr(f"{file_counter:02d}_VECM_Deterministic_Coeffs_Error.txt", "Error converting Det Coeffs to CSV.")
                                    except Exception as e_det_zip: zip_file.writestr(f"{file_counter:02d}_VECM_Deterministic_Coeffs_Error.txt", f"Error: {e_det_zip}")
                                else:
                                     zip_file.writestr(f"{file_counter:02d}_VECM_Deterministic_Coeffs_NA.txt", "Deterministic coefficients not applicable or not found.")
                                file_counter += 1

                            else:
                                zip_file.writestr(f"{file_counter:02d}_VECM_Estimation_Skipped.txt", f"VECM skipped or failed (r={r}). No summary or coeffs generated."); file_counter += 1
                                file_counter += 4 


                            # 10. IRF Numeric Values
                            if run_irf_analysis and irf_vals is not None:
                                irf_num_text = f"--- IRF Values (Non-Orthogonalized) ---\nPeriods: {irf_periods}\nVariables: {list(data.columns)}\n\n(Format: Period, Impulse -> Response = Value [StdErr])\n\n"
                                neqs = irf_vals.shape[1]; var_names = list(data.columns)
                                for period in range(irf_vals.shape[0]):
                                    irf_num_text += f"--- Period {period} ---\n"
                                    for i in range(neqs):
                                        for j in range(neqs):
                                            imp_v, res_v = var_names[i], var_names[j]; val = irf_vals[period, j, i]; se_str = ""
                                            if irf_stderr is not None and irf_stderr.ndim == 3 and irf_stderr.shape == irf_vals.shape:
                                                try: se_val = irf_stderr[period, j, i]; se_str = f" [{se_val:.4f}]"
                                                except IndexError: se_str = " [SE Idx Err]"
                                                except Exception: se_str = " [SE Err]"
                                            elif irf_stderr is not None: se_str = " [SE Shape Err]"
                                            irf_num_text += f"{imp_v} -> {res_v} = {val:.4f}{se_str}\n"
                                        irf_num_text += "\n"
                                    irf_num_text += "---\n"
                                zip_file.writestr(f"{file_counter:02d}_IRF_Numeric_Values.txt", irf_num_text)
                            elif run_irf_analysis: zip_file.writestr(f"{file_counter:02d}_IRF_Numeric_Values_Skipped.txt", f"Numeric IRF N/A. Reason: {irf_summary_text}")
                            else: zip_file.writestr(f"{file_counter:02d}_IRF_Analysis_Disabled.txt", "IRF analysis disabled.")
                            file_counter += 1

                            # 11. VECM Residual Diagnostics
                            vecm_diag_full_text = "--- VECM Residual Diagnostics ---\n\n"
                            if vecm_diagnostics_log: vecm_diag_full_text += f"Log:\n{vecm_diagnostics_log}\n\n---\n\n"
                            if isinstance(vecm_diagnostics_results, dict) and vecm_diagnostics_results:
                                for var_name, results_text in vecm_diagnostics_results.items(): vecm_diag_full_text += f"--- Diag: {var_name} ---\n{results_text}\n--------------------\n"
                            elif not vecm_diagnostics_log: vecm_diag_full_text += "No diagnostics generated.\n"
                            zip_file.writestr(f"{file_counter:02d}_VECM_Residual_Diagnostics.txt", vecm_diag_full_text); file_counter += 1

                            # 12. GARCH Log
                            zip_file.writestr(f"{file_counter:02d}_GARCH_Log.txt", garch_log if garch_log else "N/A"); file_counter += 1

                            # 13. GARCH Estimation Summaries
                            garch_est_full_text = f"--- GARCH Summaries ({garch_model_type_selection}) ---\nDist: {garch_distribution}\n\n"
                            if garch_summaries:
                                for n, s in garch_summaries.items(): garch_est_full_text += f"--- Summary: {n} ---\n{s}\n--------------------\n"
                            else: garch_est_full_text += "N/A\n"
                            zip_file.writestr(f"{file_counter:02d}_GARCH_Estimation_Summaries.txt", garch_est_full_text); file_counter += 1

                            # 14. GARCH Diagnostics
                            garch_diag_full_text = f"--- GARCH Diagnostics ({garch_model_type_selection}) ---\nDist: {garch_distribution}\n\n"
                            if garch_diags_text:
                                if isinstance(garch_diags_text, dict):
                                    for n, r_garch_diag in garch_diags_text.items(): garch_diag_full_text += f"--- Diag: {n} ---\n{r_garch_diag if isinstance(r_garch_diag, str) else 'Format Error'}\n--------------------\n"
                                else: garch_diag_full_text += "Format Error (Expected Dict).\n"
                            else: garch_diag_full_text += "N/A\n"
                            zip_file.writestr(f"{file_counter:02d}_GARCH_Diagnostics.txt", garch_diag_full_text); file_counter += 1

                            # --- END OF ZIP BLOCK ---

                        zip_buffer.seek(0)
                        st.download_button(label="📥 Download All Results (.zip)", data=zip_buffer, file_name="analysis_results.zip", mime="application/zip", key="download_all_results")

                    except Exception as zip_error:
                         st.error(f"An error occurred during ZIP file creation: {zip_error}"); st.exception(zip_error)



            # --- last message ---

            st.write(f"Analysis Successful Flag: {analysis_successful}")
            st.write(f"VECM Model Object: {'Exists' if vecm_results_model else 'None'}")
            st.write(f"VECM Residuals Object: {'Exists' if residuals is not None else 'None'}, Shape: {residuals.shape if residuals is not None else 'N/A'}")


    # --- Error Handling  ---
    except Exception as e:
        st.error(f"An unexpected error occurred in the application: {e}"); st.exception(e)

# --- Early message ---
else:
    st.info("Please upload a CSV file using the sidebar to begin.")
    st.markdown("---")
    st.markdown("Please **note** that this is my first time creating this kind of application. If you find any problem, please let me know leq5dayval@gmail.com")
    st.markdown("Thank you and please do enjoy :)")
    st.markdown("---")
    st.markdown("If you want to support me:")
    st.markdown("You can donate through my paypal https://www.paypal.me/claydayval or via Trakteer http://teer.id/LEQ5dayval ")
    st.markdown("---")
    st.header("VECM-GARCH")
    st.markdown("""
                       This application performs comprehensive multivariate time series analysis using an integrated approach that combines Vector Error Correction Model (VECM) for capturing long-run equilibrium relationships and short-run dynamics, with Generalized Autoregressive Conditional Heteroskedasticity (GARCH) modeling for analyzing volatility patterns in the residuals.
                       """)

    st.subheader("Theoretical Background")
    st.markdown("""
                       **Vector Error Correction Model (VECM)** is an extension of Vector Autoregression (VAR) designed for non-stationary time series that are cointegrated. When variables share common stochastic trends (cointegration), VECM incorporates both short-term dynamics and long-term equilibrium relationships.

                       **GARCH models** capture time-varying volatility (conditional heteroskedasticity) in time series data, particularly the tendency of volatility to cluster and persist over time. These models are essential for financial time series where volatility is not constant.

                       The combined VECM-GARCH approach allows for modeling both the conditional mean (through VECM) and conditional variance (through GARCH) of multivariate time series.
                       """)

    st.subheader("Workflow")
    with st.expander ("", expanded=True):
        st.markdown("""
                           **1. Data Preparation:**
                           The application processes the user-uploaded `.csv` file, treating the first column as a date index (`index_col=0`, `parse_dates=True`). It attempts to read the file using both comma and semicolon separators for flexibility. Rows containing any missing values (`NaN`) in numeric columns are automatically removed (`dropna()`), and non-numeric columns are excluded from the analysis to ensure data quality.
                           """)

        st.markdown("""
                           **2. Stationarity Testing:**
                           The Augmented Dickey-Fuller (ADF) test is applied to both the original *levels* and *first differences* of each variable to assess stationarity. The null hypothesis of the ADF test is that the series contains a unit root (non-stationary). A significance level of 5% is used for interpretation:
                           - p-value ≤ 0.05: Reject H₀, series is likely stationary
                           - p-value > 0.05: Fail to reject H₀, series is likely non-stationary

                           For VECM, variables should typically be non-stationary in levels but stationary in first differences (I(1) processes).
                           """)

        st.markdown("""
                           **3. VAR Lag Selection:**
                           The lag structure for the underlying Vector Autoregression (VAR) model can be determined in two ways:

                           **A. Automatic Selection (Default):**
                           The optimal lag is determined using information criteria through `statsmodels.tsa.api.VAR.select_order()`. Users can select their preferred **Information Criterion**:
                           - **AIC** (Akaike Information Criterion): Tends to select more complex models
                           - **BIC** (Bayesian Information Criterion): Tends to favor parsimony
                           - **HQIC** (Hannan-Quinn Information Criterion): Intermediate between AIC and BIC
                           - **FPE** (Final Prediction Error): Focuses on forecast accuracy

                           **B. Manual Selection:**
                           Users can bypass the automatic selection and directly specify the VAR lag order (`k_ar`) based on prior knowledge or specific research requirements.

                           In both cases, the VECM lag order (`p`) is derived as `p = k_ar - 1`, where `k_ar` is the selected VAR lag order.
                           """)

        st.markdown("""
                           **4. Cointegration Testing:**
                           The Johansen cointegration test is performed using `statsmodels.tsa.vector_ar.vecm.coint_johansen()` to determine the number of cointegrating relationships (rank `r`). The test uses the Trace statistic approach, comparing test statistics against critical values at the user-selected significance level (1%, 5%, or 10%).

                           The test employs the VECM lag `p` from the previous step and assumes a deterministic order of 0 (constant in the data process). The rank `r` represents the number of independent cointegrating vectors in the system, with:
                           - r = 0: No cointegration (use VAR in differences)
                           - 0 < r < n: Partial cointegration (use VECM)
                           - r = n: All variables are stationary (use VAR in levels)
                           """)

        st.markdown("""
                           **5. VECM Estimation:**
                           If cointegration is detected (r > 0), a VECM is estimated using `statsmodels.tsa.api.VECM()`. The model incorporates:

                           - **Cointegrating rank** (`r`): Number of long-run relationships
                           - **VECM lag** (`p`): Number of lags in differenced terms
                           - **Deterministic term** specification: Controls how constants and trends enter the model
                             - 'co': Constant in cointegration
                             - 'ci': Constant in levels
                             - 'const': Constant in both
                             - 'lo': Linear trend in cointegration
                             - 'li': Linear trend in levels
                             - 'none': No deterministic terms

                           The results provide:
                           - **Alpha** (α): Adjustment coefficients showing speed of adjustment to equilibrium
                           - **Beta** (β): Cointegrating vectors representing long-run equilibrium relationships
                           - **Gamma** (Γ): Short-run coefficients capturing immediate dynamics
                           """)

        st.markdown("""
                           **6. Impulse Response Function (IRF) Analysis:**
                           If cointegration is detected and a valid VECM is estimated, Impulse Response Function analysis can be performed to examine how variables in the system respond to shocks over time.

                           The IRF is calculated using `vecm_results_model.irf(periods=irf_periods)` where:
                           - **periods**: User-specified number of time steps ahead to forecast (5-50)

                           IRF analysis provides:
                               - **Dynamic responses** of each variable to shocks in every other variable
                               - **Time path** of effects showing how shocks propagate through the system
                               - **Confidence intervals** (95% by default) to assess statistical significance

                           The application displays non-orthogonalized IRFs, which show the combined direct and indirect effects of shocks, providing a comprehensive view of system dynamics without imposing structural assumptions about contemporaneous relationships.

                           IRF plots are organized in a matrix where:
                               - Rows represent the **response variables**
                               - Columns represent the **impulse (shock) variables**
                               - Each cell shows how a one standard deviation shock to the column variable affects the row variable over time
                           """)
        st.markdown("""
                           **7. VECM Residual Diagnostics:**
                           The residuals from the estimated VECM are subjected to diagnostic tests to assess model adequacy:

                           - **Ljung-Box test** (`acorr_ljungbox`): Tests for autocorrelation in residuals
                             - H₀: No autocorrelation up to lag k
                             - Significant p-values indicate remaining autocorrelation

                           - **ARCH-LM test** (`het_arch`): Tests for ARCH effects (volatility clustering)
                             - H₀: No ARCH effects up to lag k
                             - Significant p-values suggest heteroskedasticity, justifying GARCH modeling

                           - **Jarque-Bera test** (`jarque_bera`): Tests for normality of residuals
                             - H₀: Residuals are normally distributed
                             - Significant p-values indicate non-normality
                           """)

        st.markdown("""
                           **8. GARCH Modeling:**
                           When ARCH effects are detected in VECM residuals, GARCH models are fitted to each residual series using `arch.arch_model()`. Users can select:

                           - **GARCH Model Type**:
                             - **Standard GARCH**: Symmetric response to shocks
                             - **EGARCH**: Exponential GARCH allowing asymmetric effects
                             - **GJR-GARCH**: Threshold GARCH capturing leverage effects

                           - **Model Orders**:
                             - **p**: Order of ARCH terms (past squared shocks)
                             - **q**: Order of GARCH terms (past conditional variances)
                             - **o**: Order of asymmetry terms (automatically set to 1 for EGARCH and GJR-GARCH)

                           - **Error Distribution**:
                             - **normal**: Gaussian distribution
                             - **t**: Student's t-distribution (heavier tails)
                             - **skewt**: Skewed t-distribution (asymmetric heavy tails)
                           """)

        st.markdown("""
                           **9. GARCH Diagnostics:**
                           The standardized residuals from each GARCH model are tested to verify model adequacy:

                           - **Ljung-Box test on standardized residuals**: Checks if serial correlation has been removed

                           - **Ljung-Box test on squared standardized residuals**: Verifies if ARCH effects have been captured

                           - **Jarque-Bera test**: Assesses if the standardized residuals follow the assumed distribution

                           A well-specified GARCH model should show no significant autocorrelation or remaining ARCH effects in the standardized residuals.
                           """)

    st.divider()
    st.subheader("User Parameter Choices")
    st.markdown("""
                       The application provides extensive analytical flexibility through user-selectable parameters in the sidebar:

                       - **VAR/VECM Parameters**: 
                         - Lag Selection: Automatic (using information criteria) or Manual (user-specified)
                         - Lag Criterion (when using automatic selection): AIC, BIC, HQIC, FPE
                         - Maximum Lags (when using automatic selection) or Manual Lag Value
                         - Johansen Significance Level
                         - VECM Deterministic Term
                       - **GARCH Parameters**: Model Type, Orders (p,q), Error Distribution

                       These options allow for tailoring the analysis to specific data characteristics and research questions.
                       """)

    st.divider()
    st.subheader("Note")
    st.info("""
                       This software serves as a comprehensive analytical tool for time series modeling. The final interpretation of results, selection of the optimal model specification, and research conclusions require expert judgment and domain knowledge. Always ensure your data meets the underlying assumptions of the models used and validate results through multiple approaches.
                       """)
