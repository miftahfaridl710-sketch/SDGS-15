import streamlit as st
import runpy
import types
import sys
from pathlib import Path
import inspect

st.set_page_config(page_title="SDGs 15 — Analisis (Auto)", layout="wide")
st.title("SDGs 15 — Analisis (Auto-converted)")
st.markdown("A best-effort Streamlit wrapper generated from the uploaded notebook. If something fails, open `/mnt/data/sdgs_15_converted.py` and adjust variable names or add functions as needed.")

converted = Path("sdgs_15_converted.py")
if not converted.exists():
    st.error("Converted script not found at /mnt/data/sdgs_15_converted.py. Please upload the notebook first.")
    st.stop()

st.sidebar.header("Execution controls")
run_button = st.sidebar.button("Run converted notebook code now (may take long)")
st.sidebar.markdown("**Note:** This will execute all code extracted from the notebook. Make sure required packages are installed (see `requirements.txt`).")

if run_button:
    st.info("Executing converted script... output and variables will be inspected.")
    try:
        # Execute the converted script in its own namespace
        ns = {}
        with open(converted, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, str(converted), "exec"), ns)
        st.success("Execution finished. Inspecting namespace for common objects...")
    except Exception as e:
        st.exception(e)
        st.stop()

    # Heuristics to find useful objects
    df_candidates = {k:v for k,v in ns.items() if 'DataFrame' in str(type(v)) or (hasattr(v, 'head') and hasattr(v, 'shape'))}
    model_candidates = {k:v for k,v in ns.items() if 'xgboost' in str(type(v)).lower() or 'sklearn' in str(type(v)).lower() or 'model' in k.lower()}
    fig_candidates = {k:v for k,v in ns.items() if "Figure" in str(type(v)) or hasattr(v, 'savefig') or 'matplotlib' in str(type(v)).lower()}

    if df_candidates:
        st.subheader("Detected DataFrames")
        for name, df in df_candidates.items():
            try:
                st.write(f"**{name}** — shape: {getattr(df,'shape',None)}")
                st.dataframe(df.head(100))
            except Exception as e:
                st.write(f"Could not display DataFrame {name}: {e}")

    if model_candidates:
        st.subheader("Detected Model Objects")
        for name, m in model_candidates.items():
            st.write(f"**{name}** — type: {type(m)}")
            # attempt to show attributes
            attrs = [a for a in dir(m) if not a.startswith("_")][:20]
            st.write("Attributes:", attrs)

    if fig_candidates:
        st.subheader("Detected Matplotlib-like Figures")
        import matplotlib.pyplot as plt
        for name, fig in fig_candidates.items():
            st.write(f"**{name}** — type: {type(fig)}")
            try:
                # If it's a Figure object, display directly; else, try to call plt.figure() then show
                if "Figure" in str(type(fig)):
                    st.pyplot(fig)
                else:
                    # fallback: try to call if it's a plotting function
                    st.write("Attempting to render by calling object...")
                    try:
                        fig_result = fig()
                        if hasattr(fig_result, "savefig"):
                            st.pyplot(fig_result)
                    except Exception:
                        st.write("Could not render figure directly.")
            except Exception as e:
                st.write(f"Failed to display figure {name}: {e}")

    st.success("Inspection complete. If visuals are missing, open the converted script to locate plotting code and adapt variable names.")
else:
    st.info("Press the button in the sidebar to execute the converted notebook code and show results.\n\n**Caveat:** This app runs code from your notebook; ensure you trust the notebook and have installed required packages from `requirements.txt`.")
    st.markdown("---")
    st.markdown("### Files generated for deployment")
    st.write("- `/mnt/data/sdgs_15_converted.py` — the converted python script (from your notebook).")
    st.write("- `/mnt/data/requirements.txt` — suggested packages to install before running the app.")
    st.write("You can download them from the file browser or the notebook environment and push to GitHub for Streamlit Cloud deployment.")

