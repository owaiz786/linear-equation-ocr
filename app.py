import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import sympy as sp
import cv2
import re
from typing import List, Tuple, Dict, Any

# ──── Streamlit page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Equation Solver",
    page_icon="📷",
    layout="centered"
)

st.title("📷 OCR-based Linear Equation Solver")
st.markdown("Upload an image containing mathematical equations and let AI solve them for you!")

# ──── Sidebar configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Image Processing")
    blur_kernel = st.slider("Blur kernel size", 1, 9, 3, step=2)
    threshold_block_size = st.slider("Threshold block size", 3, 21, 11, step=2)
    threshold_c = st.slider("Threshold constant (C)", 0, 10, 2)
    st.subheader("Equation Detection")
    min_equation_length = st.slider("Minimum equation length", 3, 20, 5)
    st.subheader("Variables to solve for")
    available_vars = ['x', 'y', 'z', 'a', 'b', 'c', 't', 'u', 'v', 'w']
    selected_vars = st.multiselect(
        "Select variables (leave empty to auto-detect)",
        available_vars
    )

# ──── Helper functions ─────────────────────────────────────────────────────────
def preprocess_image(
    image: Image.Image, blur_size: int, block_size: int, c_value: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three thresholded variants for OCR."""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0) if blur_size > 1 else gray

    # Adaptive, binary, and inverted binary thresholds
    t_adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_value
    )
    _, t_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, t_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up
    kernel = np.ones((2, 2), np.uint8)
    return (
        cv2.morphologyEx(t_adapt, cv2.MORPH_CLOSE, kernel),
        cv2.morphologyEx(t_bin, cv2.MORPH_CLOSE, kernel),
        cv2.morphologyEx(t_inv, cv2.MORPH_CLOSE, kernel),
    )

def clean_ocr_text(text: str) -> str:
    """Normalize common OCR misreads in math contexts."""
    replacements = {
        'O': '0', 'o': '0', 'l': '1', 'I': '1', 'S': '5', '§': '5',
        '×': '*', '÷': '/', '−': '-', '–': '-', '—': '-',
        # Common OCR mistakes in mathematical context
        'ty': '+y',  # 'ty' is often a misread '+'
        'tx': '+x',  # 'tx' is often a misread '+'
        'tz': '+z',  # 'tz' is often a misread '+'
        'ta': '+a',  # 'ta' is often a misread '+'
        'tb': '+b',  # 'tb' is often a misread '+'
        'tc': '+c',  # 'tc' is often a misread '+'
    }
    
    # Apply basic replacements
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Handle case sensitivity - convert all variables to lowercase
    text = re.sub(r'\b[A-Z]\b', lambda m: m.group().lower(), text)
    
    # Fix common OCR patterns
    text = re.sub(r'(\w)t(\w)', r'\1+\2', text)  # xtY -> x+Y
    text = re.sub(r't(\w)', r'+\1', text)        # ty -> +y
    text = re.sub(r'(\w)t', r'\1+', text)        # xt -> x+
    
    # Collapse multiple zeros or ones
    text = re.sub(r'0{2,}', '0', text)
    text = re.sub(r'1{2,}', '1', text)
    
    return text

def extract_equations(text: str, min_len: int) -> List[str]:
    """Pull out any line-like segments containing '='."""
    eqs = []
    for line in text.splitlines():
        s = line.strip()
        if '=' in s and len(s) >= min_len:
            # Keep only valid characters
            s2 = re.sub(r'[^\w=+\-*/().^ ]', '', s)
            # Additional cleaning for common OCR issues
            s2 = s2.replace(' ', '')
            # Convert to lowercase for consistency
            s2 = s2.lower()
            eqs.append(s2)
    return list(dict.fromkeys(eqs))  # Deduplicate while preserving order

def detect_variables(equations: List[str]) -> List[str]:
    """Detect variables present in the equations."""
    vars_ = set(re.findall(r'\b[a-z]\b', ' '.join(equations)))
    common_vars = ['x', 'y', 'z', 'a', 'b', 'c', 't', 'u', 'v', 'w']
    ordered_vars = [v for v in common_vars if v in vars_]
    return ordered_vars + sorted(vars_ - set(ordered_vars))

def parse_equations(
    equations: List[str], variables: List[str]
) -> Tuple[List[Any], List[str]]:
    """Turn strings like '2x+3y=5' into SymPy expressions."""
    syms = {v: sp.Symbol(v) for v in variables}
    parsed, errors = [], []
    
    for i, eq in enumerate(equations, 1):
        try:
            # Additional preprocessing for common OCR issues
            eq_cleaned = eq.lower()
            
            # Handle implicit multiplication better
            eq_cleaned = re.sub(r'(\d)([a-z])', r'\1*\2', eq_cleaned)  # 2x -> 2*x
            eq_cleaned = re.sub(r'([a-z])([a-z])', r'\1*\2', eq_cleaned)  # xy -> x*y
            
            left, right = eq_cleaned.split('=', 1)
            # Create expression as left - right = 0
            expr = f"({left}) - ({right})"
            expr = expr.replace('^', '**')
            
            parsed_expr = sp.sympify(expr, locals=syms)
            parsed.append(parsed_expr)
            
        except Exception as e:
            errors.append(f"Eq{i}: '{eq}' → {e}")
    
    return parsed, errors

# ──── Main UI ──────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload image (PNG/JPG/BMP/TIFF)", 
    type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

if uploaded:
    img = Image.open(uploaded)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📤 Original")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("🔧 Preprocessed")
        methods = ["Adaptive", "Binary", "Binary Inverted"]
        tabs = st.tabs(methods)
        outs = preprocess_image(img, blur_kernel, threshold_block_size, threshold_c)
        for tab, out in zip(tabs, outs):
            with tab:
                st.image(out, use_container_width=True)

    # OCR on each, pick best by simple score
    best_text, best_config = "", ""
    results = []
    configs = [
        "--oem 3 --psm 6", "--oem 3 --psm 7",
        "--oem 3 --psm 8", "--oem 3 --psm 13"
    ]
    for method_name, processed_image in zip(methods, outs):
        for config in configs:
            raw_text = pytesseract.image_to_string(processed_image, config=config)
            cleaned_text = clean_ocr_text(raw_text)
            score = cleaned_text.count('=') * 10 + sum(cleaned_text.count(v) for v in 'xyzabc') * 2
            results.append((f"{method_name}|{config}", cleaned_text, score))
            if score > 0 and len(cleaned_text) > len(best_text):
                best_text, best_config = cleaned_text, f"{method_name} + {config}"

    st.subheader("🔍 All OCR passes")
    for label, text, score in results:
        st.write(f"**{label}** (score {score}) → `{text.strip()}`")

    if not best_text.strip():
        st.error("❌ No text extracted. Try different settings or a clearer image.")
        st.stop()

    st.success(f"✅ Chosen OCR: {best_config}")
    st.subheader("📝 Extracted Text")
    st.text(best_text)

    equations = extract_equations(best_text, min_equation_length)
    if not equations:
        st.warning("⚠️ No equations found. Check extracted text above.")
        st.stop()

    st.subheader("🧮 Equations")
    for i, equation in enumerate(equations, 1):
        st.write(f"{i}. `{equation}`")

    vars_to_use = selected_vars or detect_variables(equations)
    if not vars_to_use:
        st.warning("⚠️ No variables selected or detected.")
        st.stop()

    st.info(f"Solving for: {', '.join(vars_to_use)}")
    parsed_equations, parse_errors = parse_equations(equations, vars_to_use)
    
    if parse_errors:
        st.warning("Parse errors:")
        for error in parse_errors:
            st.write(f"• {error}")
    
    if not parsed_equations:
        st.error("❌ Nothing to solve after parsing.")
        st.stop()

    st.subheader("📐 Parsed (as SymPy =0)")
    for parsed_eq in parsed_equations:
        st.latex(f"{sp.latex(parsed_eq)} = 0")

    with st.spinner("🧠 Solving..."):
        symbols_list = [sp.Symbol(v) for v in vars_to_use]
        try:
            solutions = sp.solve(parsed_equations, symbols_list, dict=True)
        except Exception as e:
            st.error(f"❌ Solver error: {e}")
            st.stop()

    if solutions:
        st.subheader("✅ Solution")
        for idx, solution in enumerate(solutions, 1):
            st.write(f"Solution {idx}:")
            for var, val in solution.items():
                # Convert to float if possible for cleaner display
                try:
                    numeric_val = float(val.evalf())
                    if numeric_val.is_integer():
                        display_val = int(numeric_val)
                    else:
                        display_val = numeric_val
                    st.write(f"• {var} = {display_val}")
                except:
                    st.write(f"• {var} = {val}")
            
            # LaTeX display
            latex_parts = []
            for k, v in solution.items():
                latex_parts.append(f"{k}={sp.latex(v)}")
            st.latex(",\\quad ".join(latex_parts))
    else:
        st.error("❌ No solution found (system may be inconsistent, underdetermined, or non-linear).")
        
        # Provide some debugging info
        st.subheader("🔍 Debug Information")
        st.write("**Detected equations:**")
        for eq in parsed_equations:
            st.write(f"• {eq}")
        
        # Try to identify the issue
        if len(parsed_equations) < len(vars_to_use):
            st.warning(f"⚠️ Underdetermined system: {len(parsed_equations)} equations for {len(vars_to_use)} variables")
        elif len(parsed_equations) > len(vars_to_use):
            st.warning(f"⚠️ Overdetermined system: {len(parsed_equations)} equations for {len(vars_to_use)} variables")

else:
    st.info("👆 Please upload an image to get started.")