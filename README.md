# 📷 OCR-based Linear Equation Solver

This Streamlit web app lets you upload a scanned or handwritten image of linear equations (like `x + y = 2`, `x - y = 0`) and solves them automatically using Tesseract OCR and SymPy.

---

## ✨ Features

- 🖼️ Upload printed or handwritten math images
- 🔍 Uses Tesseract OCR to extract equations
- 🔧 Auto-repairs common OCR issues (e.g. `t` ➝ `+`, `Y` ➝ `y`)
- ➗ Solves linear equations using SymPy
- 📐 Displays step-by-step parsed equations and LaTeX output
- 🧪 Supports up to 3 variables (`x`, `y`, `z` etc.)

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
````

Also install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki):

* **Windows**: Download the installer and add path to `tesseract.exe`
* **macOS**: `brew install tesseract`
* **Ubuntu**: `sudo apt install tesseract-ocr`

---

## 🚀 Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
ocr-equation-solver/
├── app.py               # Streamlit app
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🧠 Example Input

Upload an image with:

```
x + y = 2
x - y = 0
```

✅ Output:

```
x = 1
y = 1
```

---

## 📚 Tech Stack

* [Streamlit](https://streamlit.io) – UI
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) – Text extraction
* [SymPy](https://www.sympy.org/) – Symbolic math engine
* [OpenCV](https://opencv.org/) – Image processing

---

## 🙋‍♂️ Author

Made with ❤️ by [Your Name](https://github.com/yourusername)

---

## 📸 Screenshots

> Replace with real screenshots or animated demo GIF

---

## ☁️ Deploy on Streamlit Cloud

1. Push code to GitHub
2. Visit [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New App" → Connect your repo → Deploy

---


