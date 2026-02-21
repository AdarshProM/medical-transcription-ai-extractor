# 🏥 Medical Transcription AI Data Extractor

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)
![Healthcare](https://img.shields.io/badge/Domain-Healthcare%20AI-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

An **AI-powered medical transcription analyzer** that automatically extracts structured patient data from unstructured medical transcriptions using **OpenAI GPT-4o-mini** with **Function Calling**.

This project demonstrates real-world AI/ML application in healthcare — extracting patient age, recommended treatments, and ICD-10 medical codes from raw doctor transcriptions.

---

## 🎯 What This Project Does

| Step | Action |
|------|--------|
| 1️⃣ | Loads medical transcription CSV data using Pandas |
| 2️⃣ | Uses **OpenAI Function Calling** to extract structured data (age + treatment) |
| 3️⃣ | Retrieves **ICD-10 codes** for each identified treatment |
| 4️⃣ | Outputs clean structured CSV with all extracted information |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.8+ | Core programming language |
| OpenAI GPT-4o-mini | AI extraction engine |
| OpenAI Function Calling | Structured JSON output from AI |
| Pandas | Data loading and manipulation |
| ICD-10 Codes | Medical billing standard codes |

---

## 🔑 Key AI Concepts Demonstrated

- **Function Calling (Tool Use)** — Forces AI to return structured JSON output
- **Prompt Engineering** — System + user role configuration for medical context
- **Multi-step AI Pipeline** — Two sequential API calls per record
- **Structured Data Extraction** — Converting unstructured text to structured CSV
- **Healthcare AI** — Real-world domain application

---

## 📁 Project Structure

medical-transcription-ai-extractor/
│
├── main.py # Main application code
├── requirements.txt # Python dependencies
├── .env.example # API key template
├── .gitignore # Git ignore rules
├── README.md # Project documentation
│
├── data/
│ └── sample_transcription.csv # Sample data (demo only)
│
└── output/
└── structured_medical_data.csv # Generated output (auto-created)

text

---

## ⚙️ How to Run

### Step 1: Clone Repository

git clone https://github.com/AdarshProM/medical-transcription-ai-extractor.git
cd medical-transcription-ai-extractor

Step 2: Install Dependencies

pip install -r requirements.txt

Step 3: Set API Key

cp .env.example .env

# Edit .env and add your OpenAI API key
Step 4: Run
python main.py
📊 Sample Output
Input (raw transcription):

text
"Patient is a 45-year-old male with knee pain. 
Recommended treatment is physical therapy and knee replacement."
Output (structured data):

text
Age: 45
Recommended Treatment: Physical therapy and knee replacement surgery
Medical Specialty: Orthopedics
ICD Code: Z96.641, M17.11
🚀 Future Enhancements
 Add Streamlit web interface for real-time transcription analysis

 Deploy as Azure Function / AWS Lambda for cloud processing

 Add HIPAA compliance and data anonymization

 Integrate with hospital management systems via REST API

 Add batch processing for large transcription datasets

 Implement confidence scoring for extracted data

⚠️ Important Note
This project uses sample/synthetic data only. No real patient data is included. Always ensure HIPAA/healthcare data compliance in production environments.

👨‍💻 Author
Adarsh Singh — Google Associate Cloud Engineer | Azure Solutions Architect Expert

LinkedIn: linkedin.com/in/narayanadarsh

Email: adarsh.narayan.official@outlook.com

GitHub: github.com/YOUR_USERNAME

Built as part of AI/Cloud engineering portfolio demonstrating OpenAI API integration,
Function Calling, and healthcare data processing capabilities.
