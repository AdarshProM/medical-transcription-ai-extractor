# Medical Transcription AI Data Extractor
# Author: Adarsh Singh
# Description: Extracts patient age, recommended treatments, and ICD codes
# from medical transcriptions using OpenAI GPT-4o-mini with Function Calling

import json
import os
import pandas as pd
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ─────────────────────────────────────────────
# STEP 1: Load the data
# ─────────────────────────────────────────────
print("Loading medical transcription data...")
df = pd.read_csv("data/transcriptions.csv")
print(f"Loaded {len(df)} transcriptions successfully.\n")


# ─────────────────────────────────────────────
# STEP 2: Extract Age & Treatment using Function Calling
# ─────────────────────────────────────────────
def extract_info_with_openai(transcription):
    """
    Extracts patient age and recommended treatment from a transcription
    using OpenAI GPT-4o-mini with Function Calling for structured output.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a healthcare professional extracting patient data. "
                "Always return both the age and recommended treatment. "
                "If information is missing, still create the field and specify 'Unknown'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Please extract and return both the patient's age and recommended "
                f"treatment from the following transcription.\n\nTranscription: {transcription}"
            )
        }
    ]

    function_definition = [
        {
            "type": "function",
            "function": {
                "name": "extract_medical_data",
                "description": (
                    "Get the age and recommended treatment from the input text. "
                    "Always return both age and recommended treatment."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Age": {
                            "type": "integer",
                            "description": "Age of the patient"
                        },
                        "Recommended Treatment/Procedure": {
                            "type": "string",
                            "description": "Recommended treatment or procedure for the patient"
                        }
                    },
                    "required": ["Age", "Recommended Treatment/Procedure"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=function_definition,
        tool_choice={"type": "function", "function": {"name": "extract_medical_data"}}
    )

    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)


# ─────────────────────────────────────────────
# STEP 3: Get ICD Codes for Treatment
# ─────────────────────────────────────────────
def get_icd_codes(treatment):
    """
    Retrieves ICD-10 codes for a given medical treatment using OpenAI.
    Returns 'Unknown' if treatment is not identified.
    """
    if treatment == "Unknown":
        return "Unknown"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Provide the ICD-10 codes for the following treatment or procedure: "
                f"{treatment}. Return the answer as a list of codes only. "
                f"No explanations, just the codes."
            )
        }],
        temperature=0.3
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# STEP 4: Process All Transcriptions
# ─────────────────────────────────────────────
print("Processing medical transcriptions with AI...\n")
processed_data = []

for index, row in df.iterrows():
    print(f"Processing record {index + 1}/{len(df)}...")

    medical_specialty = row["medical_specialty"]

    # Extract age and treatment
    extracted_data = extract_info_with_openai(row["transcription"])

    # Get ICD codes
    treatment = extracted_data.get("Recommended Treatment/Procedure", "Unknown")
    icd_code = get_icd_codes(treatment)

    # Add additional fields
    extracted_data["Medical Specialty"] = medical_specialty
    extracted_data["ICD Code"] = icd_code

    processed_data.append(extracted_data)


# ─────────────────────────────────────────────
# STEP 5: Save Results
# ─────────────────────────────────────────────
df_structured = pd.DataFrame(processed_data)

# Save to CSV
output_path = "output/structured_medical_data.csv"
os.makedirs("output", exist_ok=True)
df_structured.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("✅ Processing Complete!")
print("=" * 60)
print(f"\n📊 Processed {len(df_structured)} records")
print(f"💾 Results saved to: {output_path}")
print("\n📋 Sample Output:")
print(df_structured.head())
