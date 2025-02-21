import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# Directly configure the Gemini API key
genai.configure(api_key="AIzaSyC0ecScQmY0pvJlkiFB3w6VFI9bWrVCDsM")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

def convert_table_to_sentences_gemini(table_data, table_index):
    """Converts table data into descriptive sentences using Gemini Generative AI."""
    table_input = f"Table {table_index}: " + "\n".join([", ".join(row) for row in table_data])
    
    chat_session = model.start_chat(
        history=[
            {"role": "user", "parts": [
                "Convert table to descriptive sentences. Example:",
                "For customers under the age of 60 with a special period: \n"
                "- For 18 months, the interest rates are 7.80% per annum at maturity..."
            ]},
            {"role": "model", "parts": ["Please provide the table content."]},
        ]
    )
    
    response = chat_session.send_message(table_input)
    return response.text

def scrape_webpage(url):
    """Extracts tables and FAQs from a given webpage."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print("Failed to fetch the webpage.")
        return {"error": "Failed to fetch webpage"}
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract tables
    tables = soup.find_all("table")
    extracted_data = {"tables": [], "faqs": []}

    for i, table in enumerate(tables, start=1):
        rows = table.find_all("tr")
        table_data = []

        for row in rows:
            columns = row.find_all("td")
            column_text = [col.get_text(strip=True) for col in columns]
            if column_text:
                table_data.append(column_text)

        if table_data:
            descriptive_sentences = convert_table_to_sentences_gemini(table_data, i)
            extracted_data["tables"].append({"table_index": i, "description": descriptive_sentences})
    
    # Extract FAQs
    faq_section = soup.find(class_="faqs")
    if faq_section:
        for faq in faq_section.find_all(class_="accordion_row"):
            question = faq.find(class_="accordion_toggle").get_text(strip=True) if faq.find(class_="accordion_toggle") else ""
            answer = faq.find(class_="accordion_body").get_text(strip=True) if faq.find(class_="accordion_body") else ""
            if question and answer:
                extracted_data["faqs"].append({"question": question, "answer": answer})

    return extracted_data

# Remove direct execution of scrape_website()
