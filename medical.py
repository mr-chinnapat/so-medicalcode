import subprocess
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import requests
import re

class MedicalDocAnalyzer:
    def __init__(self, use_typhoon=False):
        print("กำลังเตรียมระบบ...")
        
        self.use_typhoon = use_typhoon
        
        # MedGemma AI
        self.model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-4b-it",
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
        
        print("พร้อมใช้งาน\n")
    
    def ocr_typhoon(self, image_path):
        """ใช้ Typhoon OCR ผ่าน Ollama"""
        print("กำลังอ่านด้วย Typhoon OCR...")
        result = subprocess.run(
            ["ollama", "run", "scb10x/typhoon-ocr-3b", image_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # แปลง JSON response
        try:
            # หา JSON ในผลลัพธ์
            json_match = re.search(r'\{.*\}', result.stdout, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("natural language", "")
        except:
            pass
        
        return result.stdout
    
    def search_icd10_api(self, diagnosis):
        """ค้นหา ICD-10"""
        url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
        response = requests.get(url, params={
            'sf': 'code,name',
            'terms': diagnosis,
            'maxList': 3
        })
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 3 and data[3]:
                return data[3]
        return []
    
    def analyze(self, image_path):
        image = Image.open(image_path).convert('RGB')
        
        # 1. OCR
        if self.use_typhoon:
            ocr_text = self.ocr_typhoon(image_path)
        else:
            # ใช้ EasyOCR
            import easyocr
            reader = easyocr.Reader(['th', 'en'], gpu=False)
            results = reader.readtext(image_path)
            ocr_text = ' '.join([text for (_, text, conf) in results if conf > 0.3])
        
        print("="*70)
        print("ข้อความที่อ่านได้:")
        print("="*70)
        print(ocr_text[:300])
        print("="*70)
        
        # 2. MedGemma AI วิเคราะห์
        print("\nกำลังให้ AI วิเคราะห์...")
        
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Medical document OCR text:
{ocr_text[:500]}

Extract:
1. Diagnosis (what disease/condition mentioned)
2. Be brief and accurate"""
                },
                {"type": "image", "image": image}
            ]
        }]
        
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        )
        
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=200,
                do_sample=False, repetition_penalty=2.5
            )
        
        ai_result = self.processor.decode(
            generation[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        print("\n" + "="*70)
        print("🤖 AI Analysis:")
        print("="*70)
        print('\n'.join(ai_result.split('\n')[:5]))
        
        # 3. ค้นหา ICD-10
        diagnosis = input("\nพิมพ์การวินิจฉัยภาษาอังกฤษ: ")
        
        icd_results = self.search_icd10_api(diagnosis)
        
        print("\n" + "="*70)
        print("🏥 ICD-10 Codes:")
        print("="*70)
        
        if icd_results:
            for code, desc in icd_results:
                print(f"  {code} - {desc}")
        else:
            print("  ไม่พบ")
        
        print("="*70)

# ใช้งาน
# แบบที่ 1: ใช้ Typhoon OCR
analyzer = MedicalDocAnalyzer(use_typhoon=True)

# แบบที่ 2: ใช้ EasyOCR (แม่นกว่า)
# analyzer = MedicalDocAnalyzer(use_typhoon=False)

analyzer.analyze("/Users/misterchin/Downloads/S__29212816.jpg")