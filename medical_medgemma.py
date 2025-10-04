from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import requests
import re

class AIOnlyMedicalAnalyzer:
    def __init__(self):
        print("กำลังโหลด MedGemma AI...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-4b-it",
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
        print("พร้อมใช้งาน\n")
    
    def search_icd10(self, diagnosis):
        """ค้นหา ICD-10"""
        url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
        response = requests.get(url, params={
            'sf': 'code,name',
            'terms': diagnosis,
            'maxList': 5
        })
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 3 and data[3]:
                return data[3]
        return []
    
    def analyze(self, image_path):
        image = Image.open(image_path).convert('RGB')
        
        # AI วิเคราะห์และให้ ICD-10 ตรงๆ
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this medical document image and provide ICD-10 code.

Output format (STRICTLY follow this):
Diagnosis: [medical condition in English]
ICD-10: [code]

Example:
Diagnosis: Chronic obstructive pulmonary disease
ICD-10: J44.9

Now analyze:"""
                },
                {"type": "image", "image": image}
            ]
        }]
        
        print("AI กำลังวิเคราะห์...")
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
            )
        
        result = self.processor.decode(
            generation[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        
        # ดึงข้อมูล
        diagnosis = ""
        ai_icd = ""
        
        for line in result.split('\n')[:5]:
            if 'diagnosis' in line.lower() and ':' in line:
                diagnosis = line.split(':')[1].strip()
                diagnosis = re.sub(r'[^a-zA-Z\s]', '', diagnosis)
            elif 'icd' in line.lower() and ':' in line:
                ai_icd = line.split(':')[1].strip()
        
        # ถ้าไม่มี diagnosis ให้ใช้บรรทัดแรก
        if not diagnosis:
            diagnosis = re.sub(r'[^a-zA-Z\s]', '', result.split('\n')[0])
        
        diagnosis = diagnosis[:50]  # ตัดให้สั้น
        
        print(f"\nAI Analysis: {result[:200]}")
        
        # ค้นหา ICD-10 จาก API
        print(f"\nค้นหา ICD-10 สำหรับ: {diagnosis}")
        icd_results = self.search_icd10(diagnosis)
        
        # แสดงผล
        print("\n" + "="*70)
        print("ผลการวิเคราะห์")
        print("="*70)
        print(f"การวินิจฉัย: {diagnosis}")
        
        if ai_icd:
            print(f"ICD-10 (จาก AI): {ai_icd}")
        
        print("\nICD-10 (จาก Database):")
        print("="*70)
        
        if icd_results:
            for code, desc in icd_results:
                print(f"{code} - {desc}")
        else:
            print("ไม่พบรหัส")
        
        print("="*70)
        
        return {
            "diagnosis": diagnosis,
            "ai_icd": ai_icd,
            "db_icd": icd_results
        }

# ใช้งาน
analyzer = AIOnlyMedicalAnalyzer()
result = analyzer.analyze("/Users/misterchin/Downloads/S__29212816.jpg")