import requests
import json
import base64
from pathlib import Path
import simple_icd_10_cm as cm
import sys

class ICD10Analyzer:
    def __init__(self, model="qwen2.5vl:7b"):
        self.model = model
        self.api_url = "http://127.0.0.1:11434/api/generate"
        
        try:
            r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            models = [m['name'] for m in r.json().get('models', [])]
            if model not in models:
                print(f"Model '{model}' not found. Run: ollama pull {model}")
                sys.exit(1)
        except:
            print("Ollama not running. Run: ollama serve")
            sys.exit(1)
        
        all_codes = cm.get_all_codes()
        self.chapters = [c for c in all_codes if len(c) <= 3]
    
    def _call_ollama(self, prompt, image_path):
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2048}
        }
        
        r = requests.post(self.api_url, json=payload, timeout=180)
        return r.json()['response']
    
    def analyze(self, image_path):
        if not Path(image_path).exists():
            print(f"File not found: {image_path}")
            return None
        
        print(f"\nAnalyzing: {image_path}\n")
        print("="*80)
        
        # Extract medical conditions
        prompt = """Read this medical document/image carefully.

Extract ONLY:
1. Primary diagnosis/condition
2. Secondary diagnoses (if any)
3. Symptoms mentioned
4. Procedures performed

List them as bullet points. Be precise and brief."""
        
        findings = self._call_ollama(prompt, image_path)
        print("FINDINGS:")
        print(findings)
        print("="*80 + "\n")
        
        # Get ICD-10 codes
        prompt2 = f"""Medical findings:
{findings}

Task: Provide ONLY the most relevant ICD-10 codes (max 5 codes).

For EACH code, respond EXACTLY as:
CODE: <ICD-10 code>
DIAGNOSIS: <condition name>
REASON: <one sentence why this code applies>

Example:
CODE: J43.9
DIAGNOSIS: Emphysema, unspecified
REASON: Patient diagnosed with alveolar emphysema

Your response:"""
        
        response = self._call_ollama(prompt2, image_path)
        
        # Parse codes
        codes = []
        lines = response.strip().split('\n')
        
        current = {}
        for line in lines:
            line = line.strip()
            if line.startswith('CODE:'):
                if current:
                    codes.append(current)
                current = {'code': line.replace('CODE:', '').strip()}
            elif line.startswith('DIAGNOSIS:'):
                current['diagnosis'] = line.replace('DIAGNOSIS:', '').strip()
            elif line.startswith('REASON:'):
                current['reason'] = line.replace('REASON:', '').strip()
        
        if current:
            codes.append(current)
        
        # Display results
        print("ICD-10 CODES:")
        print("="*80)
        
        if codes:
            for i, code in enumerate(codes, 1):
                print(f"{i}. {code.get('code', 'N/A')}")
                print(f"   {code.get('diagnosis', 'N/A')}")
                print(f"   > {code.get('reason', 'N/A')}\n")
        else:
            print("No ICD-10 codes identified\n")
        
        # Save
        result = {
            'image': image_path,
            'findings': findings,
            'icd_codes': codes
        }
        
        output = f"{Path(image_path).stem}_result.json"
        with open(output, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {output}")
        return result

if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    analyzer = ICD10Analyzer()
    analyzer.analyze(image)