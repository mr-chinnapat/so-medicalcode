import os
"""
Flask Backend API for ICD-10 Analyzer
pip install flask flask-cors google-generativeai pillow
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image
import os
import tempfile
import logging

app = Flask(__name__)
CORS(app)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise Exception("GEMINI_API_KEY not set")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')

# Config
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze():
    tmp_path = None
    
    try:
        # Validate file upload
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use JPG or PNG'}), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': f'File too large. Max size: 10MB'}), 400
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        logger.info(f"Processing image: {file.filename} ({file_size} bytes)")
        
        # Open and validate image
        try:
            image = Image.open(tmp_path)
            image.verify()
            image = Image.open(tmp_path)  # Reopen after verify
        except Exception as e:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Step 1: Read handwriting
        prompt1 = """อ่านเอกสารทางการแพทย์นี้ โดยเฉพาะลายมือ

ดึงข้อมูล:
1. วินิจฉัย 1 (Diagnosis 1): [อ่านลายมือให้แม่นยำ]
2. วินิจฉัย 2 (Diagnosis 2): [ถ้ามี]
3. การรักษา (Treatment): [ถ้ามี]
4. การผ่าตัด (Surgery): [ถ้ามี]

ตอบสั้นๆ:
Diagnosis 1: <ข้อความ>
Diagnosis 2: <ข้อความ or none>
Treatment: <ข้อความ or none>
Surgery: <ข้อความ or none>"""
        
        try:
            response1 = model.generate_content([prompt1, image])
            findings = response1.text
        except Exception as e:
            logger.error(f"Gemini API error (step 1): {e}")
            return jsonify({'error': 'AI service unavailable'}), 503
        
        # Step 2: Get ICD-10
        prompt2 = f"""จากข้อมูล:
{findings}

ระบุรหัส ICD-10 ที่แม่นยำ (สูงสุด 3 รหัส)

ตอบในรูปแบบนี้:
CODE: <ICD-10>
DIAGNOSIS: <ชื่อโรค English>
REASON: <เหตุผลสั้นๆ>

กฎ:
- ใช้รหัส ICD-10 ที่ถูกต้อง (เช่น L72.0 สำหรับ Epidermal cyst)
- วินิจฉัยหลักก่อน
- ต้องตรงกับที่วินิจฉัยจริง"""
        
        try:
            response2 = model.generate_content([prompt2])
        except Exception as e:
            logger.error(f"Gemini API error (step 2): {e}")
            return jsonify({'error': 'AI service unavailable'}), 503
        
        # Parse codes
        codes = []
        lines = response2.text.strip().split('\n')
        
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
        
        logger.info(f"Analysis complete: {len(codes)} codes found")
        
        return jsonify({
            'success': True,
            'findings': findings,
            'icd_codes': codes
        })
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal server error'}), 500
    
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


if __name__ == '__main__':
    print("="*60)
    print("ICD-10 Analyzer Backend Server")
    print("="*60)
    print(f"GEMINI_API_KEY: {'✓ Set' if api_key else '✗ Not set'}")
    print(f"Server: http://localhost:5000")
    print("="*60)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=True, port=port)