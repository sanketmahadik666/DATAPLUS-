# Test Data Directory

## 📁 How to Use This Directory

### 1. **For PDF Documents**
Place your PDF files directly in this directory:
```
test_data/
├── your_document1.pdf
├── your_document2.pdf
├── your_document3.pdf
└── ...
```

### 2. **For Image Documents**
Place your image files (PNG, JPG, etc.) in this directory:
```
test_data/
├── your_image1.png
├── your_image2.jpg
├── your_image3.tiff
└── ...
```

### 3. **For Mixed Documents**
You can mix PDFs and images in the same directory:
```
test_data/
├── document1.pdf
├── image1.png
├── document2.pdf
├── image2.jpg
└── ...
```

## 🚀 How to Run Tests

### **Option 1: Test with Your Data (Recommended)**
```bash
# From the ocr_routing_pipeline directory
python test_with_custom_data.py
```

### **Option 2: Test with Original Data**
```bash
# From the ocr_routing_pipeline directory  
python run_comprehensive_tests.py
```

## 📊 What Happens

1. **Metadata Extraction**: All documents in `test_data/` will be processed
2. **OCR Routing**: Each document gets an optimal OCR engine recommendation
3. **Results**: Generated in `test_data_results.json`
4. **Performance**: Uses 32 workers for maximum speed

## 🎯 Expected Results

- **Processing Speed**: 100+ files/second
- **Success Rate**: 100%
- **Output**: Complete metadata + OCR routing recommendations
- **Format**: JSON with all document features and routing decisions

## 📝 Notes

- **No code changes needed** - just place your files here
- **Supports all document types** - PDFs, images, mixed
- **Automatic detection** - finds all supported files
- **Parallel processing** - uses 32 workers for speed
