# Procurement Documents - PDF Conversion Guide

This directory contains synthetic procurement documents that have been split into individual files and converted to HTML format for easy PDF generation.

## Directory Structure

```
procurement_documents_html/
├── index.html                          # Main index page
├── contracts/
│   ├── domestic/                       # Domestic contracts
│   │   ├── DOC-001.html               # Contract renewal notice
│   │   └── DOC-011.html               # Contract amendment
│   └── international/                  # International contracts
│       └── DOC-006.html               # International service agreement
├── vendor_management/
│   ├── registration/                   # Vendor registration
│   │   ├── DOC-002.html               # New vendor registration
│   │   └── DOC-012.html               # Vendor blacklist notice
│   └── performance/                    # Vendor performance
│       └── DOC-007.html               # Performance evaluation
├── purchase_orders/
│   ├── domestic/                       # Domestic purchase orders
│   │   └── DOC-008.html               # Domestic PO
│   └── international/                  # International purchase orders
│       ├── DOC-003.html               # International PO
│       └── DOC-013.html               # Urgent international PO
├── invoices/
│   ├── supplier/                       # Supplier invoices
│   │   ├── DOC-004.html               # Invoice processing
│   │   └── DOC-014.html               # Invoice discrepancy
│   └── service/                        # Service invoices
│       └── DOC-009.html               # Service invoice processing
└── compliance/
    ├── regulatory_circulars/           # Regulatory circulars
    │   ├── DOC-005.html               # GST compliance circular
    │   └── DOC-015.html               # Sustainable procurement policy
    └── audit_notices/                  # Audit notices
        └── DOC-010.html               # Internal audit notice
```

## Converting to PDF

### Method 1: Browser Print to PDF (Recommended)

1. **Open the index file**: Double-click `index.html` to open it in your browser
2. **Navigate to documents**: Click on any document link to view it
3. **Print to PDF**: 
   - Click the "Print to PDF" button in the top-right corner, OR
   - Press `Ctrl+P` and select "Save as PDF"

### Method 2: Automated Batch Conversion

1. **Run the batch script**: Double-click `convert_to_pdf.bat`
2. **Wait for completion**: The script will automatically convert all HTML files to PDF
3. **Check results**: PDF files will be created in the same directories as HTML files

### Method 3: Manual Browser Conversion

1. Open each HTML file individually in your browser
2. Use `Ctrl+P` → "Save as PDF" for each document
3. Save with the same filename but `.pdf` extension

## Document Features

Each HTML document includes:
- **Professional formatting** with company branding colors
- **Print-optimized styles** for clean PDF output
- **Metadata section** showing document classification
- **Responsive design** that works on different screen sizes
- **Malayalam text support** for bilingual content
- **Table formatting** with proper borders and styling
- **Print button** for easy PDF generation

## Document Types Generated

### Contracts (4 documents)
- **DOC-001**: Contract renewal notice (Domestic)
- **DOC-006**: International service agreement
- **DOC-011**: Contract amendment notice (Domestic)

### Vendor Management (3 documents)
- **DOC-002**: New vendor registration form
- **DOC-007**: Vendor performance evaluation report
- **DOC-012**: Vendor blacklist notice

### Purchase Orders (3 documents)
- **DOC-003**: International purchase order
- **DOC-008**: Domestic purchase order
- **DOC-013**: Urgent international purchase order

### Invoices (3 documents)
- **DOC-004**: Invoice processing notice
- **DOC-009**: Service invoice processing
- **DOC-014**: Invoice discrepancy report

### Compliance Notices (2 documents)
- **DOC-005**: Regulatory compliance circular
- **DOC-010**: Internal audit notice
- **DOC-015**: Sustainable procurement policy update

## Quality Assurance

All documents include:
- ✅ **Unique document IDs** (DOC-001 through DOC-015)
- ✅ **Realistic business data** (amounts, dates, vendor names)
- ✅ **Proper hierarchy classification** (Department → Sub-Department → Desk → Sub-Desk)
- ✅ **Mixed formatting** (tables, lists, headers, signatures)
- ✅ **Bilingual content** (English + Malayalam snippets)
- ✅ **Professional appearance** suitable for DMS testing

## Usage for DMS Testing

These documents are designed for testing Document Management Systems and can be used to:
- Test document classification algorithms
- Validate metadata extraction
- Test search and retrieval functionality
- Validate document routing workflows
- Test compliance and audit features
- Validate multi-language support

## Technical Notes

- **File encoding**: UTF-8 for proper Malayalam text support
- **Browser compatibility**: Works with Chrome, Firefox, Edge, Safari
- **Print settings**: Optimized for A4 paper size
- **Font support**: Uses system fonts with fallbacks
- **CSS**: Print-specific styles included for clean PDF output

## Support

If you encounter any issues with PDF conversion:
1. Ensure your browser supports PDF generation
2. Check that JavaScript is enabled
3. Try a different browser if conversion fails
4. Use the manual print method as a fallback
