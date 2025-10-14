#!/usr/bin/env python3
"""
Script to split synthetic procurement documents and convert them to PDF format.
This script will:
1. Split the combined document into individual files
2. Convert each document to PDF format
3. Organize files in a structured directory
"""

import os
import re
from pathlib import Path
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def create_directories():
    """Create directory structure for organized output"""
    base_dir = Path("procurement_documents")
    subdirs = [
        "contracts/domestic",
        "contracts/international", 
        "vendor_management/registration",
        "vendor_management/performance",
        "purchase_orders/domestic",
        "purchase_orders/international",
        "invoices/supplier",
        "invoices/service",
        "compliance/regulatory_circulars",
        "compliance/audit_notices"
    ]
    
    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return base_dir

def parse_documents(file_path):
    """Parse the combined document file and extract individual documents"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by document markers
    doc_pattern = r'---DOC START---(.*?)---DOC END---'
    documents = re.findall(doc_pattern, content, re.DOTALL)
    
    parsed_docs = []
    for doc in documents:
        lines = doc.strip().split('\n')
        
        # Extract metadata
        doc_id = None
        department = None
        sub_department = None
        desk = None
        sub_desk = None
        content_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith('Document_ID:'):
                doc_id = line.split(':', 1)[1].strip()
            elif line.startswith('Department:'):
                department = line.split(':', 1)[1].strip()
            elif line.startswith('Sub-Department:'):
                sub_department = line.split(':', 1)[1].strip()
            elif line.startswith('Desk:'):
                desk = line.split(':', 1)[1].strip()
            elif line.startswith('Sub-Desk:'):
                sub_desk = line.split(':', 1)[1].strip()
            elif line.startswith('Document Content:'):
                content_start = i + 1
                break
        
        # Extract document content
        doc_content = '\n'.join(lines[content_start:]).strip()
        
        parsed_docs.append({
            'id': doc_id,
            'department': department,
            'sub_department': sub_department,
            'desk': desk,
            'sub_desk': sub_desk,
            'content': doc_content
        })
    
    return parsed_docs

def get_file_path(base_dir, doc_info):
    """Determine the file path based on document metadata"""
    sub_department = doc_info['sub_department'].lower().replace(' ', '_')
    sub_desk = doc_info['sub_desk'].lower().replace(' ', '_')
    
    if sub_department == 'contracts':
        if 'domestic' in sub_desk:
            return base_dir / "contracts" / "domestic" / f"{doc_info['id']}.pdf"
        else:
            return base_dir / "contracts" / "international" / f"{doc_info['id']}.pdf"
    
    elif sub_department == 'vendor_management':
        if 'registration' in sub_desk:
            return base_dir / "vendor_management" / "registration" / f"{doc_info['id']}.pdf"
        else:
            return base_dir / "vendor_management" / "performance" / f"{doc_info['id']}.pdf"
    
    elif sub_department == 'purchase_orders':
        if 'domestic' in sub_desk:
            return base_dir / "purchase_orders" / "domestic" / f"{doc_info['id']}.pdf"
        else:
            return base_dir / "purchase_orders" / "international" / f"{doc_info['id']}.pdf"
    
    elif sub_department == 'invoices':
        if 'supplier' in sub_desk:
            return base_dir / "invoices" / "supplier" / f"{doc_info['id']}.pdf"
        else:
            return base_dir / "invoices" / "service" / f"{doc_info['id']}.pdf"
    
    elif sub_department == 'compliance_notices':
        if 'regulatory' in sub_desk:
            return base_dir / "compliance" / "regulatory_circulars" / f"{doc_info['id']}.pdf"
        else:
            return base_dir / "compliance" / "audit_notices" / f"{doc_info['id']}.pdf"
    
    # Fallback
    return base_dir / f"{doc_info['id']}.pdf"

def create_html_content(doc_info):
    """Create HTML content for the document"""
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{doc_info['id']} - {doc_info['sub_department']}</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
                background-color: #fff;
            }}
            .header {{
                border-bottom: 3px solid #2c3e50;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .doc-id {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }}
            .metadata {{
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 4px solid #007bff;
                margin-bottom: 30px;
            }}
            .metadata table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .metadata td {{
                padding: 8px;
                border-bottom: 1px solid #dee2e6;
            }}
            .metadata td:first-child {{
                font-weight: bold;
                width: 30%;
            }}
            .content {{
                margin-top: 30px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
                margin-top: 30px;
                margin-bottom: 15px;
            }}
            h1 {{
                font-size: 28px;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
            }}
            h2 {{
                font-size: 22px;
                color: #007bff;
            }}
            h3 {{
                font-size: 18px;
                color: #495057;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th, td {{
                border: 1px solid #dee2e6;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #007bff;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            ul, ol {{
                margin: 15px 0;
                padding-left: 30px;
            }}
            li {{
                margin: 8px 0;
            }}
            .signature {{
                margin-top: 40px;
                border-top: 2px solid #dee2e6;
                padding-top: 20px;
                text-align: right;
                font-style: italic;
            }}
            .malayalam {{
                font-family: 'Noto Sans Malayalam', 'Arial', sans-serif;
                background-color: #fff3cd;
                padding: 10px;
                border-left: 4px solid #ffc107;
                margin: 15px 0;
            }}
            .footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 12px;
                color: #6c757d;
                border-top: 1px solid #dee2e6;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="doc-id">{doc_info['id']}</div>
            <div style="font-size: 18px; color: #007bff;">{doc_info['sub_department']}</div>
        </div>
        
        <div class="metadata">
            <table>
                <tr><td>Document ID:</td><td>{doc_info['id']}</td></tr>
                <tr><td>Department:</td><td>{doc_info['department']}</td></tr>
                <tr><td>Sub-Department:</td><td>{doc_info['sub_department']}</td></tr>
                <tr><td>Desk:</td><td>{doc_info['desk']}</td></tr>
                <tr><td>Sub-Desk:</td><td>{doc_info['sub_desk']}</td></tr>
            </table>
        </div>
        
        <div class="content">
            {markdown.markdown(doc_info['content'])}
        </div>
        
        <div class="footer">
            <p>Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Document Management System - Synthetic Data</p>
        </div>
    </body>
    </html>
    """
    return html_template

def convert_to_pdf(html_content, output_path):
    """Convert HTML content to PDF using WeasyPrint"""
    try:
        font_config = FontConfiguration()
        html_doc = HTML(string=html_content)
        html_doc.write_pdf(output_path, font_config=font_config)
        return True
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        return False

def main():
    """Main function to process documents"""
    print("Starting document processing...")
    
    # Create directory structure
    base_dir = create_directories()
    print(f"Created directory structure: {base_dir}")
    
    # Parse documents
    input_file = "synthetic_procurement_documents.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    documents = parse_documents(input_file)
    print(f"Parsed {len(documents)} documents")
    
    # Process each document
    success_count = 0
    for doc in documents:
        try:
            # Create HTML content
            html_content = create_html_content(doc)
            
            # Determine output path
            output_path = get_file_path(base_dir, doc)
            
            # Convert to PDF
            if convert_to_pdf(html_content, output_path):
                print(f"✓ Created: {output_path}")
                success_count += 1
            else:
                print(f"✗ Failed: {doc['id']}")
                
        except Exception as e:
            print(f"✗ Error processing {doc['id']}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully created {success_count}/{len(documents)} PDF files")
    print(f"Files saved in: {base_dir.absolute()}")

if __name__ == "__main__":
    main()
