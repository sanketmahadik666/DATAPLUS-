#!/usr/bin/env python3
"""
Simple script to split synthetic procurement documents into individual files.
This script will create separate text files for each document and organize them in directories.
"""

import os
import re
from pathlib import Path

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
            return base_dir / "contracts" / "domestic" / f"{doc_info['id']}.txt"
        else:
            return base_dir / "contracts" / "international" / f"{doc_info['id']}.txt"
    
    elif sub_department == 'vendor_management':
        if 'registration' in sub_desk:
            return base_dir / "vendor_management" / "registration" / f"{doc_info['id']}.txt"
        else:
            return base_dir / "vendor_management" / "performance" / f"{doc_info['id']}.txt"
    
    elif sub_department == 'purchase_orders':
        if 'domestic' in sub_desk:
            return base_dir / "purchase_orders" / "domestic" / f"{doc_info['id']}.txt"
        else:
            return base_dir / "purchase_orders" / "international" / f"{doc_info['id']}.txt"
    
    elif sub_department == 'invoices':
        if 'supplier' in sub_desk:
            return base_dir / "invoices" / "supplier" / f"{doc_info['id']}.txt"
        else:
            return base_dir / "invoices" / "service" / f"{doc_info['id']}.txt"
    
    elif sub_department == 'compliance_notices':
        if 'regulatory' in sub_desk:
            return base_dir / "compliance" / "regulatory_circulars" / f"{doc_info['id']}.txt"
        else:
            return base_dir / "compliance" / "audit_notices" / f"{doc_info['id']}.txt"
    
    # Fallback
    return base_dir / f"{doc_info['id']}.txt"

def create_document_content(doc_info):
    """Create formatted document content"""
    content = f"""
DOCUMENT INFORMATION
===================
Document ID: {doc_info['id']}
Department: {doc_info['department']}
Sub-Department: {doc_info['sub_department']}
Desk: {doc_info['desk']}
Sub-Desk: {doc_info['sub_desk']}

DOCUMENT CONTENT
================

{doc_info['content']}

---
Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Document Management System - Synthetic Data
"""
    return content

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
            # Create document content
            doc_content = create_document_content(doc)
            
            # Determine output path
            output_path = get_file_path(base_dir, doc)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)
            
            print(f"✓ Created: {output_path}")
            success_count += 1
                
        except Exception as e:
            print(f"✗ Error processing {doc['id']}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully created {success_count}/{len(documents)} document files")
    print(f"Files saved in: {base_dir.absolute()}")
    
    # Create a summary file
    summary_path = base_dir / "document_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("PROCUREMENT DOCUMENTS SUMMARY\n")
        f.write("============================\n\n")
        f.write(f"Total Documents: {len(documents)}\n")
        f.write(f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DOCUMENT LIST:\n")
        f.write("=============\n")
        for doc in documents:
            f.write(f"{doc['id']} - {doc['sub_department']} - {doc['sub_desk']}\n")
    
    print(f"✓ Created summary: {summary_path}")

if __name__ == "__main__":
    main()
