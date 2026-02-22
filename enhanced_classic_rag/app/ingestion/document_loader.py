"""
Document Loader — extracts text and tables from PDFs using PyMuPDF.

Uses PyMuPDF (fitz) to:
- Extract text from each page
- Extract tables from each page
- Maintain page number tracking
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and extracts content from PDF documents."""
    
    def __init__(self):
        """Initialize document loader."""
        pass
    
    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a PDF and extract text and tables from each page.
        
        Args:
            file_path: Path to the PDF file.
        
        Returns:
            List of page data dictionaries with structure:
            [
                {
                    "page_number": 1,
                    "text": "extracted text...",
                    "tables": [
                        {
                            "data": [[cell, cell, ...], ...],
                            "bbox": (x0, y0, x1, y1)
                        },
                        ...
                    ]
                },
                ...
            ]
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"Only PDF files are supported, got: {file_path}")
        
        pages_data = []
        
        try:
            doc = fitz.open(file_path)
            logger.info(f"Loading PDF: {Path(file_path).name} ({len(doc)} pages)")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text("text")
                
                # Extract tables
                tables = self._extract_tables(page)
                
                pages_data.append({
                    "page_number": page_num + 1,  # 1-indexed
                    "text": text,
                    "tables": tables
                })
            
            doc.close()
            logger.info(f"Extracted {len(pages_data)} pages from {Path(file_path).name}")
            
            return pages_data
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def _extract_tables(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """
        Extract tables from a page.
        
        Args:
            page: PyMuPDF page object.
        
        Returns:
            List of table dictionaries.
        """
        tables = []
        
        try:
            # PyMuPDF's table extraction
            table_data = page.find_tables()
            
            if table_data and hasattr(table_data, 'tables'):
                for table in table_data.tables:
                    if table:
                        tables.append({
                            "data": table.extract(),
                            "bbox": table.bbox if hasattr(table, 'bbox') else None
                        })
        
        except Exception as e:
            logger.warning(f"Could not extract tables from page: {e}")
        
        return tables
    
    def _table_to_string(self, table: Dict[str, Any]) -> str:
        """
        Convert table data to structured string format.
        
        Args:
            table: Table dictionary with 'data' key.
        
        Returns:
            Formatted table string.
        """
        if not table or "data" not in table:
            return ""
        
        rows = table["data"]
        if not rows:
            return ""
        
        # Format as markdown-style table
        lines = []
        for i, row in enumerate(rows):
            # Filter out None values and convert to strings
            row_str = " | ".join(str(cell) if cell else "" for cell in row)
            lines.append(row_str)
            
            # Add separator after header row
            if i == 0 and len(rows) > 1:
                separator = " | ".join("---" for _ in row)
                lines.append(separator)
        
        return "\n".join(lines)
    
    def get_table_strings(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert tables in pages data to string format.
        
        Args:
            pages_data: Pages data from load_pdf().
        
        Returns:
            List of table data with string representations:
            [
                {
                    "page_number": 1,
                    "table_index": 0,
                    "table_string": "formatted table..."
                },
                ...
            ]
        """
        table_strings = []
        
        for page_data in pages_data:
            page_num = page_data["page_number"]
            tables = page_data.get("tables", [])
            
            for idx, table in enumerate(tables):
                table_str = self._table_to_string(table)
                if table_str:
                    table_strings.append({
                        "page_number": page_num,
                        "table_index": idx,
                        "table_string": table_str
                    })
        
        return table_strings


if __name__ == "__main__":
    import sys
    
    print("=== Document Loader Test ===\n")
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        try:
            loader = DocumentLoader()
            pages = loader.load_pdf(pdf_path)
            
            print(f"✅ Loaded {len(pages)} pages")
            
            # Show first page preview
            if pages:
                first_page = pages[0]
                print(f"\n📄 Page {first_page['page_number']}:")
                print(f"   Text length: {len(first_page['text'])} chars")
                print(f"   Tables found: {len(first_page['tables'])}")
                print(f"   Text preview: {first_page['text'][:200]}...")
                
                # Show table strings
                table_strings = loader.get_table_strings(pages)
                print(f"\n📊 Total tables: {len(table_strings)}")
                if table_strings:
                    print(f"   First table preview:\n{table_strings[0]['table_string'][:200]}...")
            
            print("\n✅ Test passed!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Usage: python document_loader.py <path_to_pdf>")
        print("\nNote: This is a test script. The module is ready for integration.")
