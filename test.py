import fitz  # PyMuPDF
import pytesseract # Keep for potential future use or simple tasks
from PIL import Image
import io
import os
import regex as re
import pandas as pd
from dateutil import parser as date_parser
from collections import defaultdict
import json
import camelot # For text-based PDF table extraction
import boto3   # For AWS Textract
import time    # For potential Textract async polling
import datetime # For date handling
import logging # Use logging for better output control

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---

# AWS Textract Configuration
AWS_REGION = "ap-south-1"  # <--- CHANGE THIS TO YOUR AWS REGION

# Confidence Thresholds & Weights (Math checks removed)
CONF_THRESHOLD_TRUSTED = 0.80 # Might need adjustment after removing math checks
BASE_CONF_PYMUPDF_TEXT = 0.90 # High base confidence for clean digital text
# Textract provides its own confidences, use them as base
BONUS_FORMAT_VALID = 0.10
PENALTY_FORMAT_FAIL = -0.30 # Slightly less penalty as math check is gone
# PENALTY_MATH_CHECK_FAIL = -0.30 # REMOVED
# BONUS_MATH_CHECK_PASS = 0.25 # REMOVED
# LINE_ITEM_MATH_TOLERANCE = 0.10 # REMOVED (for header/footer totals)
LINE_ITEM_QTY_PRICE_TOLERANCE = 0.10 # Keep tolerance for internal line item check

# PDF Type Detection Heuristics
MIN_TEXT_LENGTH_FOR_TEXT_PDF = 150
REQUIRED_KEYWORDS_FOR_TEXT_PDF = ['invoice', 'total', 'date', 'bill', 'amount']


# --- Define Helper Regex String Variables FIRST ---
# (Keep existing regex strings like amount_capture_decimal_required, etc.)
amount_capture_decimal_required = r"(?:[₹$€£]\s*)?([\d,]+\.\d{1,2})"
# Handles optional currency symbol, optional spaces, commas, optional decimal (allows integers)
amount_capture_optional_decimal = r"(?:[₹$€£]\s*)?([\d,.]*\d)" # Simpler version for broader match
# Captures rate like X%, XX.X%, XX.XX% (inside brackets helper remains same)
rate_inside_brackets_capture = r"\(?\s*(\d{1,2}(?:\.\d{1,2})?)\s*%\s*\)?"
# Captures rate like @X%, @XX.X%
rate_after_at_capture = r"@\s*(\d{1,2}(?:\.\d{1,2})?)\s*%"

# --- NOW Define the PATTERNS Dictionary ---
PATTERNS = {
    # --- Updated Invoice/Date Keywords ---
    "invoice_number": re.compile(r"(?:Invoice Number|Invoice No\.?|Inv\.? No\.?|Invoice #|Inv\.? #|Bill No\.?|Bill Number|Order No\.?|Facture No\.?|Document No\.?|Voucher No\.?|Invoice\.\s*No\.?\s*&\s*Date)" # Added Invoice. No.& Date
                                 r"[:\s-]*#?\s*"
                                 r"([A-Za-z0-9/_-]+)", # Allow underscore as well
                                 re.IGNORECASE),
    "invoice_date": re.compile(r"(?:Invoice Date|Inv\.? Date|Bill Date|Date\s*[:\s-]*|Invoice\.\s*No\.?\s*&\s*Date)" # Added Invoice. No.& Date
                               r"(?!\s*(?:due|Due))[:\s-]*"
                               r"(\d{1,2}[-/.\s]+\w+[-/.\s]+\d{2,4}|\w+\s+\d{1,2},?\s+\d{2,4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})",
                               re.IGNORECASE),

    # --- GSTIN and Keywords ---
    "gstin_pattern": re.compile(r"(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})", re.IGNORECASE), # Keep as is
    # Added "GST IN" to keywords
    "supplier_keywords": re.compile(r"(?:From|Vendor|Seller|Sold By|Company\s*GSTIN|Our\s*GSTIN|GST\s*IN)", re.IGNORECASE),
    "recipient_keywords": re.compile(r"(?:To|Billed To|Bill To|Customer|Buyer|Ship To|Client\s*GSTIN|Your\s*GSTIN|GSTIN\s*Number|GST\s*IN)", re.IGNORECASE), # Added GST IN
    "place_keywords": re.compile(r"Place of (?:Supply|Delivery)|Destination", re.IGNORECASE), # Keep as is

    # --- Refined Tax Amount Patterns ---
    # SGST/CGST likely won't match, which is fine
     "sgst_amount": re.compile(r"(?:SGST|S\.G\.S\.T\.)\d*(?:\s*\(?[^)]+\)?\s*)?\s*" + amount_capture_decimal_required, flags=re.IGNORECASE | re.DOTALL),
    "cgst_amount": re.compile(r"(?:CGST|C\.G\.S\.T\.)\d*(?:\s*\(?[^)]+\)?\s*)?\s*" + amount_capture_decimal_required, flags=re.IGNORECASE | re.DOTALL),
    # IGST won't match either
    "igst_amount": re.compile(r"IGST(?:@[\d.]+%)?\s*" + amount_capture_decimal_required, flags=re.IGNORECASE | re.DOTALL),

    # --- Total Patterns ---
    # Taxable value is the "Total" before GST. Let's make it more specific if possible.
    # Try matching "Total" on a line potentially followed by "GST" line. Using lookahead.
    # OR simpler: rely on first "Total" match before other totals, and use optional decimal.
    "taxable_value": re.compile(r"(?:Taxable Value|Subtotal|Sub-total|Base Amount|Sub Total|^Total$)" + # Added ^Total$ for line start/end match
                               r"\s*[:\s]*" +
                               amount_capture_optional_decimal, # Needs to handle 41,350.00
                               flags=re.IGNORECASE | re.DOTALL | re.MULTILINE), # Added MULTILINE for ^$

    # Tax Amount matches "GST 18%". Capture the numeric value (7443). Use optional decimal.
    "tax_amount": re.compile(r"(?:Total Tax|Tax Amount|Total GST|GST Amount|GST\s*\d{1,2}(?:\.\d+)?%)" + # Added GST % pattern
                             r"\s*[:\s]*" +
                             amount_capture_optional_decimal, # Changed to optional decimal for 7443
                             flags=re.IGNORECASE | re.DOTALL),

    # Final amount matches "Final Net Amount" or "Grand Total"
    "final_amount": re.compile(r"(?:Total Amount|Grand Total|Net Amount|Amount Due|Total Payable|Balance Due|Final Net Amount|(?<!Sub\s)Total)" + # Added Final Net Amount
                               r"\s*[:\s]*" +
                               amount_capture_optional_decimal,
                               flags=re.IGNORECASE | re.DOTALL),

    # --- Rate Patterns ---
    # General pattern for rates like 18%
    "rate_pattern": re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s*%"),
    "sgst_rate": re.compile(r"(?:SGST|S\.G\.S\.T\.)\d*\s*" + rate_inside_brackets_capture, flags=re.IGNORECASE),
    "cgst_rate": re.compile(r"(?:CGST|C\.G\.S\.T\.)\d*\s*" + rate_inside_brackets_capture, flags=re.IGNORECASE),
    "igst_rate": re.compile(r"IGST\s*" + rate_after_at_capture, flags=re.IGNORECASE),
    # Maybe add a generic GST rate pattern if needed, e.g., from "GST 18%"
    "gst_rate": re.compile(r"GST\s*(\d{1,2}(?:\.\d+)?)\s*%", flags=re.IGNORECASE),


    # --- Other Patterns ---
    "place_of_supply": re.compile(r"Place of Supply\s*[:\s-]+([A-Za-z0-9\s,-]+)(?:\s*\(?State Code\s*[:\s-]*(\d{2})\)?)?", flags=re.IGNORECASE), # Keep previous enhancement
    "state_code_pattern": re.compile(r"(?:State Code|State\s*[:\s-])\s*(\d{1,2})"), # Keep as is
    "page_number_pattern": re.compile(r"(?:Page|Pg\.?)\s*(\d+)\s*(?:of|/|-)\s*(\d+)", re.IGNORECASE), # Keep as is

    # --- Camelot Column Keywords (Updated) ---
    "desc_kw": re.compile(r"Description|Item name|Details|Particulars", flags=re.IGNORECASE), # DESCRIPTION is present
    "qty_kw": re.compile(r"Qty|Quantity|Nos?", flags=re.IGNORECASE), # QTY is present
    "unit_price_kw": re.compile(r"Unit Price|Rate|Price/\s?unit", flags=re.IGNORECASE), # Missing column, pattern won't match
    "line_total_kw": re.compile(r"(?<!Taxable\s)Amount|Total(?!\sTax)|Line Total|Net Amount|Value|TOTAL\s*AMOUNT", flags=re.IGNORECASE), # Added TOTAL AMOUNT
    "hsn_sac_kw": re.compile(r"hsn|sac|hsn/\s?sac|HSN\s*CODE", flags=re.IGNORECASE), # Added HSN CODE
}

# Mapping from Textract keys (lowercase, simplified) to our schema keys
TEXTRACT_KEY_SYNONYMS = {
    # Keep existing synonyms...
    'invoice number': 'invoice_number', 'invoice no': 'invoice_number', 'inv no': 'invoice_number', 'invoice #': 'invoice_number',
    'invoice date': 'invoice_date', 'inv date': 'invoice_date', 'bill date': 'invoice_date', 'date': 'invoice_date',
    'supplier gstin': 'gstin_supplier', 'gstin': 'gstin_supplier', 'gst registration no': 'gstin_supplier',
    'recipient gstin': 'gstin_recipient', 'customer gstin': 'gstin_recipient', 'bill to gstin': 'gstin_recipient',
    'total amount': 'final_amount', 'grand total': 'final_amount', 'net amount': 'final_amount', 'amount due': 'final_amount', 'balance due': 'final_amount',
    'taxable value': 'taxable_value', 'subtotal': 'taxable_value', 'sub total': 'taxable_value',
    'total tax': 'tax_amount', 'gst amount': 'tax_amount',
    'sgst': 'sgst_amount', 'cgst': 'cgst_amount', 'igst': 'igst_amount',
    'sgst rate': 'sgst_rate', 'cgst rate': 'cgst_rate', 'igst rate': 'igst_rate',
    'place of supply': 'place_of_supply',
}

# Schema Keys (ensure all are present in final output)
SCHEMA_KEYS = ["taxable_value", "sgst_amount", "cgst_amount", "igst_amount",
               "sgst_rate", "cgst_rate", "igst_rate", "tax_amount", "tax_rate",
               "final_amount", "invoice_number", "invoice_date", "place_of_supply",
               "place_of_origin", "gstin_supplier", "gstin_recipient", "line_items",
               "_pdf_page_start", "_pdf_page_end"] # Add page info

# --- Helper Functions ---

def clean_text(text):
    # Keep existing clean_text
    if text is None: return None
    text = str(text)
    text = re.sub(r'(\r\n|\n|\r)+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_amount(text):
    # Keep existing parse_amount
    if text is None: return None
    text = str(text)
    cleaned = re.sub(r"[₹$€£,\s]", "", text)
    cleaned = re.sub(r"(\d)\s+(\d)", r"\1\2", cleaned)
    if cleaned.endswith('-'): cleaned = '-' + cleaned[:-1]
    try:
        return float(cleaned)
    except ValueError:
        cleaned = re.sub(r"[^0-9.-]", "", cleaned)
        try:
            if cleaned in ['-', '.', '.-', '-.']: return None
            return float(cleaned)
        except ValueError: return None

def parse_quantity(text):
    """
    Extracts the leading numeric part from a quantity string (e.g., "200Set" -> 200).
    Uses parse_amount for robustness after initial extraction.
    """
    if text is None: return None
    text = str(text).strip()
    # Find the first sequence of digits, possibly with decimals or commas
    match = re.match(r"([\d,]+(?:\.\d+)?)", text)
    if match:
        num_str = match.group(1)
        # Use parse_amount to handle commas and convert to float/int
        return parse_amount(num_str)
    else:
        # Fallback: Try parsing the whole string if no leading number found
        # This might handle cases where the number isn't at the very start
        # but relies on parse_amount stripping non-numeric chars correctly.
        return parse_amount(text)

def parse_date_robust(text):
    # Keep existing parse_date_robust
    if not text: return None
    try:
        return date_parser.parse(text, fuzzy=False).date()
    except (ValueError, OverflowError, TypeError):
        try:
            cleaned_text = text.replace('l','1').replace('O','0').replace('S','5')
            return date_parser.parse(cleaned_text, fuzzy=False).date()
        except: return None

def find_best_match(pattern, text, group_index=1, find_all=False):
    # Keep existing find_best_match
    try:
        matches = pattern.finditer(text)
        results = []
        for match in matches:
            if match and len(match.groups()) >= group_index:
                value = clean_text(match.group(group_index))
                if value: results.append(value)
        if not results: return None
        return results if find_all else results[0]
    except Exception: return None

def find_gstin_regex(text):
    # Keep existing find_gstin_regex
    supplier_gstin, recipient_gstin = None, None
    gstin_matches = list(PATTERNS["gstin_pattern"].finditer(text))
    if not gstin_matches: return None, None
    max_search_distance = 150
    found_gstins = set()
    for match in gstin_matches:
        gstin = match.group(1)
        if gstin in found_gstins: continue
        start_pos = max(0, match.start() - max_search_distance)
        search_area = text[start_pos : match.start()]
        if PATTERNS["supplier_keywords"].search(search_area):
            if not supplier_gstin: supplier_gstin = gstin; found_gstins.add(gstin)
        elif PATTERNS["recipient_keywords"].search(search_area):
            if not recipient_gstin: recipient_gstin = gstin; found_gstins.add(gstin)
    unique_gstins = [m.group(1) for m in gstin_matches if m.group(1) not in found_gstins]
    if not supplier_gstin and unique_gstins: supplier_gstin = unique_gstins.pop(0); found_gstins.add(supplier_gstin)
    if not recipient_gstin and unique_gstins: recipient_gstin = unique_gstins.pop(0); found_gstins.add(recipient_gstin)
    return supplier_gstin, recipient_gstin

def extract_place_regex(text, gstin):
    # Keep existing extract_place_regex
    if gstin and len(gstin) >= 2: pass
    match = PATTERNS["place_of_supply"].search(text)
    if match:
        place = clean_text(match.group(1))
        state_code = match.group(2)
        if state_code: return f"{place} (State Code: {state_code})"
        return place
    return None

# --- MODIFIED Confidence Calculation (No Math Check) ---
def calculate_confidence(value, base_confidence, format_valid):
    """Calculates confidence score (no math check parameter)."""
    if value is None: return 0.0
    if not isinstance(base_confidence, (int, float)) or not (0 <= base_confidence <= 1):
        base_confidence = 0.5 # Default fallback confidence (slightly lower now)

    score = float(base_confidence)
    if format_valid is True: score += BONUS_FORMAT_VALID
    elif format_valid is False: score += PENALTY_FORMAT_FAIL
    # Removed Math Check Bonus/Penalty
    return max(0.0, min(1.0, score))

# --- PDF Type Detection (Checks only first few pages for speed) ---
def detect_pdf_type(doc):
    """Classifies PDF as 'text' or 'image' based on extractable text from first few pages."""
    full_text = ""
    is_text_based = False
    try:
        # Check only first 3 pages for faster detection
        num_pages_to_check = min(3, len(doc))
        for i in range(num_pages_to_check):
            page = doc.load_page(i)
            full_text += page.get_text("text", sort=True)

        full_text = clean_text(full_text)

        if full_text and len(full_text) >= MIN_TEXT_LENGTH_FOR_TEXT_PDF:
            text_lower = full_text.lower()
            keyword_count = sum(1 for keyword in REQUIRED_KEYWORDS_FOR_TEXT_PDF if keyword in text_lower)
            if keyword_count >= 2:
                is_text_based = True
    except Exception as e:
        logging.error(f"Error during PDF type detection: {e}")
        is_text_based = False
        full_text = ""

    pdf_type = "text" if is_text_based else "image"
    logging.info(f"Detected PDF type based on first {num_pages_to_check} pages: {pdf_type}")
    return pdf_type

# --- Page Segmentation Logic ---
def get_page_identifiers(page):
    """Extracts potential invoice identifiers from a single page's text."""
    text = page.get_text("text", sort=True)
    invoice_num = find_best_match(PATTERNS["invoice_number"], text)
    invoice_date = find_best_match(PATTERNS["invoice_date"], text)
    # Extract first GSTIN found as potential supplier indicator for segmentation
    gstin_match = PATTERNS["gstin_pattern"].search(text)
    supplier_gstin = gstin_match.group(1) if gstin_match else None
    # Check for page number pattern
    page_num_match = PATTERNS["page_number_pattern"].search(text)
    page_info = {"current": None, "total": None}
    if page_num_match:
        try:
            page_info["current"] = int(page_num_match.group(1))
            page_info["total"] = int(page_num_match.group(2))
        except ValueError:
            pass # Ignore if conversion fails

    return {
        "invoice_number": invoice_num,
        "invoice_date": invoice_date,
        "supplier_gstin": supplier_gstin,
        "page_info": page_info,
        "has_text": bool(text and len(text.strip()) > 10) # Basic check if page has meaningful text
    }

def segment_pdf_into_invoices(doc):
    """
    Analyzes pages to identify boundaries between distinct invoices.

    Returns:
        list: A list of lists, where each inner list contains the
              page indices (0-based) belonging to a detected invoice.
              e.g., [[0, 1], [2], [3, 4, 5]]
    """
    page_count = len(doc)
    if page_count == 0:
        return []
    if page_count == 1:
        return [[0]] # Single page is always one invoice

    page_data = [get_page_identifiers(doc.load_page(i)) for i in range(page_count)]

    invoice_segments = []
    current_segment = [0] # Start with the first page
    last_identifier_page_index = 0 # Track the last page with a useful identifier

    for i in range(1, page_count):
        current_page_info = page_data[i]
        prev_page_key_info = page_data[last_identifier_page_index]

        # --- Heuristics for starting a new invoice segment ---
        new_invoice = False

        # 1. Explicit Page Numbering (Strongest Indicator when present and consistent)
        # If page says "Page 1 of X" and previous wasn't part of an ongoing sequence, it's likely new.
        if current_page_info["page_info"]["current"] == 1 and current_page_info["page_info"]["total"] is not None:
             # Check if previous page was the *last* page of its sequence or had no sequence info
            prev_page_num_info = prev_page_key_info.get("page_info", {"current": None, "total": None})
            if prev_page_num_info["current"] is None or prev_page_num_info["current"] == prev_page_num_info["total"]:
                 logging.info(f"Segment Split Trigger: Page {i} detected as 'Page 1 of X'.")
                 new_invoice = True

        # 2. Change in Primary Identifier (Invoice Number)
        # Only trigger if the *current* page has an invoice number AND it's different from the *last known* number
        elif current_page_info["invoice_number"] is not None and \
             prev_page_key_info["invoice_number"] is not None and \
             current_page_info["invoice_number"] != prev_page_key_info["invoice_number"]:
            logging.info(f"Segment Split Trigger: Invoice number changed from '{prev_page_key_info['invoice_number']}' to '{current_page_info['invoice_number']}' on page {i}.")
            new_invoice = True

        # 3. Fallback: Change in Supplier GSTIN (if invoice number was missing on previous key page)
        elif current_page_info["invoice_number"] is None and \
             current_page_info["supplier_gstin"] is not None and \
             prev_page_key_info["supplier_gstin"] is not None and \
             prev_page_key_info["invoice_number"] is None and \
             current_page_info["supplier_gstin"] != prev_page_key_info["supplier_gstin"]:
             logging.info(f"Segment Split Trigger: Supplier GSTIN changed (and no inv#) on page {i}.")
             new_invoice = True

        # --- Logic ---
        if new_invoice:
            invoice_segments.append(current_segment) # Finalize the previous segment
            current_segment = [i] # Start a new segment
            last_identifier_page_index = i # Update the reference page
        else:
            current_segment.append(i) # Add page to the current segment
            # Update last_identifier_page_index only if the current page has *some* key info
            if current_page_info["invoice_number"] or current_page_info["supplier_gstin"] or current_page_info["invoice_date"]:
                 last_identifier_page_index = i

    # Add the last segment
    if current_segment:
        invoice_segments.append(current_segment)

    logging.info(f"Detected {len(invoice_segments)} invoice segment(s): {invoice_segments}")
    return invoice_segments


# --- Text-Based Path Functions (Modified for Page List) ---

def map_camelot_columns(df):
    # Keep existing map_camelot_columns (No changes needed here)
    # ... (previous implementation) ...
    if df.empty or len(df) < 1:
        # logging.warning("map_camelot_columns received empty or header-only DataFrame.") # Less verbose
        return {}
    header_row = df.iloc[0]
    headers_cleaned = header_row.astype(str).str.lower().str.strip()
    headers_cleaned = headers_cleaned.str.replace(r'\s{2,}', ' ', regex=True)
    headers_cleaned = headers_cleaned.str.replace(r'\s*/\s*', '/', regex=True)
    cols = df.columns
    mapping = {}
    assigned_cols_indices = set()
    col_candidates = {
        "description": (PATTERNS["desc_kw"], None), "quantity": (PATTERNS["qty_kw"], parse_quantity),
        "unit_price": (PATTERNS["unit_price_kw"], parse_amount), "line_total": (PATTERNS["line_total_kw"], parse_amount),
        "hsn_sac": (re.compile(r"hsn|sac", re.IGNORECASE), None),
    }
    potential_line_total_indices, potential_qty_indices, potential_price_indices = [], [], []

    # Stage 1: Header Keyword Matching
    # logging.debug(f"Cleaned Headers: {headers_cleaned.tolist()}") # Debug level
    for c_idx, header_text in enumerate(headers_cleaned):
        if c_idx in assigned_cols_indices: continue
        matched_key = None
        for key, (pattern, _) in col_candidates.items():
            if key in ["line_total"]: continue
            if pattern.search(header_text) and key not in mapping:
                 if key in ["quantity", "unit_price"] and ("cgst" in header_text or "sgst" in header_text or "tax" in header_text):
                     # logging.debug(f"Skipping potential '{key}' mapping for header '{header_text}' due to tax keyword.") # Debug
                     continue
                 # logging.debug(f"Mapping '{key}' to column index {c_idx} based on header '{header_text}'") # Debug
                 mapping[key] = cols[c_idx]
                 assigned_cols_indices.add(c_idx)
                 matched_key = key
                 break
        if not matched_key:
             if col_candidates["quantity"][0].search(header_text): potential_qty_indices.append(c_idx)
             elif col_candidates["unit_price"][0].search(header_text): potential_price_indices.append(c_idx)
             elif col_candidates["line_total"][0].search(header_text):
                  is_tax_amount_col = "cgst" in header_text or "sgst" in header_text or "tax" in header_text
                  if not is_tax_amount_col:
                       potential_line_total_indices.append(c_idx)
                       # logging.debug(f"Identified potential 'line_total' column index {c_idx} (header: '{header_text}')") # Debug
                  # else: logging.debug(f"Ignoring potential 'line_total' column index {c_idx} (header: '{header_text}') due to tax keywords.") # Debug

    # Stage 2: Resolve Ambiguity
    if "quantity" not in mapping and len(potential_qty_indices) == 1:
         idx = potential_qty_indices[0]
         if idx not in assigned_cols_indices: # logging.debug(f"Assigning 'quantity' to column index {idx} (unique candidate).");
             mapping["quantity"] = cols[idx]; assigned_cols_indices.add(idx)
    if "unit_price" not in mapping and len(potential_price_indices) == 1:
        idx = potential_price_indices[0]
        if idx not in assigned_cols_indices: # logging.debug(f"Assigning 'unit_price' to column index {idx} (unique candidate).");
             mapping["unit_price"] = cols[idx]; assigned_cols_indices.add(idx)
    if "line_total" not in mapping and potential_line_total_indices:
        if len(potential_line_total_indices) == 1:
            idx = potential_line_total_indices[0]
            if idx not in assigned_cols_indices: # logging.debug(f"Assigning 'line_total' to column index {idx} (unique candidate).");
                mapping["line_total"] = cols[idx]; assigned_cols_indices.add(idx)
        else:
            potential_line_total_indices.sort(reverse=True)
            # logging.debug(f"Multiple 'line_total' candidates: {potential_line_total_indices}. Trying rightmost.") # Debug
            for idx in potential_line_total_indices:
                if idx not in assigned_cols_indices:
                    # logging.debug(f"Assigning 'line_total' to column index {idx} (rightmost available candidate).") # Debug
                    mapping["line_total"] = cols[idx]; assigned_cols_indices.add(idx); break

    # Stage 3: Content Analysis Fallback (Simplified for brevity, can be enhanced)
    if len(df) > 1 and any(key not in mapping for key in ["quantity", "unit_price", "line_total"]):
        # logging.debug("Mapping incomplete, attempting content analysis fallback...") # Debug
        data_df = df.iloc[1:]; unassigned_indices = [i for i, col in enumerate(cols) if i not in assigned_cols_indices]
        numeric_col_scores = {}
        for idx in unassigned_indices:
             col_name = cols[idx];
             try:
                  numeric_ratio = data_df[col_name].apply(lambda x: pd.notna(parse_amount(x))).mean()
                  if numeric_ratio > 0.6: numeric_col_scores[idx] = numeric_ratio; # logging.debug(f"Content Analysis: Col {idx} is {numeric_ratio*100:.1f}% numeric.") # Debug
             except Exception: continue
        sorted_numeric_indices = sorted(numeric_col_scores.keys())
        if "line_total" not in mapping and sorted_numeric_indices:
            potential_total_idx = sorted_numeric_indices[-1]; # logging.debug(f"Content Analysis: Assigning 'line_total' to rightmost numeric col {potential_total_idx}.") # Debug
            mapping["line_total"] = cols[potential_total_idx]; assigned_cols_indices.add(potential_total_idx); sorted_numeric_indices.pop()
        if "unit_price" not in mapping and sorted_numeric_indices:
             potential_price_idx = sorted_numeric_indices[-1]; # logging.debug(f"Content Analysis: Assigning 'unit_price' to next rightmost numeric col {potential_price_idx}.") # Debug
             mapping["unit_price"] = cols[potential_price_idx]; assigned_cols_indices.add(potential_price_idx); sorted_numeric_indices.pop()
        if "quantity" not in mapping and sorted_numeric_indices:
             potential_qty_idx = sorted_numeric_indices[0]; # logging.debug(f"Content Analysis: Assigning 'quantity' to leftmost remaining numeric col {potential_qty_idx}.") # Debug
             mapping["quantity"] = cols[potential_qty_idx]; assigned_cols_indices.add(potential_qty_idx)

    essential_keys = ["description", "quantity", "unit_price", "line_total"]
    # if not all(key in mapping for key in essential_keys): # logging.warning(f"Failed to map all essential line item columns. Found: {list(mapping.keys())}") # Warning

    # logging.debug(f"Final Camelot column mapping: {mapping}") # Debug
    return mapping


# --- Modified Camelot extraction to accept page list ---
def extract_line_items_camelot(pdf_path, page_list):
    """Extracts line items using Camelot from specified pages."""
    line_items_lattice = []
    line_items_stream = []
    tables_lattice = []
    tables_stream = []

    # Convert page list (0-based index) to Camelot page string (1-based index)
    pages_str = ','.join(map(lambda x: str(x + 1), page_list))
    logging.info(f"Camelot: Attempting extraction from pages: {pages_str}")

    logging.info("Camelot: Trying 'lattice' flavor...")
    try:
        tables_lattice = camelot.read_pdf(pdf_path, pages=pages_str, flavor='lattice', suppress_stdout=True, line_scale=40)
    except Exception as e:
        logging.warning(f"Camelot lattice error: {e}") # Changed to warning

    logging.info("Camelot: Trying 'stream' flavor...")
    try:
        tables_stream = camelot.read_pdf(pdf_path, pages=pages_str, flavor='stream', suppress_stdout=True, edge_tol=50, row_tol=10) # Adjusted tolerance
    except Exception as e:
        logging.warning(f"Camelot stream error: {e}") # Changed to warning

    if tables_lattice:
        logging.info(f"Camelot lattice found {tables_lattice.n} tables.")
        line_items_lattice = process_camelot_tables(tables_lattice, flavor='lattice')

    if tables_stream:
        logging.info(f"Camelot stream found {tables_stream.n} tables.")
        line_items_stream = process_camelot_tables(tables_stream, flavor='stream')

    # --- Evaluation Logic (Choose the best result) ---
    score_lattice = sum(1 for item in line_items_lattice if item.get("line_total", {}).get("value") is not None)
    score_stream = sum(1 for item in line_items_stream if item.get("line_total", {}).get("value") is not None)

    logging.info(f"Camelot Evaluation: Lattice Score={score_lattice}, Stream Score={score_stream}")
    if score_stream > score_lattice:
        logging.info("Choosing Stream results.")
        return line_items_stream
    elif score_lattice > 0:
         logging.info("Choosing Lattice results.")
         return line_items_lattice
    else:
         logging.warning("Camelot found no usable line items with either flavor.") # Warning
         return []


# --- Modified Camelot table processing (No major logic change, but remove math check use) ---
def process_camelot_tables(tables, flavor):
    """Processes tables extracted by Camelot. No longer uses cross-total math checks."""
    all_line_items = []
    col_candidates = {
        "description": (PATTERNS["desc_kw"], None), "quantity": (PATTERNS["qty_kw"], parse_quantity),
        "unit_price": (PATTERNS["unit_price_kw"], parse_amount), "line_total": (PATTERNS["line_total_kw"], parse_amount),
        "hsn_sac": (PATTERNS["hsn_sac_kw"], None),
    }
    item_schema_keys = ["description", "quantity", "unit_price", "line_total", "hsn_sac"]
    EXCLUSION_KEYWORDS = {
        "cgst", "sgst", "igst", "gst total", "total", "subtotal", "sub total",
        "sub-total", "gross total", "amount chargeable", "total amount", "taxable value"
    }
    REQUIRED_FIELDS_FOR_LINE_ITEM = {"description", "line_total"} # Relaxed requirement slightly

    logging.info(f"Processing {tables.n} tables found with flavor='{flavor}'...")
    for i, table in enumerate(tables):
        df = table.df
        # logging.info(f"\nProcessing Table {i+1}/{tables.n} (Shape: {df.shape})") # Too verbose for INFO
        if df.empty: continue
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        if df.empty: continue

        header_row_index = df.index[0]
        data_df = df.drop(header_row_index).reset_index(drop=True)
        if data_df.empty: continue

        col_mapping = map_camelot_columns(df.copy())
        if not col_mapping: continue

        # logging.info(f"Table {i+1}: Processing {len(data_df)} data rows with mapping: {col_mapping}") # Too verbose
        for idx, row in data_df.iterrows():
            item_data = {key: {"value": None, "confidence": 0.0, "trusted": False} for key in item_schema_keys}
            raw_vals = {}
            parsed_vals = {}
            source_conf = table.parsing_report.get('accuracy', 70) / 100.0 # Camelot accuracy
            all_required_fields_present = True

            # Extract and Parse Raw Values
            for key, col_idx in col_mapping.items():
                if key not in col_candidates or col_idx not in row.index:
                    if key in REQUIRED_FIELDS_FOR_LINE_ITEM: all_required_fields_present = False
                    continue
                raw_val = row[col_idx]; parser_func = col_candidates[key][1]
                parsed_val = parser_func(raw_val) if parser_func else clean_text(raw_val)
                parsed_vals[key] = parsed_val
                if parsed_val is not None: raw_vals[key] = parsed_val
                if key in REQUIRED_FIELDS_FOR_LINE_ITEM and parsed_val is None:
                     if not (key == 'description' and isinstance(parsed_val, str)): # Allow empty desc
                          # logging.debug(f"Debug Row {idx}: Required field '{key}' is None after parsing '{raw_val}'.") # Debug
                          all_required_fields_present = False

            # Apply Filtering Logic
            skip_row = False; reason = ""
            desc_text = parsed_vals.get("description", "")
            if isinstance(desc_text, str):
                desc_lower = desc_text.lower().strip()
                if desc_lower in EXCLUSION_KEYWORDS: skip_row, reason = True, f"Exclusion keyword '{desc_text}'."
                else:
                    for keyword in EXCLUSION_KEYWORDS:
                        if keyword in desc_lower and len(desc_lower) < len(keyword) + 10:
                             if f" {keyword} " in f" {desc_lower} " or desc_lower.startswith(keyword) or desc_lower.endswith(keyword):
                                  skip_row, reason = True, f"Contains exclusion keyword '{keyword}'."
                                  break
            elif "description" in REQUIRED_FIELDS_FOR_LINE_ITEM: skip_row, reason = True, "Required 'description' missing/not text."

            if not skip_row and not all_required_fields_present:
                missing_required = [req for req in REQUIRED_FIELDS_FOR_LINE_ITEM if req not in parsed_vals or parsed_vals[req] is None]
                if "description" in missing_required and isinstance(parsed_vals.get("description"), str): missing_required.remove("description")
                if missing_required: skip_row, reason = True, f"Missing required fields: {missing_required}"

            if not skip_row and "description" in parsed_vals and isinstance(parsed_vals["description"], str) and len(parsed_vals["description"].strip()) < 3:
                 other_values_present = any(k in parsed_vals and parsed_vals[k] is not None for k in ["quantity", "unit_price", "line_total", "hsn_sac"])
                 if not other_values_present: skip_row, reason = True, "Short description, no other values."

            if skip_row: # logging.debug(f"Skipping Row {idx}: {reason}"); # Debug
                continue
            # logging.debug(f"Processing Row {idx} as potential line item.") # Debug

            # Populate item_data
            for key in item_schema_keys:
                 if key in parsed_vals: item_data[key]["value"] = parsed_vals[key]

            # --- Line Item Internal Math Check (Qty * Price vs Line Total) --- REMOVED external checks
            qty, price, total = raw_vals.get("quantity"), raw_vals.get("unit_price"), raw_vals.get("line_total")
            line_item_math_passed = None # Internal check for Q*P=T
            if qty is not None and price is not None and total is not None:
                try:
                    line_item_math_passed = abs((qty * price) - total) < LINE_ITEM_QTY_PRICE_TOLERANCE
                except TypeError: line_item_math_passed = False

            # Calculate Final Confidence and Trust for each item field
            for key in item_data.keys():
                value = item_data[key]["value"]
                format_valid = False
                if value is not None:
                   if key in ["quantity", "unit_price", "line_total"]: format_valid = isinstance(value, (int, float))
                   elif key in ["description", "hsn_sac"]: format_valid = isinstance(value, str) and len(value) > 0
                   else: format_valid = True

                # Confidence now only based on source, format, and internal Q*P=T check (if applicable)
                base_field_conf = source_conf # Start with Camelot's table accuracy
                final_conf = calculate_confidence(value, base_field_conf, format_valid)

                # Apply small bonus/penalty for internal line item math check where relevant
                if key in ["quantity", "unit_price", "line_total"]:
                    if line_item_math_passed is True: final_conf = min(1.0, final_conf + 0.1) # Small bonus
                    elif line_item_math_passed is False: final_conf = max(0.0, final_conf - 0.1) # Small penalty

                item_data[key]['confidence'] = round(final_conf, 3)
                item_data[key]['trusted'] = final_conf >= CONF_THRESHOLD_TRUSTED if value is not None else False

            all_line_items.append(item_data)

    logging.info(f"Finished processing tables with flavor='{flavor}'. Extracted {len(all_line_items)} potential line items.")
    return all_line_items

# --- Modified Text-based processing to accept full_text and page list ---
def process_text_based_pdf(pdf_path, full_text, page_list):
    """Extracts data using Regex (on full text) and Camelot (on specific pages)."""
    results = {} # Raw extracted values from Regex on combined text

    # Extract header/footer fields using Regex on the combined text of the segment
    results["invoice_number"] = find_best_match(PATTERNS["invoice_number"], full_text)
    results["invoice_date"] = find_best_match(PATTERNS["invoice_date"], full_text)
    results["gstin_supplier"], results["gstin_recipient"] = find_gstin_regex(full_text)
    results["place_of_supply"] = extract_place_regex(full_text, results.get("gstin_recipient"))
    results["place_of_origin"] = extract_place_regex(full_text, results.get("gstin_supplier")) # Less common
    results["taxable_value"] = find_best_match(PATTERNS["taxable_value"], full_text)
    results["sgst_amount"] = find_best_match(PATTERNS["sgst_amount"], full_text)
    results["cgst_amount"] = find_best_match(PATTERNS["cgst_amount"], full_text)
    results["igst_amount"] = find_best_match(PATTERNS["igst_amount"], full_text)
    results["tax_amount"] = find_best_match(PATTERNS["tax_amount"], full_text)
    results["final_amount"] = find_best_match(PATTERNS["final_amount"], full_text)
    results["sgst_rate"] = find_best_match(PATTERNS["sgst_rate"], full_text)
    results["cgst_rate"] = find_best_match(PATTERNS["cgst_rate"], full_text)
    results["igst_rate"] = find_best_match(PATTERNS["igst_rate"], full_text)
    multi_rates = find_best_match(PATTERNS["rate_pattern"], full_text, find_all=True)
    results["tax_rate"] = [parse_amount(r) for r in multi_rates if parse_amount(r) is not None] if multi_rates else None

    # Extract line items using Camelot only on the pages for this segment
    line_items = extract_line_items_camelot(pdf_path, page_list)

    return results, line_items


# --- Image-Based Path Functions (AWS Textract) ---

def get_textract_block_text(block, blocks_map):
    # Keep existing get_textract_block_text
    text = ""
    if 'Relationships' in block:
        for relationship in block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    child_block = blocks_map.get(child_id)
                    if child_block and child_block['BlockType'] == 'WORD':
                        text += child_block['Text'] + ' '
    elif 'Text' in block: text = block['Text']
    return text.strip()

# --- Modified Textract parsing to filter by page ---
def parse_textract_response(response, target_pages):
    """Parses Textract response, filtering blocks for target pages."""
    all_blocks = response.get('Blocks', [])
    target_page_set = set(target_pages) # Use 1-based page numbers for Textract

    # Filter blocks by page number *before* processing
    blocks = [block for block in all_blocks if block.get('Page') in target_page_set]
    if not blocks:
         logging.warning(f"Textract: No blocks found for target pages {target_pages}.")
         return {}, []

    logging.info(f"Textract: Processing {len(blocks)} blocks relevant to pages {target_pages} (out of {len(all_blocks)} total blocks).")

    blocks_map = {block['Id']: block for block in blocks} # Map only relevant blocks
    key_value_pairs = {}
    tables = []

    # Extract Key-Value pairs (from filtered blocks)
    key_blocks = {b_id: b for b_id, b in blocks_map.items() if b['BlockType'] == 'KEY'}
    value_blocks = {b_id: b for b_id, b in blocks_map.items() if b['BlockType'] == 'VALUE'}
    # KVS sets link KEY and VALUE blocks which should already be filtered by page
    key_value_set_blocks = [b for b in blocks if b['BlockType'] == 'KEY_VALUE_SET']

    for kv_set in key_value_set_blocks:
        key_text, value_text = "", ""
        key_conf, value_conf = 0.0, 0.0
        key_id, value_id = None, None

        # Find linked KEY and VALUE blocks
        if 'Relationships' in kv_set:
            for rel in kv_set['Relationships']:
                if rel['Type'] == 'KEY' and rel['Ids']: key_id = rel['Ids'][0]
                elif rel['Type'] == 'VALUE' and rel['Ids']: value_id = rel['Ids'][0]

        # Ensure the linked KEY/VALUE blocks are in our filtered set
        key_block_content = key_blocks.get(key_id)
        value_block_content = value_blocks.get(value_id)

        if key_block_content:
             key_text = get_textract_block_text(key_block_content, blocks_map)
             key_conf = key_block_content.get('Confidence', 0.0) / 100.0
        if value_block_content:
            value_text = get_textract_block_text(value_block_content, blocks_map)
            value_conf = value_block_content.get('Confidence', 0.0) / 100.0

        if key_text:
            avg_conf = (key_conf + value_conf) / 2.0 if key_conf > 0 and value_conf > 0 else max(key_conf, value_conf)
            key_value_pairs[clean_text(key_text.lower())] = {"value": value_text, "confidence": avg_conf}

    # Extract Tables (from filtered blocks)
    table_blocks = [b for b in blocks if b['BlockType'] == 'TABLE']
    for table_block in table_blocks:
        table_data = []
        cell_blocks = {}
        if 'Relationships' in table_block:
            for rel in table_block['Relationships']:
                if rel['Type'] == 'CHILD':
                    for cell_id in rel['Ids']:
                        # Ensure the linked CELL block is in our filtered map
                        cell_block = blocks_map.get(cell_id)
                        if cell_block and cell_block['BlockType'] == 'CELL':
                             cell_blocks[cell_id] = cell_block

        rows = defaultdict(lambda: defaultdict(dict))
        max_row, max_col = 0, 0
        for cell_id, cell_block in cell_blocks.items():
             row_idx, col_idx = cell_block['RowIndex'], cell_block['ColumnIndex']
             max_row, max_col = max(max_row, row_idx), max(max_col, col_idx)
             cell_text = get_textract_block_text(cell_block, blocks_map)
             cell_conf = cell_block.get('Confidence', 0.0) / 100.0
             rows[row_idx][col_idx] = {"text": cell_text, "confidence": cell_conf}

        if rows:
            table_matrix = []
            for r in range(1, max_row + 1):
                row_list = [rows.get(r, {}).get(c, {"text": "", "confidence": 0.0}) for c in range(1, max_col + 1)]
                table_matrix.append(row_list)
            tables.append({"matrix": table_matrix, "confidence": table_block.get('Confidence', 0.0)/100.0})

    return key_value_pairs, tables

# --- Modified Textract mapping (No major change, but remove math check use) ---
def map_textract_output(key_value_pairs, tables):
    """Maps parsed Textract KVs and Tables to schema. No longer uses cross-total math checks."""
    results = defaultdict(lambda: {"value": None, "confidence": 0.0})
    line_items = []

    # Map Key-Value Pairs (Keep existing logic)
    processed_gstins = 0
    for raw_key, data in key_value_pairs.items():
        matched = False
        for synonym, schema_key in TEXTRACT_KEY_SYNONYMS.items():
             if synonym in raw_key:
                 if schema_key == 'gstin_supplier' and results['gstin_supplier']['value'] is None:
                     if PATTERNS['gstin_pattern'].search(data['value']):
                         results['gstin_supplier'] = data; processed_gstins += 1; matched = True; break
                 elif schema_key == 'gstin_recipient' and results['gstin_recipient']['value'] is None:
                     if PATTERNS['gstin_pattern'].search(data['value']):
                         if results['gstin_supplier']['value'] is not None or processed_gstins == 1:
                             results['gstin_recipient'] = data; matched = True; break
                 elif 'gstin' not in schema_key:
                      if results[schema_key]['confidence'] < data['confidence']: results[schema_key] = data
                      matched = True; break
        if not matched and 'gstin' in raw_key and PATTERNS['gstin_pattern'].search(data['value']):
            if results['gstin_supplier']['value'] is None: results['gstin_supplier'] = data; processed_gstins += 1
            elif results['gstin_recipient']['value'] is None: results['gstin_recipient'] = data

    # Map Tables to Line Items
    col_candidates = {
        "description": (PATTERNS["desc_kw"], None), "quantity": (PATTERNS["qty_kw"], parse_quantity),
        "unit_price": (PATTERNS["unit_price_kw"], parse_amount), "line_total": (PATTERNS["line_total_kw"], parse_amount),
    }
    item_schema_keys = ["description", "quantity", "unit_price", "line_total"] # Simplified for example
    REQUIRED_FIELDS_FOR_LINE_ITEM = {"description", "line_total"} # Relaxed

    for table in tables:
        matrix = table['matrix']
        if not matrix or len(matrix) < 2: continue

        header_row_data = matrix[0]; data_rows = matrix[1:]
        col_mapping_idx = {}; assigned_cols_idx = set()
        for c_idx, cell_data in enumerate(header_row_data):
            header_text = clean_text(cell_data['text']).lower()
            if c_idx in assigned_cols_idx: continue
            for key, (pattern, _) in col_candidates.items():
                if key not in col_mapping_idx and pattern.search(header_text):
                    col_mapping_idx[key] = c_idx; assigned_cols_idx.add(c_idx); break
        # logging.debug(f"Textract table column mapping: {col_mapping_idx}") # Debug
        if not col_mapping_idx or not any(k in col_mapping_idx for k in REQUIRED_FIELDS_FOR_LINE_ITEM): continue

        for row_data in data_rows:
             item = {key: {"value": None, "confidence": 0.0, "trusted": False} for key in item_schema_keys}
             raw_vals = {}; parsed_vals = {}
             all_required_present_parsed = True

             for key, col_idx in col_mapping_idx.items():
                 if key not in item_schema_keys: continue # Only map schema keys
                 if col_idx < len(row_data):
                     cell_data = row_data[col_idx]; raw_val = cell_data['text']
                     base_conf = cell_data['confidence'] # Textract cell confidence
                     parser_func = col_candidates[key][1]
                     parsed_val = parser_func(raw_val) if parser_func else clean_text(raw_val)

                     format_valid = (parsed_val is not None) if key != "description" else (isinstance(parsed_val, str) and len(parsed_val) > 0)
                     conf = calculate_confidence(parsed_val, base_conf, format_valid) # No math check here

                     item[key] = {"value": parsed_val, "confidence": round(conf, 3)} # Trust TBD later
                     raw_vals[key] = parsed_val # Store parsed for internal math check
                     parsed_vals[key] = parsed_val
                     if key in REQUIRED_FIELDS_FOR_LINE_ITEM and parsed_val is None:
                         if not (key == 'description' and isinstance(parsed_val, str)):
                              all_required_present_parsed = False
                 elif key in REQUIRED_FIELDS_FOR_LINE_ITEM: all_required_present_parsed = False # Column doesn't exist

             # Skip row if required fields are missing after parsing
             if not all_required_present_parsed:
                 # logging.debug(f"Skipping Textract row: Missing required fields after parsing.") # Debug
                 continue

             # Perform internal line item math check (Q*P=T)
             qty, price, total = raw_vals.get("quantity"), raw_vals.get("unit_price"), raw_vals.get("line_total")
             line_item_math_passed = None
             if qty is not None and price is not None and total is not None:
                 try: line_item_math_passed = abs((qty * price) - total) < LINE_ITEM_QTY_PRICE_TOLERANCE
                 except TypeError: line_item_math_passed = False

             # Re-calculate confidence incorporating internal math check & apply trust
             valid_item = False
             for key in item.keys():
                 if key not in item_schema_keys: continue
                 current_conf = item[key]['confidence']
                 # Apply small bonus/penalty for internal check
                 if key in ["quantity", "unit_price", "line_total"]:
                     if line_item_math_passed is True: current_conf = min(1.0, current_conf + 0.1)
                     elif line_item_math_passed is False: current_conf = max(0.0, current_conf - 0.1)

                 item[key]['confidence'] = round(current_conf, 3)
                 item[key]['trusted'] = current_conf >= CONF_THRESHOLD_TRUSTED if item[key]['value'] is not None else False
                 if item[key]['value'] is not None and key in REQUIRED_FIELDS_FOR_LINE_ITEM:
                      valid_item = True # Mark as valid if at least one required field has data

             if valid_item: # Append if any required field was populated
                 line_items.append(item)

    return dict(results), line_items


# --- Modified Image-based processing to pass Textract response and page list ---
def process_image_based_pdf(textract_response, page_list_1_based):
    """Processes pre-fetched Textract response for specific pages."""
    results_mapped = {}
    line_items = []
    try:
        # Parse the relevant parts of the already fetched response
        key_value_pairs, tables = parse_textract_response(textract_response, page_list_1_based)
        if not key_value_pairs and not tables:
             logging.warning("Textract parsing yielded no K/V pairs or Tables for the target pages.")
             # Return empty structure
             results_mapped = {key: {"value": None, "confidence": 0.0} for key in SCHEMA_KEYS if key not in ['line_items', '_pdf_page_start', '_pdf_page_end']}
             line_items = []
        else:
             results_mapped, line_items = map_textract_output(key_value_pairs, tables)

    except Exception as e:
        logging.error(f"Error during Textract result processing: {e}", exc_info=True)
        # Return empty results on error
        results_mapped = {key: {"value": None, "confidence": 0.0} for key in SCHEMA_KEYS if key not in ['line_items', '_pdf_page_start', '_pdf_page_end']}
        line_items = []

    return results_mapped, line_items


# --- Main Hybrid Orchestration (Modified for Segmentation) ---

def process_invoice_segment(pdf_path, pdf_doc, page_list_0_based, pdf_type, textract_response=None):
    """Processes a single detected invoice segment (specific pages)."""
    segment_start_page = page_list_0_based[0] + 1 # 1-based for output
    segment_end_page = page_list_0_based[-1] + 1   # 1-based for output
    logging.info(f"Processing segment: Pages {segment_start_page}-{segment_end_page}, Type: {pdf_type}")

    final_output = defaultdict(lambda: {"value": None, "confidence": 0.0, "trusted": False})
    final_output["line_items"] = []
    final_output["_pdf_page_start"] = segment_start_page
    final_output["_pdf_page_end"] = segment_end_page

    raw_results = {}
    line_items_extracted = []
    base_confidences = {} # Store base confidences

    if pdf_type == "text":
        logging.info("Using Text-Based Path for segment (PyMuPDF + Regex + Camelot)")
        # Combine text only from relevant pages
        segment_text = ""
        for page_index in page_list_0_based:
            try:
                 segment_text += pdf_doc.load_page(page_index).get_text("text", sort=True) + "\n"
            except Exception as e:
                 logging.warning(f"Could not extract text from page {page_index + 1}: {e}")
        full_text = clean_text(segment_text)

        if not full_text:
             logging.warning(f"Segment pages {segment_start_page}-{segment_end_page} had no extractable text.")
        else:
            raw_results, line_items_extracted = process_text_based_pdf(pdf_path, full_text, page_list_0_based)
            for key in raw_results.keys():
                 base_confidences[key] = BASE_CONF_PYMUPDF_TEXT
            # Camelot line items already have base confidence from parsing report

    elif pdf_type == "image":
        logging.info("Using Image-Based Path for segment (AWS Textract Results)")
        if textract_response:
            page_list_1_based = [p + 1 for p in page_list_0_based]
            # Process the *pre-fetched* Textract response, filtering by page
            raw_results_with_conf, line_items_extracted = process_image_based_pdf(textract_response, page_list_1_based)

            for key, data in raw_results_with_conf.items():
                 raw_results[key] = data.get('value') # Use .get for safety
                 base_confidences[key] = data.get('confidence', 0.0) # Use .get for safety
            # Textract line items already processed with confidence
        else:
             logging.error("Textract response missing for image-based segment.")
             # Handle error - return empty structure?
             for key in SCHEMA_KEYS: final_output[key] # Initialize empty
             return dict(final_output)

    else:
        logging.error(f"Unknown PDF type '{pdf_type}' for segment.")
        for key in SCHEMA_KEYS: final_output[key]
        return dict(final_output)


    # --- Common Post-Processing & Validation (Math checks removed) ---
    logging.info("Performing common post-processing for segment...")
    parsed_results = {}

    # Parse amounts and dates
    for key, raw_value in raw_results.items():
        if key == "line_items": continue
        parsed_value = raw_value; is_valid_format = None
        if raw_value is not None and raw_value != '':
            try:
                if any(k in key for k in ["amount", "value", "total"]) and key not in ["invoice_number"]:
                     parsed_value = parse_amount(raw_value); is_valid_format = (parsed_value is not None)
                elif "rate" in key and key != "tax_rate": # Handle single rates
                     parsed_value = parse_amount(raw_value); is_valid_format = (parsed_value is not None)
                elif "date" in key:
                     parsed_value = parse_date_robust(raw_value); is_valid_format = (parsed_value is not None)
                     if is_valid_format: parsed_value = parsed_value.isoformat()
                elif "gstin" in key:
                     is_valid_format = bool(parsed_value and PATTERNS["gstin_pattern"].fullmatch(str(parsed_value)))
                elif key == "tax_rate": # Handle list of rates
                    if isinstance(parsed_value, list):
                        is_valid_format = all(isinstance(r, (int, float)) for r in parsed_value)
                    else: # Try parsing if it's a single string value? Less ideal.
                        pv = parse_amount(parsed_value)
                        if pv is not None: parsed_value = [pv]; is_valid_format = True
                        else: is_valid_format = False
                else: # Text fields
                     is_valid_format = isinstance(parsed_value, str) and len(parsed_value) > 0
            except Exception as e:
                logging.warning(f"Parsing error for key '{key}', value '{raw_value}': {e}")
                parsed_value = raw_value # Keep raw value on error
                is_valid_format = False # Mark as invalid format
        elif raw_value == '': # Treat empty strings as None for consistency, invalid format
             parsed_value = None
             is_valid_format = False

        parsed_results[key] = {"value": parsed_value, "format_valid": is_valid_format}

    # Mathematical Cross-Validation REMOVED

    # Final Confidence Calculation and Trust Assignment
    logging.info("Calculating final confidence and trust for segment...")
    for key in SCHEMA_KEYS:
        # Skip keys managed elsewhere
        if key in ['line_items', '_pdf_page_start', '_pdf_page_end']:
            continue

        data = parsed_results.get(key)
        if data:
            value = data['value']
            format_valid = data['format_valid']
            # Use stored base conf, default 0.5 if missing
            base_conf = base_confidences.get(key, 0.5)

            # Math check REMOVED from here
            final_conf = calculate_confidence(value, base_conf, format_valid)
            trusted = final_conf >= CONF_THRESHOLD_TRUSTED if value is not None else False

            final_output[key] = {"value": value, "confidence": round(final_conf, 3), "trusted": trusted}
        else:
            # Ensure key exists even if not found by extraction paths
            final_output[key] = {"value": None, "confidence": 0.0, "trusted": False}

    # Assign processed line items
    final_output['line_items'] = line_items_extracted # Assign already processed items

    return dict(final_output)


# --- New Top-Level Orchestrator ---
def extract_invoice_data_from_pdf(pdf_path):
    """
    Top-level function to handle PDF segmentation and processing.
    Returns a list of dictionaries, one for each detected invoice.
    """
    logging.info(f"\n--- Processing {os.path.basename(pdf_path)} ---")
    start_time = time.time()
    all_invoice_results = []
    doc = None
    textract_response = None # Store if fetched

    try:
        doc = fitz.open(pdf_path)
        if not doc or len(doc) == 0:
            logging.error("PDF is empty or could not be opened.")
            return [{"_filename": os.path.basename(pdf_path), "_error": "Empty or invalid PDF"}]

        # 1. Detect PDF Type (Text or Image)
        pdf_type = detect_pdf_type(doc)

        # 2. Segment PDF into potential invoices
        invoice_segments_pages = segment_pdf_into_invoices(doc) # List of lists of 0-based indices

        # 3. If Image type, call Textract ONCE for the whole document
        if pdf_type == "image":
            logging.info("Image PDF detected. Calling AWS Textract for the entire document...")
            try:
                textract_client = boto3.client('textract', region_name=AWS_REGION)
                with open(pdf_path, 'rb') as document_bytes:
                    image_bytes = document_bytes.read()
                textract_response = textract_client.analyze_document(
                    Document={'Bytes': image_bytes},
                    FeatureTypes=['FORMS', 'TABLES']
                )
                logging.info("Received response from Textract.")
            except Exception as e:
                logging.error(f"AWS Textract API call failed: {e}", exc_info=True)
                # Cannot proceed with image path if Textract fails
                return [{
                    "_filename": os.path.basename(pdf_path),
                    "_error": f"Textract API Error: {e}",
                    "_pdf_page_start": 1,
                    "_pdf_page_end": len(doc)
                }]


        # 4. Process each segment
        for i, segment_pages in enumerate(invoice_segments_pages):
            logging.info(f"--- Starting processing for Invoice Segment {i+1}/{len(invoice_segments_pages)} (Pages {min(segment_pages)+1}-{max(segment_pages)+1}) ---")
            segment_result = process_invoice_segment(
                pdf_path,
                doc,
                segment_pages, # 0-based indices
                pdf_type,
                textract_response # Pass full response if image type
            )
            segment_result['_filename'] = os.path.basename(pdf_path)
            segment_result['_invoice_index_in_pdf'] = i + 1 # Add index
            all_invoice_results.append(segment_result)
            logging.info(f"--- Finished processing for Invoice Segment {i+1}/{len(invoice_segments_pages)} ---")


    except Exception as e:
        logging.error(f"!!!!!!!! Critical error processing {os.path.basename(pdf_path)}: {e} !!!!!!!!", exc_info=True)
        # Add a placeholder error result for the whole file
        all_invoice_results.append({
            "_filename": os.path.basename(pdf_path),
            "_error": f"Critical processing error: {e}",
            "_pdf_page_start": 1,
            "_pdf_page_end": len(doc) if doc else 'N/A'
        })
    finally:
        if doc:
            doc.close() # Ensure PDF is closed

    end_time = time.time()
    logging.info(f"--- Finished processing {os.path.basename(pdf_path)} in {end_time - start_time:.2f} seconds ---")
    return all_invoice_results


# --- Main Execution ---
if __name__ == "__main__":
    pdf_directory = "./testing"  # Default directory
    output_file_json = "invoice_extraction_hybrid_results.json"
    output_file_excel = "invoice_extraction_hybrid_results.xlsx"

    if not os.path.isdir(pdf_directory):
        logging.error(f"Error: Directory not found: {pdf_directory}")
        exit()
    if not AWS_REGION:
         logging.error("Error: AWS_REGION is not set in the script configuration.")
         exit()

    all_results_flat = [] # Will store one dict per detected invoice
    for filename in sorted(os.listdir(pdf_directory)):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            # extract_invoice_data_from_pdf returns a LIST of results (one per invoice)
            invoice_results_list = extract_invoice_data_from_pdf(filepath)
            all_results_flat.extend(invoice_results_list) # Add results to the flat list

    # Save results to JSON
    try:
        with open(output_file_json, 'w', encoding='utf-8') as f:
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (datetime.date, datetime.datetime)): return obj.isoformat()
                    # Add handling for defaultdict if it sneaks in
                    if isinstance(obj, defaultdict): return dict(obj)
                    try:
                        return super().default(obj)
                    except TypeError:
                        return str(obj) # Fallback to string representation
            json.dump(all_results_flat, f, indent=4, ensure_ascii=False, cls=CustomEncoder)
        logging.info(f"\nExtraction complete. Results saved to {output_file_json}")
    except Exception as e: logging.error(f"\nError saving results to JSON: {e}", exc_info=True)

    # Save results to Excel
    try:
        excel_rows = []
        for res in all_results_flat:
             row = {
                 "filename": res.get("_filename", "N/A"),
                 "invoice_index": res.get("_invoice_index_in_pdf", 1),
                 "page_start": res.get("_pdf_page_start", "N/A"),
                 "page_end": res.get("_pdf_page_end", "N/A"),
                 "error": res.get("_error") # Will be None if no error
            }
             # Add other schema keys, handling potential errors if 'res' is just an error dict
             for key in SCHEMA_KEYS:
                 if key not in ['_pdf_page_start', '_pdf_page_end']: # Already handled
                     data = res.get(key)
                     if key == 'line_items':
                         # Ensure line items are JSON serializable even if extraction failed partially
                         try: row[key] = json.dumps(data, cls=CustomEncoder) if data is not None else '[]'
                         except Exception: row[key] = '[]' # Default to empty list string on error
                     elif isinstance(data, dict): # Expected structure
                         row[f"{key}_value"] = data.get("value")
                         row[f"{key}_confidence"] = data.get("confidence")
                         row[f"{key}_trusted"] = data.get("trusted")
                     else: # Handle cases where data might be missing or is part of an error dict
                          row[f"{key}_value"] = None
                          row[f"{key}_confidence"] = 0.0
                          row[f"{key}_trusted"] = False
             excel_rows.append(row)

        df_excel = pd.DataFrame(excel_rows)
        # Define desired column order
        cols_order = ['filename', 'invoice_index', 'page_start', 'page_end', 'error']
        value_cols = []
        conf_cols = []
        trust_cols = []
        other_cols = []

        for skey in SCHEMA_KEYS:
             if skey not in ['_pdf_page_start', '_pdf_page_end']:
                 if skey == 'line_items':
                     other_cols.append(skey)
                 else:
                    value_cols.append(f"{skey}_value")
                    conf_cols.append(f"{skey}_confidence")
                    trust_cols.append(f"{skey}_trusted")

        # Combine ordered columns
        cols_order.extend(value_cols)
        cols_order.extend(conf_cols)
        cols_order.extend(trust_cols)
        cols_order.extend(other_cols) # Add line_items at the end

        # Filter to only include columns present in the DataFrame
        final_cols = [c for c in cols_order if c in df_excel.columns]
        df_excel = df_excel[final_cols]

        df_excel.to_excel(output_file_excel, index=False)
        logging.info(f"Results also saved to Excel: {output_file_excel}")
    except Exception as e: logging.error(f"Could not save results to Excel: {e}", exc_info=True)