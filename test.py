import fitz  # PyMuPDF
# import pytesseract # Keep for potential future use or simple tasks # Removed as not used
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
import numpy as np # For metrics calculation if needed

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') # Use for detailed debug

# --- Configuration ---

# AWS Textract Configuration
AWS_REGION = "ap-south-1"  # <--- CHANGE THIS TO YOUR AWS REGION

# Confidence Thresholds & Weights (Math checks NOW included in validation)
CONF_THRESHOLD_TRUSTED = 0.80 # Base confidence threshold (one signal for trust)
BASE_CONF_PYMUPDF_TEXT = 0.90 # High base confidence for clean digital text
# Textract provides its own confidences, use them as base
BONUS_FORMAT_VALID = 0.10 # Bonus remains
PENALTY_FORMAT_FAIL = -0.30 # Penalty remains

# --- NEW Validation & Metrics Constants ---
# GSTIN Checksum validation constants
GSTIN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_CHECK_DIGIT_INDEX = 14

# Date Plausibility
MAX_YEARS_IN_FUTURE = 1
MAX_YEARS_IN_PAST = 10

# Standard GST Rates (India) - Using floats requires tolerance or careful comparison
VALID_GST_RATES = {5.0, 12.0, 18.0, 28.0, 2.5, 6.0, 9.0, 14.0, 0.0} # Added 0%
RATE_COMPARISON_TOLERANCE = 0.01 # For float comparisons

# Math Check Tolerance
TOTAL_MATH_TOLERANCE = 0.50 # Tolerance for Taxable + Tax = Final check (adjust as needed)
LINE_ITEM_QTY_PRICE_TOLERANCE = 0.10 # Keep tolerance for internal line item check

# Metrics Calculation
GROUND_TRUTH_CSV = "actual.csv" # Name of your ground truth file
METRICS_FLOAT_TOLERANCE = 0.01 # Tolerance for comparing float values in metrics
METRICS_DATE_FIELDS = {'invoice_date'}
METRICS_FLOAT_FIELDS = {'taxable_value', 'sgst_amount', 'cgst_amount', 'igst_amount',
                        'sgst_rate', 'cgst_rate', 'igst_rate', 'tax_amount',
                        'final_amount'} # Add others if needed
METRICS_GSTIN_FIELDS = {'gstin_supplier', 'gstin_recipient'}
METRICS_LIST_FIELDS = {'tax_rate'} # Fields that might be lists

# PDF Type Detection Heuristics
MIN_TEXT_LENGTH_FOR_TEXT_PDF = 150
REQUIRED_KEYWORDS_FOR_TEXT_PDF = ['invoice', 'total', 'date', 'bill', 'amount']


# --- Define Helper Regex String Variables FIRST ---
amount_capture_decimal_required = r"(?:[₹$€£*]\s*)?([\d,]+\.\d{1,2})"
amount_capture_optional_decimal = r"(?:[₹$€£*]\s*)?([\d,.]*\d)"
rate_inside_brackets_capture = r"\(?\s*(\d{1,2}(?:\.\d{1,2})?)\s*%\s*\)?"
rate_after_at_capture = r"@\s*(\d{1,2}(?:\.\d{1,2})?)\s*%"
state_code_in_pos_pattern = re.compile(r'\(?\s*(?:State\s*Code)?\s*[:\s-]*(\d{2})\s*\)?')
rate_fragment = r"(?:(?:@\s*\d{1,2}(?:\.\d{1,2})?%)|(?:\(\s*\d{1,2}(?:\.\d{1,2})?%\s*\)))"
separator_fragment = r"(?:\s+)|(?:\s*[:₹]\s*)" # Ensure at least one space OR symbols like : ₹

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
    # MODIFIED: Allow optional @Rate% after keyword, use decimal required capture


    "sgst_amount": re.compile(
        r"(?:SGST|S\.G\.S\.T\.)"                     # Keyword
        r"(?:\d*\s*" + rate_fragment + r"?\s*)?"     # Optional digits (like 9) + Optional Rate fragment + Optional space(s)
        r"[:\s-]*?"                                  # Minimal separator (non-greedy) << REVERTED
        + amount_capture_optional_decimal,           # Capture the amount
        flags=re.IGNORECASE
    ),
    "cgst_amount": re.compile(
        r"(?:CGST|C\.G\.S\.T\.)"                     # Keyword
        r"(?:\d*\s*" + rate_fragment + r"?\s*)?"     # Optional digits (like 9) + Optional Rate fragment + Optional space(s)
        r"[:\s-]*?"                                  # Minimal separator (non-greedy) << REVERTED
        + amount_capture_optional_decimal,           # Capture the amount
        flags=re.IGNORECASE
    ),
    "igst_amount": re.compile(
        r"(?:IGST|I\.G\.S\.T\.)"                     # Keyword
        r"(?:\s*" + rate_fragment + r"?\s*)?"        # Optional Rate fragment + Optional space(s)
        r"[:\s-]*?"                                  # Minimal separator (non-greedy) << REVERTED
        + amount_capture_optional_decimal,           # Capture the amount
        flags=re.IGNORECASE
    ),
    # --- Line Capture Patterns (Fallback) ---
    "sgst_amount_line": re.compile(
        r"(?:SGST|S\.G\.S\.T\.)"            # Keyword
        r"(?:@\d{1,2}(?:\.\d{1,2})?%)?"    # Optional @Rate%
        r"\s+"                             # Require at least one space separator
        r"(.+)",                           # Capture the rest of the potential line/segment
        flags=re.IGNORECASE
    ),
    "cgst_amount_line": re.compile(
        r"(?:CGST|C\.G\.S\.T\.)"            # Keyword
        r"(?:@\d{1,2}(?:\.\d{1,2})?%)?"    # Optional @Rate%
        r"\s+"                             # Require at least one space separator
        r"(.+)",                           # Capture the rest of the potential line/segment
        flags=re.IGNORECASE
    ),
    # Keep IGST line pattern if needed
    "igst_amount_line": re.compile(
        r"(?:IGST|I\.G\.S\.T\.)"            # Keyword
        r"(?:@\d{1,2}(?:\.\d{1,2})?%)?" 
        r"\s+"                              # Separator: One or more spaces REQUIRED
        r"(.+)", 
        flags=re.IGNORECASE
    ),

    # --- Total Patterns ---
    # MODIFIED: Taxable value uses optional decimal capture (updated to handle *)
    "taxable_value": re.compile(
        # Match "Taxable Value", "Base Amount", or "Subtotal"/"Sub Total" NOT followed by "(Tax Inclusive)"
        r"(?:Taxable Value|Taxable Amount|Base Amount|Sub(?:-| )?total(?!.*\(\s*Tax Inclusive\s*\)))" +
        r"\s*[:\s-]*" +
        amount_capture_optional_decimal,
        flags=re.IGNORECASE
    ),

    "final_amount": re.compile(
        # Match common total keywords OR "Sub Total (Tax Inclusive)" OR "Total" not preceded by "Sub" etc.
        r"(?:Total Amount|Grand Total|Net Amount|Amount Due|Total Payable|Balance Due|Final Net Amount|Sub\s*Total\s*\(Tax Inclusive\)|(?<!Sub\s|Taxable\s)Total)" +
        r"\s*[:\s-]*" +
        amount_capture_optional_decimal,
        flags=re.IGNORECASE
    ),

    "tax_amount": re.compile(
        r"(?:Total Tax|Tax Amount|Total GST|GST Amount|GST\s*\d{1,2}(?:\.\d+)?%|^GST$|^TAX$)"
        # Minimal separator allows flexibility but might be prone to capturing rate if amount isn't clearly separated
        r"\s*[:\s-]*?"                               # Minimal separator (non-greedy) << REVERTED
        + amount_capture_optional_decimal,
        flags=re.IGNORECASE | re.MULTILINE
    ),

    "final_amount": re.compile(r"(?:Total Amount|Grand Total|Net Amount|Amount Due|Total Payable|Balance Due|Final Net Amount|(?<!Sub\s)Total)" +
                               r"\s*[:\s-]*" +
                               amount_capture_optional_decimal, # Keep OPTIONAL decimal
                               flags=re.IGNORECASE),
    # Define a simple pattern to find a decimal number (used by helper)
    "decimal_number_capture": re.compile(amount_capture_decimal_required), # Uses the helper string directly


    # --- Rate Patterns ---
    "rate_pattern": re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s*%"),
    # MODIFIED: Use rate_after_at_capture for SGST/CGST rates like @9%
    "sgst_rate": re.compile(r"(?:SGST|S\.G\.S\.T\.)\s*(?:" + rate_after_at_capture + r"|" + rate_inside_brackets_capture + r")", flags=re.IGNORECASE),
    "cgst_rate": re.compile(r"(?:CGST|C\.G\.S\.T\.)\s*(?:" + rate_after_at_capture + r"|" + rate_inside_brackets_capture + r")", flags=re.IGNORECASE),
    "igst_rate": re.compile(r"(?:IGST|I\.G\.S\.T\.)\s*(?:" + rate_after_at_capture + r"|" + rate_inside_brackets_capture + r")", flags=re.IGNORECASE),
    "gst_rate": re.compile(r"GST\s*(\d{1,2}(?:\.\d+)?)\s*%", flags=re.IGNORECASE),


    # --- Other Patterns ---
    "place_of_supply": re.compile(r"Place of Supply\s*[:\s-]+([A-Za-z0-9\s,()-]+)(?:\s*\(?State Code\s*[:\s-]*(\d{2})\)?)?", flags=re.IGNORECASE), # Capture potential code in main group too
    "state_code_pattern": re.compile(r"(?:State Code|State\s*[:\s-])\s*(\d{1,2})"), # Keep as is
    "page_number_pattern": re.compile(r"(?:Page|Pg\.?)\s*(\d+)\s*(?:of|/|-)\s*(\d+)", re.IGNORECASE), # Keep as is

    # --- Camelot Column Keywords (Updated) ---
    "desc_kw": re.compile(r"Description|Item name|Details|Particulars", flags=re.IGNORECASE), # DESCRIPTION is present
    "qty_kw": re.compile(r"Qty|Quantity|Nos?", flags=re.IGNORECASE), # QTY is present
    "unit_price_kw": re.compile(r"Unit Price|Rate|Price/\s?unit", flags=re.IGNORECASE), # Missing column, pattern won't match
    "line_total_kw": re.compile(r"(?<!Taxable\s)Amount$|^Amount$|Total(?!\sTax)|Line Total|Net Amount|Value|TOTAL\s*AMOUNT", flags=re.IGNORECASE),
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
    # (Keep existing implementation)
    if text is None: return None
    text = str(text).strip()
    match = re.match(r"([\d,]+(?:\.\d+)?)", text)
    if match:
        num_str = match.group(1)
        return parse_amount(num_str)
    else:
        return parse_amount(text)


def parse_date_robust(text):
    # Keep existing parse_date_robust
    if not text: return None
    try:
        # Try parsing without fuzzy first for common formats
        dt = date_parser.parse(text, fuzzy=False)
        # Handle cases where year might be picked incorrectly (e.g., '24' -> 2024)
        # If year seems too far off, maybe try dayfirst heuristic?
        # This part can get complex, keeping it simple for now.
        return dt.date()
    except (ValueError, OverflowError, TypeError):
        try:
            # Simple cleaning, could be expanded
            cleaned_text = text.replace('l','1').replace('O','0').replace('S','5')
            # Try fuzzy as a fallback
            dt = date_parser.parse(cleaned_text, fuzzy=True)
            # Add a basic check: if fuzzy parsing results in today's date from random text, ignore it
            if dt.date() == datetime.date.today() and len(text) < 5: # Arbitrary length check
                return None
            return dt.date()
        except (ValueError, OverflowError, TypeError):
            logging.debug(f"Robust date parsing failed for: '{text}'")
            return None

FIND_AMOUNT_IN_SEGMENT_PATTERN = re.compile(
    r"([\d,]+\.\d{1,2})" +  # Prefer numbers with decimals
    r"|" +                 # OR
    r"(?<!%)(\b\d[\d,]*\b)(?!%)" # Match whole integer numbers not adjacent to %
)

# Let's update the find_decimal_in_text helper (rename?) to use this new pattern
def find_amount_in_line_fallback(text_segment, prefer_last=False):
    """
    Finds the most likely currency amount within a text segment using a specific pattern.
    Prioritizes numbers with decimals.
    """
    if not text_segment: return None
    pattern = FIND_AMOUNT_IN_SEGMENT_PATTERN
    matches = pattern.finditer(text_segment)
    found_amounts = []
    for match in matches:
        # Group 1 is decimal amount, Group 2 is integer amount
        amount_str = match.group(1) or match.group(2)
        if amount_str:
            cleaned_amount = clean_text(amount_str)
            if cleaned_amount: # Ensure it's not empty after cleaning
                 # Basic check: Avoid single digits unless they are the only option?
                 # if len(cleaned_amount) == 1 and '.' not in cleaned_amount and any(c.isdigit() for c in text_segment if c != cleaned_amount):
                 #      continue # Skip potential rate match if other digits exist
                 found_amounts.append(cleaned_amount)

    if not found_amounts: return None

    # Prioritize decimal values if available
    decimal_amounts = [a for a in found_amounts if '.' in a]
    if decimal_amounts:
        return decimal_amounts[-1] if prefer_last else decimal_amounts[0]
    else: # Only integers found
        return found_amounts[-1] if prefer_last else found_amounts[0]


def find_best_match(pattern, text, group_index=1, find_all=False, prefer_last=False):
    """
    Finds matches for a regex pattern in text and returns results based on preferences.

    Args:
        pattern (re.Pattern): Compiled regex pattern.
        text (str): Text to search within.
        group_index (int): The index of the capturing group containing the desired value.
                           Defaults to 1.
        find_all (bool): If True, returns a list of all unique, cleaned values found.
                         If False, returns a single "best" match based on other flags.
                         Defaults to False.
        prefer_last (bool): If True and find_all is False, returns the *last* valid match
                            found in the text instead of the first. Useful for totals
                            that might appear multiple times. Defaults to False.

    Returns:
        str or list or None:
            - If find_all=True: A list of unique, cleaned string values found (or empty list).
            - If find_all=False: A single cleaned string value representing the best match
                                 (first or last based on prefer_last), or None if no match.
            - None: If an error occurs during regex processing or no matches are found.
    """
    if not text or not pattern:
        return [] if find_all else None

    try:
        matches = pattern.finditer(text)
        extracted_values = []
        for match in matches:
            # Ensure the match object is valid and has enough groups
            if match and len(match.groups()) >= group_index:
                # Check if the target group actually captured something (not None)
                captured_value = match.group(group_index)
                if captured_value is not None:
                    value = clean_text(captured_value)
                    # Ensure cleaning didn't result in an empty string
                    if value:
                        extracted_values.append(value)

        if not extracted_values:
            # Return empty list if find_all is True and no results, else None
            return [] if find_all else None

        if find_all:
            # Return unique values if finding all
            return list(dict.fromkeys(extracted_values)) # Efficient way to get unique values preserving order
        else:
            # Return a single best match
            if prefer_last:
                # Return the last valid match found
                return extracted_values[-1]
            else:
                # Return the first valid match found (default behavior)
                return extracted_values[0]

    except re.error as regex_err:
        # Catch specific regex errors if possible
        logging.error(f"Regex syntax error in pattern '{pattern.pattern}': {regex_err}")
        return [] if find_all else None
    except IndexError as idx_err:
         logging.error(f"Regex group index {group_index} out of bounds for pattern '{pattern.pattern}': {idx_err}")
         return [] if find_all else None
    except Exception as e:
        # Log other potential errors during matching
        logging.error(f"Error in find_best_match for pattern '{pattern.pattern}' (Group: {group_index}): {e}", exc_info=True) # Added exc_info for more detail
        return [] if find_all else None


def find_gstin_regex(text):
    # Keep existing find_gstin_regex
    supplier_gstin, recipient_gstin = None, None
    gstin_matches = list(PATTERNS["gstin_pattern"].finditer(text))
    if not gstin_matches: return None, None
    max_search_distance = 150
    found_gstins = set()
    for match in gstin_matches:
        gstin = match.group(1).upper() # Standardize to uppercase
        if gstin in found_gstins: continue
        start_pos = max(0, match.start() - max_search_distance)
        search_area = text[start_pos : match.start()]
        if PATTERNS["supplier_keywords"].search(search_area):
            if not supplier_gstin: supplier_gstin = gstin; found_gstins.add(gstin)
        elif PATTERNS["recipient_keywords"].search(search_area):
            if not recipient_gstin: recipient_gstin = gstin; found_gstins.add(gstin)

    # Fallback: Assign remaining unique GSTINs based on order found
    unique_gstins = [m.group(1).upper() for m in gstin_matches if m.group(1).upper() not in found_gstins]
    if not supplier_gstin and unique_gstins: supplier_gstin = unique_gstins.pop(0); found_gstins.add(supplier_gstin)
    if not recipient_gstin and unique_gstins: recipient_gstin = unique_gstins.pop(0); found_gstins.add(recipient_gstin)

    # If only one found, try assigning based on generic keywords in the whole text? (More complex)
    # Keeping simple assignment for now.

    return supplier_gstin, recipient_gstin


def extract_place_regex(text, gstin):
    # Keep existing logic, but ensure it captures state code if present
    # The updated PATTERNS["place_of_supply"] regex already tries to capture the code in group 2
    match = PATTERNS["place_of_supply"].search(text)
    if match:
        place = clean_text(match.group(1))
        # state_code = match.group(2) # Code specifically after "(State Code: XX)"
        # Try finding state code within the matched place text as well
        state_code = extract_state_code_from_pos(place)

        if place:
            # Optionally clean the state code part from the place name if needed
            place_cleaned = state_code_in_pos_pattern.sub('', place).strip()
            return f"{state_code}" if state_code else place_cleaned
            # return place # Return the full matched string for now
        return None # Should not happen if place matched
    # Fallback: if no explicit "Place of Supply" found, maybe use state from GSTIN?
    # This is less reliable as PoS != recipient location always.
    # if gstin and len(gstin) >= 2:
    #     state_code = gstin[:2]
    #     # Need a mapping from state code to state name here if desired
    #     return f"State Code: {state_code}" # Placeholder
    return None

# --- NEW Validation Helper Functions ---

def get_gstin_checksum(gstin_base):
    """Calculates the check digit for the first 14 chars of a GSTIN."""
    if not isinstance(gstin_base, str) or len(gstin_base) != 14:
        return None
    total = 0
    for i, char in enumerate(gstin_base):
        val = GSTIN_ALPHABET.find(char)
        if val == -1: return None # Invalid character
        factor = 2 if (i + 1) % 2 == 0 else 1
        product = val * factor
        total += (product // len(GSTIN_ALPHABET)) + (product % len(GSTIN_ALPHABET))
    check_digit_index = (len(GSTIN_ALPHABET) - (total % len(GSTIN_ALPHABET))) % len(GSTIN_ALPHABET)
    return GSTIN_ALPHABET[check_digit_index]

def is_valid_gstin_checksum(gstin_str):
    """Validates a 15-character GSTIN using the checksum algorithm."""
    if not isinstance(gstin_str, str) or len(gstin_str) != 15:
        return False
    gstin_str = gstin_str.upper()
    base = gstin_str[:GSTIN_CHECK_DIGIT_INDEX]
    provided_checksum = gstin_str[GSTIN_CHECK_DIGIT_INDEX]
    calculated_checksum = get_gstin_checksum(base)
    if calculated_checksum is None:
        return False # Invalid characters in base
    return calculated_checksum == provided_checksum

def is_plausible_date(date_obj):
    """Checks if a date is within a plausible range (not too far future/past)."""
    if not isinstance(date_obj, datetime.date):
        return False # Not a valid date object
    today = datetime.date.today()
    future_limit = today + datetime.timedelta(days=MAX_YEARS_IN_FUTURE * 366) # Approximate
    past_limit = today - datetime.timedelta(days=MAX_YEARS_IN_PAST * 366)   # Approximate
    return past_limit <= date_obj <= future_limit

def is_valid_gst_rate(rate_val):
    """Checks if a rate value matches one of the standard GST rates within tolerance."""
    if not isinstance(rate_val, (int, float)):
        return False
    # Check if rate is close to any valid rate
    return any(abs(rate_val - valid_rate) < RATE_COMPARISON_TOLERANCE for valid_rate in VALID_GST_RATES)

def extract_state_code_from_pos(pos_string):
    """Extracts the first 2-digit number assumed to be state code from Place of Supply string."""
    if not pos_string or not isinstance(pos_string, str):
        return None
    match = state_code_in_pos_pattern.search(pos_string)
    if match:
        return match.group(1)
    # Fallback: look for any 2-digit number in the string? Might be risky.
    # match_any = re.search(r'\b(\d{2})\b', pos_string)
    # if match_any: return match_any.group(1)
    return None

def check_pos_vs_recipient_state(recipient_gstin, place_of_supply_str):
    """
    Compares state code from recipient GSTIN (first 2 digits) with
    state code extracted from Place of Supply string.
    Returns: True (match), False (mismatch), None (cannot compare).
    """
    if not recipient_gstin or len(recipient_gstin) < 2 or not place_of_supply_str:
        return None

    recipient_state_code = recipient_gstin[:2]
    pos_state_code = extract_state_code_from_pos(place_of_supply_str)

    if not pos_state_code:
        logging.debug("Could not extract state code from Place of Supply string.")
        return None # Cannot determine PoS state code

    if not recipient_state_code.isdigit() or not pos_state_code.isdigit():
         logging.debug("Non-digit state codes found.")
         return None # Should be digits

    match = (recipient_state_code == pos_state_code)
    logging.debug(f"PoS vs Recipient State Check: GSTIN Code={recipient_state_code}, PoS Code={pos_state_code}, Match={match}")
    return match

def check_rate_tax_math(parsed_vals_dict):
    """
    Checks if Taxable Value * Rate ≈ Specific Tax Amount within tolerance.
    Returns:
        dict: A dictionary indicating pass/fail/not_applicable for each tax type
              e.g., {'sgst_ok': True, 'cgst_ok': False, 'igst_ok': 'not_applicable'}
              Overall result is True if all applicable checks pass, False if any fail.
    """
    taxable_val = parsed_vals_dict.get('taxable_value', {}).get('value')
    results = {'sgst_ok': 'not_applicable', 'cgst_ok': 'not_applicable', 'igst_ok': 'not_applicable', 'overall_ok': None}
    checks_performed = 0
    checks_failed = 0

    if not isinstance(taxable_val, (int, float)):
        logging.debug("Rate-Tax Math Check: Taxable value missing or not numeric.")
        results['overall_ok'] = None # Cannot perform check
        return results

    # Check SGST
    sgst_rate = parsed_vals_dict.get('sgst_rate', {}).get('value')
    sgst_amount = parsed_vals_dict.get('sgst_amount', {}).get('value')
    if isinstance(sgst_rate, (int, float)) and isinstance(sgst_amount, (int, float)):
        checks_performed += 1
        expected_sgst = taxable_val * (sgst_rate / 100.0)
        match = abs(expected_sgst - sgst_amount) < TOTAL_MATH_TOLERANCE
        results['sgst_ok'] = match
        logging.debug(f"Rate-Tax Math Check (SGST): Taxable={taxable_val}, Rate={sgst_rate}%, Expected={expected_sgst:.2f}, Actual={sgst_amount}, Match={match}")
        if not match: checks_failed += 1
    else:
         # Log only if one part exists but not the other
         if sgst_rate is not None or sgst_amount is not None:
              logging.debug(f"Rate-Tax Math Check (SGST): Skipped (Rate: {sgst_rate}, Amount: {sgst_amount})")


    # Check CGST
    cgst_rate = parsed_vals_dict.get('cgst_rate', {}).get('value')
    cgst_amount = parsed_vals_dict.get('cgst_amount', {}).get('value')
    if isinstance(cgst_rate, (int, float)) and isinstance(cgst_amount, (int, float)):
        checks_performed += 1
        expected_cgst = taxable_val * (cgst_rate / 100.0)
        match = abs(expected_cgst - cgst_amount) < TOTAL_MATH_TOLERANCE
        results['cgst_ok'] = match
        logging.debug(f"Rate-Tax Math Check (CGST): Taxable={taxable_val}, Rate={cgst_rate}%, Expected={expected_cgst:.2f}, Actual={cgst_amount}, Match={match}")
        if not match: checks_failed += 1
    else:
         if cgst_rate is not None or cgst_amount is not None:
              logging.debug(f"Rate-Tax Math Check (CGST): Skipped (Rate: {cgst_rate}, Amount: {cgst_amount})")

    # Check IGST
    igst_rate = parsed_vals_dict.get('igst_rate', {}).get('value')
    igst_amount = parsed_vals_dict.get('igst_amount', {}).get('value')
    if isinstance(igst_rate, (int, float)) and isinstance(igst_amount, (int, float)):
        checks_performed += 1
        expected_igst = taxable_val * (igst_rate / 100.0)
        match = abs(expected_igst - igst_amount) < TOTAL_MATH_TOLERANCE
        results['igst_ok'] = match
        logging.debug(f"Rate-Tax Math Check (IGST): Taxable={taxable_val}, Rate={igst_rate}%, Expected={expected_igst:.2f}, Actual={igst_amount}, Match={match}")
        if not match: checks_failed += 1
    else:
         if igst_rate is not None or igst_amount is not None:
              logging.debug(f"Rate-Tax Math Check (IGST): Skipped (Rate: {igst_rate}, Amount: {igst_amount})")


    if checks_performed == 0:
        results['overall_ok'] = None # No checks could be performed
    else:
        results['overall_ok'] = (checks_failed == 0) # Overall OK if no checks failed

    logging.debug(f"Rate-Tax Math Overall Result: {results['overall_ok']}")
    return results

def check_total_math(parsed_vals_dict):
    """
    Checks if Taxable Value + Tax Amounts ≈ Final Amount within tolerance.
    Returns: True (match), False (mismatch), None (insufficient data).
    """
    taxable = parsed_vals_dict.get('taxable_value', {}).get('value')
    final = parsed_vals_dict.get('final_amount', {}).get('value')
    sgst = parsed_vals_dict.get('sgst_amount', {}).get('value')
    cgst = parsed_vals_dict.get('cgst_amount', {}).get('value')
    igst = parsed_vals_dict.get('igst_amount', {}).get('value')
    tax_total = parsed_vals_dict.get('tax_amount', {}).get('value')

    # Ensure required values are floats for calculation
    if not isinstance(taxable, (int, float)) or not isinstance(final, (int, float)):
        logging.debug("Total Math Check: Missing or non-numeric Taxable or Final amount.")
        return None

    calculated_tax = 0.0
    tax_components_present = False

    # Prioritize individual taxes if present
    if isinstance(sgst, (int, float)) and isinstance(cgst, (int, float)):
        calculated_tax = sgst + cgst
        tax_components_present = True
        logging.debug(f"Total Math Check: Using SGST ({sgst}) + CGST ({cgst})")
    elif isinstance(igst, (int, float)):
        calculated_tax = igst
        tax_components_present = True
        logging.debug(f"Total Math Check: Using IGST ({igst})")
    elif isinstance(tax_total, (int, float)):
        calculated_tax = tax_total
        tax_components_present = True
        logging.debug(f"Total Math Check: Using Total Tax ({tax_total})")

    if not tax_components_present:
        logging.debug("Total Math Check: No usable tax components found.")
        return None # Cannot perform check without tax value

    # Perform the check
    expected_final = taxable + calculated_tax
    is_match = abs(expected_final - final) < TOTAL_MATH_TOLERANCE
    logging.debug(f"Total Math Check: Taxable={taxable}, Calc Tax={calculated_tax}, Expected Final={expected_final}, Actual Final={final}, Match={is_match}")
    return is_match


# --- Confidence Calculation (Modified slightly if needed, or kept as is) ---
def calculate_confidence(value, base_confidence, format_valid):
    """Calculates confidence score based on base, format. (No math check parameter)."""
    if value is None: return 0.0
    if not isinstance(base_confidence, (int, float)) or not (0 <= base_confidence <= 1):
        base_confidence = 0.5 # Default fallback

    score = float(base_confidence)
    if format_valid is True: score += BONUS_FORMAT_VALID
    elif format_valid is False: score += PENALTY_FORMAT_FAIL
    return max(0.0, min(1.0, score))


# --- NEW Multi-Signal Trust Logic ---
def determine_trust(field_key, value, confidence, validation_signals):
    """
    Determines the 'trusted' flag based on confidence and validation signals.

    Args:
        field_key (str): The schema key of the field.
        value: The parsed value of the field.
        confidence (float): The calculated confidence score (0.0-1.0).
        validation_signals (dict): A dictionary containing boolean results
                                   from various validation checks like
                                   {'format_valid': True, 'gstin_checksum': True, ...}.

    Returns:
        bool: True if the value is considered trusted, False otherwise.
    """
    if value is None:
        return False # Cannot trust a missing value

    # Base requirement: Decent confidence and valid format
    if confidence < (CONF_THRESHOLD_TRUSTED - 0.2): # Lower initial bar? Or keep strict?
         logging.debug(f"Trust({field_key}): False (Low confidence: {confidence:.2f})")
         return False
    if not validation_signals.get('format_valid', False):
         logging.debug(f"Trust({field_key}): False (Invalid format)")
         return False

    # Apply specific checks based on field type
    trusted = True # Assume trusted unless a check fails

    if field_key in ('gstin_supplier', 'gstin_recipient'):
        checksum_ok = validation_signals.get('gstin_checksum', {}).get(field_key, None)
        if checksum_ok is False:
            logging.debug(f"Trust({field_key}): False (GSTIN Checksum failed)")
            trusted = False
        # Note: PoS vs State check failure might weaken trust but not outright reject?
        # Let's keep it simple: checksum failure is enough to distrust.

    elif field_key == 'invoice_date':
        plausible = validation_signals.get('date_plausible', None)
        if plausible is False:
            logging.debug(f"Trust({field_key}): False (Date not plausible)")
            trusted = False

    elif field_key in ('sgst_rate', 'cgst_rate', 'igst_rate'):
        rate_ok = validation_signals.get('gst_rate_valid', {}).get(field_key, None)
        if rate_ok is False:
            logging.debug(f"Trust({field_key}): False (Invalid standard GST rate)")
            trusted = False

    elif field_key == 'tax_rate': # Check list of rates
         rates_ok = validation_signals.get('gst_rate_valid', {}).get(field_key, None) # Expect True/False for the list
         if rates_ok is False:
             logging.debug(f"Trust({field_key}): False (One or more invalid standard GST rates in list)")
             trusted = False

    # Apply cross-field validation results (Total Math Check)
    # This check influences trust of multiple related fields
    # --- NEW: Rate * Taxable = Tax Amount Check ---
    # This check affects taxable_value, rates, and tax amounts
    if field_key in ('taxable_value', 'sgst_rate', 'sgst_amount', 'cgst_rate', 'cgst_amount', 'igst_rate', 'igst_amount'):
        rate_tax_math_results = validation_signals.get('rate_tax_math_results', {})
        overall_rate_math_ok = rate_tax_math_results.get('overall_ok')

        if overall_rate_math_ok is False:
            # If the overall check failed, distrust the specific component if its individual check failed
            check_failed_specifically = False
            if field_key in ('sgst_rate', 'sgst_amount') and rate_tax_math_results.get('sgst_ok') is False:
                check_failed_specifically = True
            elif field_key in ('cgst_rate', 'cgst_amount') and rate_tax_math_results.get('cgst_ok') is False:
                check_failed_specifically = True
            elif field_key in ('igst_rate', 'igst_amount') and rate_tax_math_results.get('igst_ok') is False:
                check_failed_specifically = True
            elif field_key == 'taxable_value': # Taxable value is involved in all failed checks
                 check_failed_specifically = True # Distrust taxable if *any* rate check fails

            if check_failed_specifically:
                trusted = False
                logging.debug(f"Trust({field_key}): False (Rate-Tax Math Check failed for this component)")


    # Apply PoS vs State check result
    if field_key == 'place_of_supply':
         pos_state_match = validation_signals.get('pos_state_match', None)
         if pos_state_match is False:
             # Distrust PoS if state code explicitly mismatches recipient GSTIN state code
             logging.debug(f"Trust({field_key}): False (PoS State Code mismatch)")
             trusted = False


    # Final decision based on combined checks and maybe overall confidence
    if trusted and confidence < CONF_THRESHOLD_TRUSTED:
         logging.debug(f"Trust({field_key}): False (Passed checks, but confidence {confidence:.2f} below threshold {CONF_THRESHOLD_TRUSTED})")
         trusted = False # Final check against the primary threshold if all other checks passed

    if trusted:
         logging.debug(f"Trust({field_key}): True (Confidence {confidence:.2f}, Checks passed)")

    return trusted


# --- PDF Type Detection (Checks only first few pages for speed) ---
def detect_pdf_type(doc):
    """Classifies PDF as 'text' or 'image' based on extractable text from first few pages."""
    # (Keep existing implementation)
    full_text = ""
    is_text_based = False
    num_pages_to_check = min(3, len(doc))
    try:
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
    # (Keep existing implementation)
    text = page.get_text("text", sort=True)
    invoice_num = find_best_match(PATTERNS["invoice_number"], text)
    invoice_date = find_best_match(PATTERNS["invoice_date"], text)
    gstin_match = PATTERNS["gstin_pattern"].search(text)
    supplier_gstin = gstin_match.group(1).upper() if gstin_match else None # Use upper
    page_num_match = PATTERNS["page_number_pattern"].search(text)
    page_info = {"current": None, "total": None}
    if page_num_match:
        try:
            page_info["current"] = int(page_num_match.group(1))
            page_info["total"] = int(page_num_match.group(2))
        except ValueError: pass
    return {
        "invoice_number": invoice_num, "invoice_date": invoice_date,
        "supplier_gstin": supplier_gstin, "page_info": page_info,
        "has_text": bool(text and len(text.strip()) > 10)
    }

def segment_pdf_into_invoices(doc):
    """Analyzes pages to identify boundaries between distinct invoices."""
    # (Keep existing implementation)
    page_count = len(doc)
    if page_count == 0: return []
    if page_count == 1: return [[0]]
    page_data = [get_page_identifiers(doc.load_page(i)) for i in range(page_count)]
    invoice_segments = []
    current_segment = [0]; last_identifier_page_index = 0
    for i in range(1, page_count):
        current_page_info = page_data[i]; prev_page_key_info = page_data[last_identifier_page_index]
        new_invoice = False
        if current_page_info["page_info"]["current"] == 1 and current_page_info["page_info"]["total"] is not None:
             prev_page_num_info = prev_page_key_info.get("page_info", {"current": None, "total": None})
             if prev_page_num_info["current"] is None or prev_page_num_info["current"] == prev_page_num_info["total"]:
                 logging.debug(f"Segment Split Trigger: Page {i} detected as 'Page 1 of X'.")
                 new_invoice = True
        elif current_page_info["invoice_number"] is not None and \
             prev_page_key_info["invoice_number"] is not None and \
             current_page_info["invoice_number"] != prev_page_key_info["invoice_number"]:
            logging.debug(f"Segment Split Trigger: Inv# changed from '{prev_page_key_info['invoice_number']}' to '{current_page_info['invoice_number']}' on page {i}.")
            new_invoice = True
        elif current_page_info["invoice_number"] is None and \
             current_page_info["supplier_gstin"] is not None and \
             prev_page_key_info["supplier_gstin"] is not None and \
             prev_page_key_info["invoice_number"] is None and \
             current_page_info["supplier_gstin"] != prev_page_key_info["supplier_gstin"]:
             logging.debug(f"Segment Split Trigger: Supplier GSTIN changed (and no inv#) on page {i}.")
             new_invoice = True

        if new_invoice:
            invoice_segments.append(current_segment); current_segment = [i]
            last_identifier_page_index = i
        else:
            current_segment.append(i)
            if current_page_info["invoice_number"] or current_page_info["supplier_gstin"] or current_page_info["invoice_date"]:
                 last_identifier_page_index = i
    if current_segment: invoice_segments.append(current_segment)
    logging.info(f"Detected {len(invoice_segments)} invoice segment(s): {invoice_segments}")
    return invoice_segments


# --- Text-Based Path Functions (Modified for Page List) ---

def map_camelot_columns(df):
    # Keep existing map_camelot_columns (No validation changes needed here)
    # ... (previous implementation - could add debug logs if needed) ...
    if df.empty or len(df) < 1: return {}
    header_row = df.iloc[0]
    headers_cleaned = header_row.astype(str).str.lower().str.strip().str.replace(r'\s{2,}', ' ', regex=True).str.replace(r'\s*/\s*', '/', regex=True)
    cols = df.columns; mapping = {}; assigned_cols_indices = set()
    logging.debug(f"map_camelot_columns: Input DF shape: {df.shape}")
    logging.debug(f"map_camelot_columns: Cleaned Headers: {headers_cleaned.tolist()}")
    col_candidates = {
        "description": (PATTERNS["desc_kw"], None), "quantity": (PATTERNS["qty_kw"], parse_quantity),
        "unit_price": (PATTERNS["unit_price_kw"], parse_amount), "line_total": (PATTERNS["line_total_kw"], parse_amount),
        "hsn_sac": (PATTERNS["hsn_sac_kw"], None),
    }
    potential_line_total_indices, potential_qty_indices, potential_price_indices = [], [], []

    # Stage 1: Header Keyword Matching
    for c_idx, header_text in enumerate(headers_cleaned):
        if c_idx in assigned_cols_indices: continue
        matched_key = None
        for key, (pattern, _) in col_candidates.items():
            if key in ["line_total", "quantity", "unit_price"]: continue # Handle these later
            if pattern.search(header_text) and key not in mapping:
                 mapping[key] = cols[c_idx]; assigned_cols_indices.add(c_idx); matched_key = key; break
        if not matched_key:
             if col_candidates["quantity"][0].search(header_text): potential_qty_indices.append(c_idx)
             elif col_candidates["unit_price"][0].search(header_text): potential_price_indices.append(c_idx)
             elif col_candidates["line_total"][0].search(header_text):
                  is_tax_amount_col = any(kw in header_text for kw in ["cgst", "sgst", "igst", "tax"])
                  if not is_tax_amount_col: potential_line_total_indices.append(c_idx)

    # Stage 2: Resolve Ambiguity (Simplified logic from original)
    if "quantity" not in mapping and len(potential_qty_indices) == 1 and potential_qty_indices[0] not in assigned_cols_indices:
        idx = potential_qty_indices[0]; mapping["quantity"] = cols[idx]; assigned_cols_indices.add(idx)
    if "unit_price" not in mapping and len(potential_price_indices) == 1 and potential_price_indices[0] not in assigned_cols_indices:
        idx = potential_price_indices[0]; mapping["unit_price"] = cols[idx]; assigned_cols_indices.add(idx)
    if "line_total" not in mapping and potential_line_total_indices:
        idx_to_use = -1
        if len(potential_line_total_indices) == 1 and potential_line_total_indices[0] not in assigned_cols_indices:
             idx_to_use = potential_line_total_indices[0]
        else: # Prefer rightmost unassigned
            for idx in sorted(potential_line_total_indices, reverse=True):
                if idx not in assigned_cols_indices: idx_to_use = idx; break
        if idx_to_use != -1: mapping["line_total"] = cols[idx_to_use]; assigned_cols_indices.add(idx_to_use)

    # Stage 3: Content Analysis Fallback (Simplified) - Could be improved
    if len(df) > 1 and any(key not in mapping for key in ["quantity", "unit_price", "line_total"]):
        data_df = df.iloc[1:]; unassigned_indices = [i for i, col in enumerate(cols) if i not in assigned_cols_indices]
        numeric_col_scores = {}
        for idx in unassigned_indices:
             try: numeric_ratio = data_df[cols[idx]].apply(lambda x: pd.notna(parse_amount(x))).mean()
             except Exception: numeric_ratio = 0
             if numeric_ratio > 0.6: numeric_col_scores[idx] = numeric_ratio
        sorted_numeric_indices = sorted(numeric_col_scores.keys()) # Left-to-right
        if "line_total" not in mapping and sorted_numeric_indices:
             idx = sorted_numeric_indices.pop(-1); mapping["line_total"] = cols[idx]; assigned_cols_indices.add(idx)
        if "unit_price" not in mapping and sorted_numeric_indices:
             idx = sorted_numeric_indices.pop(-1); mapping["unit_price"] = cols[idx]; assigned_cols_indices.add(idx)
        if "quantity" not in mapping and sorted_numeric_indices:
             idx = sorted_numeric_indices.pop(0); mapping["quantity"] = cols[idx]; assigned_cols_indices.add(idx)

    # logging.debug(f"Final Camelot column mapping: {mapping}")
    logging.debug(f"map_camelot_columns: Final Mapping Result: {mapping}")

    if not all(key in mapping for key in ["description", "line_total"]):
         logging.warning(f"map_camelot_columns: Failed to map essential columns (description, line_total). Found: {list(mapping.keys())}")


    return mapping


import camelot
import logging
# Assume process_camelot_tables is defined elsewhere

# --- Logging Setup --- (Assuming basicConfig is set elsewhere)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_line_items_camelot(pdf_path, page_list):
    """Extracts line items using Camelot from specified pages."""
    # (Keep existing implementation calling process_camelot_tables)
    line_items_lattice = []
    line_items_stream = []
    tables_lattice = None # Initialize to None
    tables_stream = None  # Initialize to None

    # Convert page list (0-based index) to Camelot page string (1-based index)
    pages_str = ','.join(map(lambda x: str(x + 1), page_list))
    logging.info(f"Camelot: Attempting extraction from pages: {pages_str}")

    logging.info("Camelot: Trying 'lattice' flavor...")
    try:
        # Standard lattice call
        tables_lattice = camelot.read_pdf(pdf_path, pages=pages_str, flavor='lattice', suppress_stdout=True, line_scale=40)
        if tables_lattice and tables_lattice.n > 0 :
            logging.info(f"Camelot lattice found {tables_lattice.n} tables.")
        else:
            # It's normal for lattice to find nothing on PDFs without clear lines
            logging.info("Camelot lattice found no tables.")
    except Exception as e:
        logging.warning(f"Camelot lattice error: {e}") # Changed to warning

    logging.info("Camelot: Trying 'stream' flavor...")
    try:
        # Stream call with adjusted parameters
        tables_stream = camelot.read_pdf(
            pdf_path,
            pages=pages_str,
            flavor='stream',
            suppress_stdout=True,
            edge_tol=500,    # Keep relatively wide edge tolerance
            row_tol=20,      # Increased row tolerance
            column_tol=15    # Added column tolerance
        )
        if tables_stream and tables_stream.n > 0:
            logging.info(f"Camelot stream found {tables_stream.n} tables.")
        else:
            logging.info("Camelot stream found no tables.")
    except Exception as e:
        logging.warning(f"Camelot stream error: {e}") # Changed to warning

    # Process tables if found
    if tables_lattice and tables_lattice.n > 0:
        line_items_lattice = process_camelot_tables(tables_lattice, flavor='lattice')
    if tables_stream and tables_stream.n > 0:
        line_items_stream = process_camelot_tables(tables_stream, flavor='stream')

    # --- Evaluation Logic ---
    # Score based on items having a non-None value for 'line_total'
    score_lattice = sum(1 for item in line_items_lattice if item.get("line_total", {}).get("value") is not None)
    score_stream = sum(1 for item in line_items_stream if item.get("line_total", {}).get("value") is not None)

    logging.info(f"Camelot Evaluation: Lattice Score={score_lattice}, Stream Score={score_stream}")

    # Choose the best result set
    chosen_items = []
    if score_stream > score_lattice:
        logging.info("Choosing Camelot Stream results.")
        chosen_items = line_items_stream
    # Prefer lattice if scores equal OR if both scores are 0 but lattice found *something*
    elif score_lattice > 0 or (score_lattice == 0 and score_stream == 0 and line_items_lattice):
         logging.info("Choosing Camelot Lattice results.")
         chosen_items = line_items_lattice
    elif line_items_stream: # Fallback to stream if lattice is empty and stream isn't
         logging.info("Choosing Camelot Stream results (fallback).")
         chosen_items = line_items_stream
    # else: both are empty lists

    # Issue warning only if the chosen list is empty
    if not chosen_items:
         logging.warning("Camelot found no usable line items with either flavor.")

    return chosen_items


def process_camelot_tables(tables, flavor):
    """Processes tables extracted by Camelot. Includes internal line item math check."""
    # (Keep most of existing implementation - validation happens later)
    # Trust determination is moved to the main processing function.
    all_line_items = []
    col_candidates = {
        "description": (PATTERNS["desc_kw"], None), "quantity": (PATTERNS["qty_kw"], parse_quantity),
        "unit_price": (PATTERNS["unit_price_kw"], parse_amount), "line_total": (PATTERNS["line_total_kw"], parse_amount),
        "hsn_sac": (PATTERNS["hsn_sac_kw"], None),
    }
    item_schema_keys = ["description", "quantity", "unit_price", "line_total", "hsn_sac"]
    EXCLUSION_KEYWORDS = {"total", "subtotal", "sub total", "sub-total", "grand total",
                           "tax", "gst", "amount chargeable", "page total"} # Refined exclusions
    REQUIRED_FIELDS_FOR_LINE_ITEM = {"description", "line_total"} # Need desc and some value

    logging.info(f"Processing {tables.n} tables found with flavor='{flavor}'...")
    for i, table in enumerate(tables):
        df = table.df
        # logging.debug(f"Processing Table {i+1}/{tables.n} (Flavor: {flavor}, Shape: {df.shape})")
        if df.empty: continue
        df = df.replace(r'^\s*$', np.nan, regex=True) # Treat empty strings as NaN
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        if df.empty or len(df) < 2: continue # Need header and at least one data row

        col_mapping = map_camelot_columns(df.copy())
        if not col_mapping or not any(k in col_mapping for k in REQUIRED_FIELDS_FOR_LINE_ITEM):
            logging.debug(f"Skipping table {i+1} (Flavor: {flavor}): No usable column mapping found ({col_mapping}).")
            continue

        header_row_index = df.index[0] # Assumes first row is header
        data_df = df.drop(header_row_index).reset_index(drop=True)
        if data_df.empty: continue

        # logging.debug(f"Table {i+1} (Flavor: {flavor}): Processing {len(data_df)} data rows with mapping: {col_mapping}")
        for idx, row in data_df.iterrows():
            item_data = {key: {"value": None, "confidence": 0.0, "trusted": False} for key in item_schema_keys}
            parsed_vals = {}
            source_conf = table.parsing_report.get('accuracy', 70) / 100.0 # Base Camelot accuracy
            is_valid_row = True

            # Extract and Parse Raw Values
            desc_val = None
            has_numeric_val = False
            for key, col_name in col_mapping.items():
                if key not in col_candidates or col_name not in row.index: continue
                raw_val = row[col_name]
                parser_func = col_candidates[key][1]
                parsed_val = parser_func(raw_val) if parser_func else clean_text(raw_val)
                parsed_vals[key] = parsed_val
                if key == 'description': desc_val = parsed_val
                if key in ['quantity', 'unit_price', 'line_total'] and isinstance(parsed_val, (int, float)):
                     has_numeric_val = True

            # --- Filtering Logic ---
            skip_row = False; reason = ""
            # 1. Check for exclusion keywords in description (if description exists)
            if isinstance(desc_val, str):
                 desc_lower = desc_val.lower().strip()
                 if len(desc_lower) < 3 and not has_numeric_val: # Skip very short descriptions with no numbers
                      skip_row, reason = True, "Short description, no numeric values."
                 else:
                      for keyword in EXCLUSION_KEYWORDS:
                          # Check if keyword is the whole string or surrounded by spaces/boundaries
                           if f" {keyword} " in f" {desc_lower} " or desc_lower == keyword or desc_lower.startswith(keyword+' ') or desc_lower.endswith(' '+keyword):
                               # More specific check: Ensure it's likely a total/tax row, not just part of a description
                               potential_total_row = not any(parsed_vals.get(k) for k in ['quantity', 'unit_price'])
                               if potential_total_row or len(desc_lower) < len(keyword) + 10:
                                     skip_row, reason = True, f"Contains exclusion keyword '{keyword}'."
                                     break
            elif "description" in REQUIRED_FIELDS_FOR_LINE_ITEM: # If description is required and missing/not text
                 skip_row, reason = True, "Required 'description' missing/not text."


            # 2. Check if required fields are present *after* parsing
            if not skip_row:
                required_missing = False
                if 'description' in REQUIRED_FIELDS_FOR_LINE_ITEM and not isinstance(parsed_vals.get('description'), str):
                      required_missing = True
                if 'line_total' in REQUIRED_FIELDS_FOR_LINE_ITEM and not isinstance(parsed_vals.get('line_total'), (int, float)):
                      required_missing = True
                # Add other required fields if necessary

                if required_missing:
                     skip_row, reason = True, f"Missing required fields after parsing (Desc: {isinstance(parsed_vals.get('description'), str)}, Total: {isinstance(parsed_vals.get('line_total'), (int, float))})."

            if skip_row:
                logging.debug(f"Skipping Camelot Row {idx} (Flavor: {flavor}): {reason} | Row data: {row.to_dict()}")
                continue

            # --- Line Item Internal Math Check (Qty * Price vs Line Total) ---
            qty = parsed_vals.get("quantity")
            price = parsed_vals.get("unit_price")
            total = parsed_vals.get("line_total")
            line_item_math_passed = None # Internal check for Q*P=T
            if isinstance(qty, (int, float)) and isinstance(price, (int, float)) and isinstance(total, (int, float)):
                try:
                    line_item_math_passed = abs((qty * price) - total) < max(LINE_ITEM_QTY_PRICE_TOLERANCE, abs(total*0.01)) # Tolerance relative or absolute
                except TypeError: line_item_math_passed = False
                logging.debug(f"Line Item Math Check (Row {idx}): Qty={qty}, Price={price}, Expected={qty*price if qty and price else 'N/A'}, Actual={total}, Pass={line_item_math_passed}")


            # Populate item_data with confidence (Trust determined later)
            for key in item_schema_keys:
                 if key in parsed_vals:
                     value = parsed_vals[key]
                     format_valid = False
                     if value is not None:
                         if key in ["quantity", "unit_price", "line_total"]: format_valid = isinstance(value, (int, float))
                         elif key in ["description", "hsn_sac"]: format_valid = isinstance(value, str) and len(value) > 0
                         else: format_valid = True # Should not happen for these keys

                     base_field_conf = source_conf # Start with Camelot's table accuracy
                     final_conf = calculate_confidence(value, base_field_conf, format_valid)

                     # Apply small bonus/penalty for internal line item math check
                     if key in ["quantity", "unit_price", "line_total"]:
                         if line_item_math_passed is True: final_conf = min(1.0, final_conf + 0.1)
                         elif line_item_math_passed is False: final_conf = max(0.0, final_conf - 0.1)

                     item_data[key]['value'] = value
                     item_data[key]['confidence'] = round(final_conf, 3)
                     # Trusted flag set later by determine_trust based on this confidence + other signals

            all_line_items.append(item_data)

    logging.info(f"Finished processing tables with flavor='{flavor}'. Extracted {len(all_line_items)} potential line items.")
    return all_line_items


def process_text_based_pdf(pdf_path, full_text, page_list):
    """Extracts data using Regex (on full text) and Camelot (on specific pages)."""
    results = {}
    logging.debug(f"Processing text:\n{full_text[:500]}...") # Log start of text

    # --- Standard Extractions (Keep as before) ---
    results["invoice_number"] = find_best_match(PATTERNS["invoice_number"], full_text)
    results["invoice_date"] = find_best_match(PATTERNS["invoice_date"], full_text)
    results["gstin_supplier"], results["gstin_recipient"] = find_gstin_regex(full_text)
    results["place_of_supply"] = extract_place_regex(full_text, results.get("gstin_recipient"))
    results["place_of_origin"] = extract_place_regex(full_text, results.get("gstin_supplier")) # Assuming uncommented
    results["taxable_value"] = find_best_match(PATTERNS["taxable_value"], full_text, prefer_last=True) # Prefer last for taxable too? Test this.
    results["final_amount"] = find_best_match(PATTERNS["final_amount"], full_text, prefer_last=True)
    results["tax_amount"] = find_best_match(PATTERNS["tax_amount"], full_text, prefer_last=True)
    results["sgst_rate"] = find_best_match(PATTERNS["sgst_rate"], full_text)
    results["cgst_rate"] = find_best_match(PATTERNS["cgst_rate"], full_text)
    results["igst_rate"] = find_best_match(PATTERNS["igst_rate"], full_text)

        # --- CGST/SGST/IGST Amount Extraction (Hybrid Method - REVISED FALLBACK) ---

    # Try direct extraction first
    sgst_amt = find_best_match(PATTERNS.get("sgst_amount"), full_text, prefer_last=True)
    cgst_amt = find_best_match(PATTERNS.get("cgst_amount"), full_text, prefer_last=True)
    igst_amt = find_best_match(PATTERNS.get("igst_amount"), full_text, prefer_last=True)

    # SGST Fallback
    if sgst_amt is None:
        logging.debug("Direct SGST Amount pattern failed, trying line capture fallback...")
        sgst_line = find_best_match(PATTERNS.get("sgst_amount_line"), full_text, group_index=1, prefer_last=True)
        if sgst_line:
             logging.debug(f"Found SGST Amount Line segment: '{sgst_line[:100]}...'")
             # *** USE NEW FALLBACK FUNCTION ***
             sgst_amt = find_amount_in_line_fallback(sgst_line, prefer_last=True)
             if sgst_amt: logging.debug(f"SGST Fallback succeeded (prefer_last): {sgst_amt}")
             else: logging.debug("SGST Fallback (prefer_last) found no suitable amount.")
        else:
             logging.debug("SGST Amount Line pattern also failed.")
    else:
         logging.debug(f"Direct SGST Amount pattern succeeded: {sgst_amt}")

    # CGST Fallback
    if cgst_amt is None:
        logging.debug("Direct CGST Amount pattern failed, trying line capture fallback...")
        cgst_line = find_best_match(PATTERNS.get("cgst_amount_line"), full_text, group_index=1, prefer_last=True)
        if cgst_line:
             logging.debug(f"Found CGST Amount Line segment: '{cgst_line[:100]}...'")
             # *** USE NEW FALLBACK FUNCTION ***
             cgst_amt = find_amount_in_line_fallback(cgst_line, prefer_last=True)
             if cgst_amt: logging.debug(f"CGST Fallback succeeded (prefer_last): {cgst_amt}")
             else: logging.debug("CGST Fallback (prefer_last) found no suitable amount.")
        else:
             logging.debug("CGST Amount Line pattern also failed.")
    else:
         logging.debug(f"Direct CGST Amount pattern succeeded: {cgst_amt}")

    # IGST Fallback
    if igst_amt is None:
        igst_pattern_line = PATTERNS.get("igst_amount_line")
        if igst_pattern_line:
            logging.debug("Direct IGST Amount pattern failed, trying line capture fallback...")
            igst_line = find_best_match(igst_pattern_line, full_text, group_index=1, prefer_last=True)
            if igst_line:
                logging.debug(f"Found IGST Amount Line segment: '{igst_line[:100]}...'")
                # *** USE NEW FALLBACK FUNCTION ***
                igst_amt = find_amount_in_line_fallback(igst_line, prefer_last=True)
                if igst_amt: logging.debug(f"IGST Fallback succeeded (prefer_last): {igst_amt}")
                else: logging.debug("IGST Fallback (prefer_last) found no suitable amount.")
            else:
                logging.debug("IGST Amount Line pattern also failed.")
    else:
         logging.debug(f"Direct IGST Amount pattern succeeded: {igst_amt}")

    results["sgst_amount"] = sgst_amt
    results["cgst_amount"] = cgst_amt
    results["igst_amount"] = igst_amt



    # --- Rate List Extraction (Keep as before) ---
    all_rates = find_best_match(PATTERNS["rate_pattern"], full_text, find_all=True)
    gst_rates = find_best_match(PATTERNS["gst_rate"], full_text, find_all=True)
    combined_rates = set((all_rates or []) + (gst_rates or []))
    parsed_rates = [parse_amount(r) for r in combined_rates if parse_amount(r) is not None]
    results["tax_rate"] = list(set(parsed_rates)) if parsed_rates else None

    # --- Line Items (Keep as before) ---
    line_items = extract_line_items_camelot(pdf_path, page_list)

    return results, line_items


# --- Image-Based Path Functions (AWS Textract) ---

def get_textract_block_text(block, blocks_map):
    # (Keep existing implementation)
    text = ""
    if 'Relationships' in block:
        for relationship in block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    child_block = blocks_map.get(child_id)
                    if child_block and child_block['BlockType'] == 'WORD':
                        text += child_block['Text'] + ' '
                    # Consider handling SELECTION_ELEMENT text if needed
    elif 'Text' in block: text = block['Text']
    return text.strip()

def parse_textract_response(response, target_pages):
    """Parses Textract response, filtering blocks for target pages."""
    # (Keep existing implementation - filtering by page)
    all_blocks = response.get('Blocks', [])
    target_page_set = set(target_pages)
    blocks = [block for block in all_blocks if block.get('Page') in target_page_set]
    if not blocks:
         logging.warning(f"Textract: No blocks found for target pages {target_pages}.")
         return {}, []
    logging.info(f"Textract: Processing {len(blocks)} blocks relevant to pages {target_pages} (out of {len(all_blocks)} total blocks).")
    blocks_map = {block['Id']: block for block in blocks}
    key_value_pairs = {}
    tables = []

    # Extract Key-Value pairs
    kv_set_blocks = [b for b in blocks if b['BlockType'] == 'KEY_VALUE_SET' and b.get('EntityTypes') and 'KEY' in b['EntityTypes']] # Ensure it's a KEY entity
    value_blocks = {b['Id']: b for b_id, b in blocks_map.items() if b['BlockType'] == 'VALUE'}

    for kv_set in kv_set_blocks:
        key_text, value_text = "", ""; key_conf, value_conf = 0.0, 0.0
        key_block, value_block = None, None
        if 'Relationships' in kv_set:
            for rel in kv_set['Relationships']:
                if rel['Type'] == 'VALUE' and rel['Ids']:
                    value_id = rel['Ids'][0]
                    if value_id in value_blocks: # Check if value block exists and is on target page
                        value_block = value_blocks[value_id]
                        value_text = get_textract_block_text(value_block, blocks_map)
                        value_conf = value_block.get('Confidence', 0.0) / 100.0
                elif rel['Type'] == 'CHILD' and rel['Ids']: # Key text is often in CHILD WORD blocks
                    for child_id in rel['Ids']:
                         child_block = blocks_map.get(child_id)
                         if child_block and child_block['BlockType'] == 'WORD':
                             key_text += child_block['Text'] + ' '
            key_text = key_text.strip()
            key_conf = kv_set.get('Confidence', 0.0) / 100.0 # Use KEY confidence

        if key_text and value_text: # Require both key and value text
            clean_key = clean_text(key_text.lower().replace(':','').replace('#','')) # Clean common chars
            avg_conf = (key_conf + value_conf) / 2.0 if key_conf > 0 and value_conf > 0 else max(key_conf, value_conf)
            # Store if key is meaningful, potentially check against synonyms?
            if len(clean_key) > 1 :
                 # Handle multiple matches - keep the one with higher confidence?
                 if clean_key not in key_value_pairs or avg_conf > key_value_pairs[clean_key]['confidence']:
                      key_value_pairs[clean_key] = {"value": value_text, "confidence": avg_conf}

    # Extract Tables
    table_blocks = [b for b in blocks if b['BlockType'] == 'TABLE']
    for table_block in table_blocks:
        table_data = []
        cell_blocks = {}
        if 'Relationships' in table_block:
            for rel in table_block['Relationships']:
                if rel['Type'] == 'CHILD':
                    for cell_id in rel['Ids']:
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
             rows[row_idx][col_idx] = {"text": cell_text, "confidence": cell_conf, "id": cell_id} # Store ID?

        if rows:
            table_matrix = []
            for r in range(1, max_row + 1):
                row_list = [rows.get(r, {}).get(c, {"text": "", "confidence": 0.0}) for c in range(1, max_col + 1)]
                table_matrix.append(row_list)
            tables.append({"matrix": table_matrix, "confidence": table_block.get('Confidence', 0.0)/100.0})

    return key_value_pairs, tables


def map_textract_output(key_value_pairs, tables):
    """Maps parsed Textract KVs and Tables to schema. Includes line item math check."""
    # (Keep existing K/V mapping logic, adapt table processing for line item math check)
    results = defaultdict(lambda: {"value": None, "confidence": 0.0})
    line_items = []

    # --- Map Key-Value Pairs ---
    processed_gstins = set() # Track assigned GSTIN values to avoid duplicates
    # Prioritize direct synonym matches
    for raw_key, data in key_value_pairs.items():
        matched = False
        for synonym, schema_key in TEXTRACT_KEY_SYNONYMS.items():
             # Use looser matching? 'in' might be too broad. Use word boundaries?
             # Simple 'in' for now, but might need refinement.
             if f" {synonym} " in f" {raw_key} " or raw_key.startswith(synonym) or raw_key.endswith(synonym):
                 current_conf = results[schema_key]['confidence']
                 # Always update if new confidence is higher
                 if data['confidence'] > current_conf:
                      # Special handling for GSTINs to avoid overwriting supplier with recipient if ambiguous
                      if 'gstin' in schema_key:
                          gstin_val = clean_text(data['value'])
                          is_valid_format = bool(gstin_val and PATTERNS['gstin_pattern'].fullmatch(gstin_val))
                          if is_valid_format and gstin_val not in processed_gstins:
                              # If target is supplier and supplier empty, assign
                              if schema_key == 'gstin_supplier' and results['gstin_supplier']['value'] is None:
                                   results['gstin_supplier'] = data; processed_gstins.add(gstin_val); matched = True; break
                              # If target is recipient and recipient empty, assign
                              elif schema_key == 'gstin_recipient' and results['gstin_recipient']['value'] is None:
                                   results['gstin_recipient'] = data; processed_gstins.add(gstin_val); matched = True; break
                              # If target is generic 'gstin' and supplier empty, assign to supplier
                              elif schema_key == 'gstin_supplier' and results['gstin_supplier']['value'] is None: # Handles 'gstin' mapping to supplier first
                                   results['gstin_supplier'] = data; processed_gstins.add(gstin_val); matched = True; break
                      else: # Not a GSTIN field
                           results[schema_key] = data; matched = True; break
                 elif data['confidence'] == current_conf and results[schema_key]['value'] is None and data['value'] is not None:
                      # If confidence is same, prefer non-null value
                      results[schema_key] = data; matched = True; break

    # Fallback for GSTINs not matched via synonyms
    if results['gstin_supplier']['value'] is None or results['gstin_recipient']['value'] is None:
         for raw_key, data in key_value_pairs.items():
             if 'gst' in raw_key: # Broader check
                 gstin_val = clean_text(data['value'])
                 is_valid_format = bool(gstin_val and PATTERNS['gstin_pattern'].fullmatch(gstin_val))
                 if is_valid_format and gstin_val not in processed_gstins:
                     # Assign to first available slot (supplier then recipient)
                     if results['gstin_supplier']['value'] is None:
                         results['gstin_supplier'] = data; processed_gstins.add(gstin_val)
                     elif results['gstin_recipient']['value'] is None:
                         results['gstin_recipient'] = data; processed_gstins.add(gstin_val)
                     # Stop if both are filled
                     if results['gstin_supplier']['value'] is not None and results['gstin_recipient']['value'] is not None:
                          break

    # --- Map Tables to Line Items ---
    col_candidates = {
        "description": (PATTERNS["desc_kw"], None), "quantity": (PATTERNS["qty_kw"], parse_quantity),
        "unit_price": (PATTERNS["unit_price_kw"], parse_amount), "line_total": (PATTERNS["line_total_kw"], parse_amount),
        "hsn_sac": (PATTERNS["hsn_sac_kw"], None), # Add HSN/SAC
    }
    item_schema_keys = ["description", "quantity", "unit_price", "line_total", "hsn_sac"]
    REQUIRED_FIELDS_FOR_LINE_ITEM = {"description", "line_total"} # Relaxed
    EXCLUSION_KEYWORDS = {"total", "subtotal", "sub total", "sub-total", "grand total",
                           "tax", "gst", "amount chargeable", "page total"} # Same exclusions as Camelot

    for t_idx, table in enumerate(tables):
        matrix = table['matrix']
        if not matrix or len(matrix) < 2: continue # Need header + data
        # logging.debug(f"Processing Textract Table {t_idx+1}")

        header_row_data = matrix[0]; data_rows = matrix[1:]
        col_mapping_idx = {}; assigned_cols_idx = set()
        # Map headers using keywords (similar to Camelot mapping logic)
        header_texts = [clean_text(cell_data['text']).lower() for cell_data in header_row_data]
        for c_idx, header_text in enumerate(header_texts):
             if c_idx in assigned_cols_idx: continue
             for key, (pattern, _) in col_candidates.items():
                 if key not in col_mapping_idx and pattern.search(header_text):
                     # Avoid mapping tax columns to quantity/price/total
                     is_tax_col = any(kw in header_text for kw in ["cgst", "sgst", "igst", "tax rate", "tax amt"])
                     if key in ["quantity", "unit_price", "line_total"] and is_tax_col:
                          continue
                     col_mapping_idx[key] = c_idx; assigned_cols_idx.add(c_idx); break
        # logging.debug(f"Textract table column mapping: {col_mapping_idx}")

        if not col_mapping_idx or not any(k in col_mapping_idx for k in REQUIRED_FIELDS_FOR_LINE_ITEM):
             logging.debug(f"Skipping Textract table {t_idx+1}: No usable column mapping found ({col_mapping_idx}).")
             continue

        for r_idx, row_data in enumerate(data_rows):
             item = {key: {"value": None, "confidence": 0.0, "trusted": False} for key in item_schema_keys}
             parsed_vals = {}
             desc_val = None
             has_numeric_val = False

             # Extract & Parse
             for key, col_idx in col_mapping_idx.items():
                 if key not in item_schema_keys: continue
                 if col_idx < len(row_data):
                     cell_data = row_data[col_idx]; raw_val = cell_data['text']
                     base_conf = cell_data['confidence'] # Textract cell confidence
                     parser_func = col_candidates[key][1]
                     parsed_val = parser_func(raw_val) if parser_func else clean_text(raw_val)
                     parsed_vals[key] = parsed_val
                     if key == 'description': desc_val = parsed_val
                     if key in ['quantity', 'unit_price', 'line_total'] and isinstance(parsed_val, (int, float)):
                         has_numeric_val = True

                     # Calculate initial confidence based on Textract + format
                     format_valid = False
                     if parsed_val is not None:
                          if key in ["quantity", "unit_price", "line_total"]: format_valid = isinstance(parsed_val, (int, float))
                          elif key in ["description", "hsn_sac"]: format_valid = isinstance(parsed_val, str) and len(parsed_val) > 0
                     conf = calculate_confidence(parsed_val, base_conf, format_valid)
                     item[key] = {"value": parsed_val, "confidence": round(conf, 3)} # Trust TBD
                 # else: Field mapped but column doesn't exist in this row

             # --- Filtering Logic (similar to Camelot) ---
             skip_row = False; reason = ""
             if isinstance(desc_val, str):
                  desc_lower = desc_val.lower().strip()
                  if len(desc_lower) < 3 and not has_numeric_val:
                       skip_row, reason = True, "Short description, no numeric values."
                  else:
                       for keyword in EXCLUSION_KEYWORDS:
                            if f" {keyword} " in f" {desc_lower} " or desc_lower == keyword or desc_lower.startswith(keyword+' ') or desc_lower.endswith(' '+keyword):
                                 potential_total_row = not any(parsed_vals.get(k) for k in ['quantity', 'unit_price'])
                                 if potential_total_row or len(desc_lower) < len(keyword) + 10:
                                      skip_row, reason = True, f"Contains exclusion keyword '{keyword}'."
                                      break
             elif "description" in REQUIRED_FIELDS_FOR_LINE_ITEM:
                  skip_row, reason = True, "Required 'description' missing/not text."

             if not skip_row:
                required_missing = False
                if 'description' in REQUIRED_FIELDS_FOR_LINE_ITEM and not isinstance(parsed_vals.get('description'), str): required_missing = True
                if 'line_total' in REQUIRED_FIELDS_FOR_LINE_ITEM and not isinstance(parsed_vals.get('line_total'), (int, float)): required_missing = True
                if required_missing:
                     skip_row, reason = True, "Missing required fields after parsing."

             if skip_row:
                 logging.debug(f"Skipping Textract Row {r_idx} (Table {t_idx+1}): {reason} | Parsed: {parsed_vals}")
                 continue

             # --- Internal Line Item Math Check (Q*P=T) ---
             qty = parsed_vals.get("quantity")
             price = parsed_vals.get("unit_price")
             total = parsed_vals.get("line_total")
             line_item_math_passed = None
             if isinstance(qty, (int, float)) and isinstance(price, (int, float)) and isinstance(total, (int, float)):
                 try: line_item_math_passed = abs((qty * price) - total) < max(LINE_ITEM_QTY_PRICE_TOLERANCE, abs(total*0.01))
                 except TypeError: line_item_math_passed = False
                 logging.debug(f"Line Item Math Check (Textract Row {r_idx}): Q*P vs T Pass={line_item_math_passed}")


             # Recalculate confidence incorporating internal math check
             valid_item = False
             for key in item.keys():
                 if key not in item_schema_keys: continue
                 current_conf = item[key]['confidence']
                 if key in ["quantity", "unit_price", "line_total"]:
                     if line_item_math_passed is True: current_conf = min(1.0, current_conf + 0.1)
                     elif line_item_math_passed is False: current_conf = max(0.0, current_conf - 0.1)
                 item[key]['confidence'] = round(current_conf, 3)
                 # Trusted flag set later
                 if item[key]['value'] is not None and key in REQUIRED_FIELDS_FOR_LINE_ITEM:
                      valid_item = True

             if valid_item:
                 line_items.append(item)
                 # logging.debug(f"Adding Textract line item {r_idx}: {item}")


    # Standardize GSTIN case in final results
    for key in ['gstin_supplier', 'gstin_recipient']:
         if results[key]['value']:
             results[key]['value'] = results[key]['value'].upper()

    return dict(results), line_items


def process_image_based_pdf(textract_response, page_list_1_based):
    """Processes pre-fetched Textract response for specific pages."""
    # (Keep existing implementation - validation happens later)
    results_mapped = {}
    line_items = []
    try:
        key_value_pairs, tables = parse_textract_response(textract_response, page_list_1_based)
        if not key_value_pairs and not tables:
             logging.warning("Textract parsing yielded no K/V pairs or Tables for the target pages.")
             results_mapped = {key: {"value": None, "confidence": 0.0} for key in SCHEMA_KEYS if key not in ['line_items', '_pdf_page_start', '_pdf_page_end']}
             line_items = []
        else:
             # map_textract_output provides values and confidences
             results_mapped, line_items = map_textract_output(key_value_pairs, tables)

    except Exception as e:
        logging.error(f"Error during Textract result processing: {e}", exc_info=True)
        results_mapped = {key: {"value": None, "confidence": 0.0} for key in SCHEMA_KEYS if key not in ['line_items', '_pdf_page_start', '_pdf_page_end']}
        line_items = []

    return results_mapped, line_items


# --- Main Hybrid Orchestration (MODIFIED for Validation & Trust) ---

def process_invoice_segment(pdf_path, pdf_doc, page_list_0_based, pdf_type, textract_response=None):
    """Processes a single detected invoice segment, including validation and trust."""
    segment_start_page = page_list_0_based[0] + 1
    segment_end_page = page_list_0_based[-1] + 1
    logging.info(f"Processing segment: Pages {segment_start_page}-{segment_end_page}, Type: {pdf_type}")

    final_output = defaultdict(lambda: {"value": None, "confidence": 0.0, "trusted": False})
    final_output["line_items"] = [] # Initialize line items list
    final_output["_pdf_page_start"] = segment_start_page
    final_output["_pdf_page_end"] = segment_end_page

    raw_results = {}
    line_items_extracted = []
    base_confidences = {} # Store base confidences from source

    # --- Stage 1: Extraction (Text or Image Path) ---
    if pdf_type == "text":
        logging.info("Using Text-Based Path for segment (PyMuPDF + Regex + Camelot)")
        segment_text = ""; full_text = ""
        for page_index in page_list_0_based:
            try: segment_text += pdf_doc.load_page(page_index).get_text("text", sort=True) + "\n"
            except Exception as e: logging.warning(f"Could not extract text from page {page_index + 1}: {e}")
        full_text = clean_text(segment_text)

        if not full_text:
             logging.warning(f"Segment pages {segment_start_page}-{segment_end_page} had no extractable text.")
             # Return empty result for this segment?
             for key in SCHEMA_KEYS: final_output[key] # Ensure keys exist
             return dict(final_output)
        else:
            # Extract raw strings/values using Regex + Camelot
            raw_results, line_items_extracted = process_text_based_pdf(pdf_path, full_text, page_list_0_based)
            # Assign base confidence for Regex results
            for key in raw_results.keys():
                 if raw_results[key] is not None: base_confidences[key] = BASE_CONF_PYMUPDF_TEXT
            # Line items from Camelot already have 'confidence' from process_camelot_tables

    elif pdf_type == "image":
        logging.info("Using Image-Based Path for segment (AWS Textract Results)")
        if textract_response:
            page_list_1_based = [p + 1 for p in page_list_0_based]
            # Process pre-fetched Textract response, filtering by page
            # This returns dicts with {'value': val, 'confidence': conf} for K/V pairs
            # and list of dicts for line items with same structure per field
            raw_results_with_conf, line_items_extracted = process_image_based_pdf(textract_response, page_list_1_based)

            # Separate raw values and base confidences
            for key, data in raw_results_with_conf.items():
                 raw_results[key] = data.get('value') # Use .get() for safety
                 if data.get('value') is not None: # Only store conf if value exists
                     base_confidences[key] = data.get('confidence', 0.0)
            # Line items from Textract already have 'confidence' from map_textract_output
        else:
             logging.error("Textract response missing for image-based segment.")
             for key in SCHEMA_KEYS: final_output[key] # Ensure keys exist
             return dict(final_output)

    else:
        logging.error(f"Unknown PDF type '{pdf_type}' for segment.")
        for key in SCHEMA_KEYS: final_output[key]
        return dict(final_output)


    # --- Stage 2: Common Post-Processing, Validation & Trust ---
    logging.info("Performing common post-processing, validation, and trust assessment for segment...")
    parsed_results = {} # Stores {'value': parsed_value, 'format_valid': bool}
    validation_signals = {} # Stores results of specific checks {'gstin_checksum': {'gstin_supplier': True}, 'date_plausible': True, ...}

    # --- 2a. Parse Amounts, Dates, Rates ---
    for key, raw_value in raw_results.items():
        if key == "line_items": continue # Handled separately
        parsed_value = raw_value; format_valid = None; is_numeric = False; is_date = False

        if raw_value is not None and raw_value != '':
            try:
                # Amount/Value/Total fields (excluding specific non-amount fields)
                if any(k in key for k in ["amount", "value", "total"]) and key not in ["invoice_number", "place_of_supply", "place_of_origin"]:
                     parsed_value = parse_amount(raw_value); format_valid = (parsed_value is not None); is_numeric=True
                # Rate fields (single value)
                elif key in ["sgst_rate", "cgst_rate", "igst_rate"]:
                     parsed_value = parse_amount(raw_value); format_valid = (parsed_value is not None); is_numeric=True
                # Date fields
                elif "date" in key:
                     parsed_value = parse_date_robust(raw_value); format_valid = (parsed_value is not None); is_date = True
                     if format_valid: parsed_value = parsed_value # Keep as date object for validation
                # GSTIN fields (format checked later by checksum)
                elif "gstin" in key:
                     parsed_value = clean_text(str(raw_value)).upper(); format_valid = (parsed_value is not None and len(parsed_value) == 15)
                # Tax Rate (potentially a list)
                elif key == "tax_rate":
                     if isinstance(raw_value, list): # Already parsed list
                         parsed_value = [r for r in raw_value if isinstance(r, (int, float))]
                         format_valid = len(parsed_value) == len(raw_value) and len(parsed_value) > 0
                     elif isinstance(raw_value, str): # Try parsing if single string representation
                         pv = parse_amount(raw_value)
                         if pv is not None: parsed_value = [pv]; format_valid = True
                         else: format_valid = False
                     else: format_valid = False
                # Default: Text fields
                else:
                     parsed_value = clean_text(str(raw_value)); format_valid = isinstance(parsed_value, str) and len(parsed_value) > 0

            except Exception as e:
                logging.warning(f"Parsing error for key '{key}', value '{raw_value}': {e}")
                parsed_value = raw_value # Keep raw on error
                format_valid = False
        else: # Handle None or empty string
             parsed_value = None
             format_valid = False # Consider empty string invalid format for most fields

        parsed_results[key] = {"value": parsed_value, "format_valid": format_valid}
        # Also store format validity in validation_signals for trust logic
        if 'format_valid' not in validation_signals: validation_signals['format_valid'] = {}
        validation_signals['format_valid'][key] = format_valid


    # --- 2b. Perform Specific Validations ---
    # Initialize signal categories
    validation_signals['gstin_checksum'] = {}
    validation_signals['gst_rate_valid'] = {}

    # GSTIN Checksum
    for key in ['gstin_supplier', 'gstin_recipient']:
        gstin_data = parsed_results.get(key)
        if gstin_data and gstin_data['value'] and gstin_data['format_valid']:
            is_valid = is_valid_gstin_checksum(gstin_data['value'])
            validation_signals['gstin_checksum'][key] = is_valid
            if not is_valid: logging.debug(f"Validation: {key} '{gstin_data['value']}' failed checksum.")
        else: validation_signals['gstin_checksum'][key] = None # Cannot check

    # Date Plausibility
    date_data = parsed_results.get('invoice_date')
    if date_data and date_data['value'] and isinstance(date_data['value'], datetime.date):
        is_plausible = is_plausible_date(date_data['value'])
        validation_signals['date_plausible'] = is_plausible
        if not is_plausible: logging.debug(f"Validation: Invoice date '{date_data['value']}' not plausible.")
    else: validation_signals['date_plausible'] = None

    # Standard GST Rate Check
    for key in ['sgst_rate', 'cgst_rate', 'igst_rate']:
         rate_data = parsed_results.get(key)
         if rate_data and rate_data['value'] is not None and rate_data['format_valid']:
             is_std_rate = is_valid_gst_rate(rate_data['value'])
             validation_signals['gst_rate_valid'][key] = is_std_rate
             if not is_std_rate: logging.debug(f"Validation: {key} '{rate_data['value']}' is not a standard GST rate.")
         else: validation_signals['gst_rate_valid'][key] = None

    # Check list of rates in tax_rate
    tax_rate_data = parsed_results.get('tax_rate')
    if tax_rate_data and tax_rate_data['value'] and isinstance(tax_rate_data['value'], list):
         all_rates_valid = all(is_valid_gst_rate(r) for r in tax_rate_data['value'])
         validation_signals['gst_rate_valid']['tax_rate'] = all_rates_valid
         if not all_rates_valid: logging.debug(f"Validation: tax_rate list {tax_rate_data['value']} contains non-standard rates.")
    elif tax_rate_data and tax_rate_data['value'] is not None : # Single value parsed earlier
         is_std_rate = is_valid_gst_rate(tax_rate_data['value'][0]) # Check the single value
         validation_signals['gst_rate_valid']['tax_rate'] = is_std_rate
         if not is_std_rate: logging.debug(f"Validation: tax_rate '{tax_rate_data['value'][0]}' is not a standard GST rate.")
    else: validation_signals['gst_rate_valid']['tax_rate'] = None


    # --- 2c. Perform Cross-Field Validations ---
    # Total Math Check (Taxable + Tax = Final)
    total_math_result = check_total_math(parsed_results) # Returns True, False, or None
    validation_signals['total_math_ok'] = total_math_result
    if total_math_result is False: logging.warning("Validation: Total Math check (Taxable + Tax = Final) failed.")
    elif total_math_result is True: logging.debug("Validation: Total Math check passed.")

    # Place of Supply vs Recipient State Code
    recipient_gstin_val = parsed_results.get('gstin_recipient', {}).get('value')
    pos_val = parsed_results.get('place_of_supply', {}).get('value')
    pos_state_match_result = check_pos_vs_recipient_state(recipient_gstin_val, pos_val) # Returns True, False, or None
    validation_signals['pos_state_match'] = pos_state_match_result
    if pos_state_match_result is False: logging.warning("Validation: Place of Supply state code mismatch with Recipient GSTIN state code.")
    elif pos_state_match_result is True: logging.debug("Validation: Place of Supply state code matches Recipient GSTIN.")

    # --- Rate * Taxable = Tax Amount Check ---
    rate_tax_math_results = check_rate_tax_math(parsed_results) # Returns dict {'sgst_ok': ..., 'overall_ok': ...}
    validation_signals['rate_tax_math_results'] = rate_tax_math_results # Store the detailed dict
    if rate_tax_math_results['overall_ok'] is False: logging.warning("Validation: Rate * Taxable = Tax Amount check failed for one or more components.")
    elif rate_tax_math_results['overall_ok'] is True: logging.debug("Validation: Rate * Taxable = Tax Amount check passed for all applicable components.")

    # --- 2d. Final Confidence Calculation and Trust Assignment ---
    logging.info("Calculating final confidence and trust for segment...")
    for key in SCHEMA_KEYS:
        if key in ['line_items', '_pdf_page_start', '_pdf_page_end']: continue

        data = parsed_results.get(key)
        value = None; final_conf = 0.0; is_trusted = False

        if data:
            value = data['value']
            format_valid = validation_signals.get('format_valid', {}).get(key, False) # Use stored format validity
            base_conf = base_confidences.get(key, 0.5) # Use stored base conf, default if missing

            # Calculate confidence (using the same function as before)
            final_conf = calculate_confidence(value, base_conf, format_valid)

            # Determine trust using the multi-signal logic
            # Pass all gathered signals relevant to this field
            field_validation_signals = {
                'format_valid': format_valid,
                'gstin_checksum': validation_signals.get('gstin_checksum', {}), # Pass the dict for specific key lookup
                'date_plausible': validation_signals.get('date_plausible'),
                'gst_rate_valid': validation_signals.get('gst_rate_valid', {}), # Pass the dict
                'total_math_ok': validation_signals.get('total_math_ok'),
                'pos_state_match': validation_signals.get('pos_state_match'),
                'rate_tax_math_results': validation_signals.get('rate_tax_math_results', {})
            }
            is_trusted = determine_trust(key, value, final_conf, field_validation_signals)

            # Convert date back to ISO format string for output JSON/Excel
            if key == 'invoice_date' and isinstance(value, datetime.date):
                value = value.isoformat()
            # Ensure list of rates is stored correctly
            elif key == 'tax_rate' and value is None:
                 value = [] # Represent no rates found as empty list


        # Assign to final output
        final_output[key] = {
            "value": value,
            "confidence": round(final_conf, 3),
            "trusted": is_trusted
        }

    # --- 2e. Process Line Items (Apply Trust) ---
    processed_line_items = []
    for item in line_items_extracted:
        # Line items already have confidence calculated incorporating internal math check
        # We just need to apply the 'trusted' flag based on their confidence
        # Could add more complex trust logic for line items if needed (e.g., based on header trust)
        processed_item = {}
        for key, data in item.items():
             conf = data.get('confidence', 0.0)
             val = data.get('value')
             # Simple trust based on confidence for line items for now
             # Requires high confidence AND the value not being None
             trusted = (conf >= CONF_THRESHOLD_TRUSTED and val is not None)
             processed_item[key] = {"value": val, "confidence": conf, "trusted": trusted}
        processed_line_items.append(processed_item)

    final_output['line_items'] = processed_line_items

    logging.info(f"Finished segment processing. Final Amount: {final_output['final_amount']['value']} (Conf: {final_output['final_amount']['confidence']:.2f}, Trusted: {final_output['final_amount']['trusted']})")
    return dict(final_output)


# --- New Top-Level Orchestrator ---
def extract_invoice_data_from_pdf(pdf_path):
    """
    Top-level function to handle PDF segmentation and processing.
    Returns a list of dictionaries, one for each detected invoice.
    """
    # (Keep existing implementation calling segment_pdf_into_invoices and process_invoice_segment)
    logging.info(f"\n--- Processing {os.path.basename(pdf_path)} ---")
    start_time = time.time()
    all_invoice_results = []
    doc = None
    textract_response = None

    try:
        doc = fitz.open(pdf_path)
        if not doc or len(doc) == 0:
            logging.error("PDF is empty or could not be opened.")
            return [{"_filename": os.path.basename(pdf_path), "_error": "Empty or invalid PDF", "_invoice_index_in_pdf": 1, "_pdf_page_start": 1, "_pdf_page_end": 0}]

        pdf_type = detect_pdf_type(doc)
        invoice_segments_pages = segment_pdf_into_invoices(doc)

        if pdf_type == "image":
            logging.info("Image PDF detected. Calling AWS Textract for the entire document...")
            try:
                textract_client = boto3.client('textract', region_name=AWS_REGION)
                with open(pdf_path, 'rb') as document_bytes: image_bytes = document_bytes.read()
                textract_response = textract_client.analyze_document(
                    Document={'Bytes': image_bytes}, FeatureTypes=['FORMS', 'TABLES'] )
                logging.info("Received response from Textract.")
            except Exception as e:
                logging.error(f"AWS Textract API call failed: {e}", exc_info=True)
                return [{
                    "_filename": os.path.basename(pdf_path), "_error": f"Textract API Error: {e}",
                    "_invoice_index_in_pdf": 1, "_pdf_page_start": 1, "_pdf_page_end": len(doc) }]

        for i, segment_pages in enumerate(invoice_segments_pages):
            logging.info(f"--- Starting processing for Invoice Segment {i+1}/{len(invoice_segments_pages)} (Pages {min(segment_pages)+1}-{max(segment_pages)+1}) ---")
            try:
                segment_result = process_invoice_segment(
                    pdf_path, doc, segment_pages, pdf_type, textract_response )
                segment_result['_filename'] = os.path.basename(pdf_path)
                segment_result['_invoice_index_in_pdf'] = i + 1
                all_invoice_results.append(segment_result)
            except Exception as e_seg:
                 logging.error(f"Error processing segment {i+1} (Pages {min(segment_pages)+1}-{max(segment_pages)+1}): {e_seg}", exc_info=True)
                 all_invoice_results.append({
                     "_filename": os.path.basename(pdf_path), "_error": f"Segment processing error: {e_seg}",
                     "_invoice_index_in_pdf": i + 1, "_pdf_page_start": min(segment_pages)+1, "_pdf_page_end": max(segment_pages)+1
                 })
            logging.info(f"--- Finished processing for Invoice Segment {i+1}/{len(invoice_segments_pages)} ---")

    except Exception as e:
        logging.error(f"!!!!!!!! Critical error processing {os.path.basename(pdf_path)}: {e} !!!!!!!!", exc_info=True)
        all_invoice_results.append({
            "_filename": os.path.basename(pdf_path), "_error": f"Critical processing error: {e}",
            "_invoice_index_in_pdf": 1, "_pdf_page_start": 1, "_pdf_page_end": len(doc) if doc else 'N/A' })
    finally:
        if doc: doc.close()

    end_time = time.time()
    logging.info(f"--- Finished processing {os.path.basename(pdf_path)} in {end_time - start_time:.2f} seconds ---")
    return all_invoice_results


# --- NEW Metrics Calculation Functions ---

# Inside compare_values function

def compare_values(extracted_val, actual_val, field_key):
    """
    Compares extracted value with actual value, handling types and tolerances.
    Returns True if matching, False otherwise.
    """
    # --- Initial Checks ---
    actual_is_empty = pd.isna(actual_val) or actual_val == ''
    # Check specifically for extracted being None or Nan before converting to string
    extracted_is_none_or_nan = extracted_val is None or (isinstance(extracted_val, float) and np.isnan(extracted_val))
    extracted_is_empty = extracted_is_none_or_nan or (isinstance(extracted_val, str) and extracted_val == '')

    if actual_is_empty and extracted_is_empty: return True
    # If one is None/NaN/empty and the other isn't, they don't match
    if actual_is_empty != extracted_is_empty: return False
    # If both are non-empty, proceed to type-specific comparison

    is_place_field = field_key in ['place_of_supply', 'place_of_origin']

    try:
        # --- Type-Specific Comparisons ---
        if field_key in METRICS_FLOAT_FIELDS:
            # ... (float comparison) ...
            ext_float = float(extracted_val); act_float = float(actual_val)
            return abs(ext_float - act_float) < METRICS_FLOAT_TOLERANCE
        elif field_key in METRICS_DATE_FIELDS:
            # ... (date comparison) ...
            ext_date = parse_date_robust(str(extracted_val)); act_date = parse_date_robust(str(actual_val))
            if ext_date and act_date: return ext_date == act_date
            else: return False
        elif field_key in METRICS_GSTIN_FIELDS:
            # ... (gstin comparison) ...
            ext_str = clean_text(str(extracted_val)).upper(); act_str = clean_text(str(actual_val)).upper()
            return ext_str == act_str
        elif field_key in METRICS_LIST_FIELDS:
            # ... (list comparison) ...
            try:
                # Ensure extracted_val is treated as a list even if it's a single value string representation from extraction fallback
                if isinstance(extracted_val, (int, float, str)):
                    ext_list_parsed = [float(extracted_val)]
                else: # Assume it's already a list/iterable
                    ext_list_parsed = [float(x) for x in extracted_val if x is not None]
                ext_list = sorted(ext_list_parsed)

                actual_parsed = []
                act_str_clean = str(actual_val).strip()
                if act_str_clean.startswith('[') and act_str_clean.endswith(']'):
                    actual_parsed = sorted([float(x.strip()) for x in act_str_clean[1:-1].split(',') if x.strip()])
                elif act_str_clean: actual_parsed = sorted([float(act_str_clean)])
                return ext_list == actual_parsed
            except (ValueError, TypeError, AttributeError): return False

        else:
            # --- Default String Comparison (Covers place fields) ---
            # Explicitly convert to string here for comparison, even if loaded as string
            ext_str_val = str(extracted_val)
            act_str_val = str(actual_val)

            ext_clean = clean_text(ext_str_val)
            act_clean = clean_text(act_str_val)

            # --- Enhanced Logging ---
            if is_place_field:
                logging.debug(
                    f"Comparing field '{field_key}':\n"
                    f"  Extracted (Original): '{extracted_val}' (Type: {type(extracted_val)})\n"
                    f"  Actual    (Original): '{actual_val}' (Type: {type(actual_val)})\n"
                    f"  Extracted (Cleaned) : '{ext_clean}'\n"
                    f"  Actual    (Cleaned) : '{act_clean}'"
                )

            match_result = ext_clean.lower() == act_clean.lower()
            if is_place_field:
                logging.debug(f"  Comparison Result: {match_result}")
            return match_result

    except (ValueError, TypeError) as e:
        logging.warning(f"Comparison error for field '{field_key}': Extracted='{extracted_val}', Actual='{actual_val}'. Error: {e}")
        return False

def calculate_metrics(extracted_results, ground_truth_df):
    """
    Calculates Precision, Recall, F1 per field and overall accuracy.

    Args:
        extracted_results (list): List of dicts from extract_invoice_data_from_pdf.
        ground_truth_df (pd.DataFrame): DataFrame loaded from actual.csv.

    Returns:
        dict: Contains 'per_field_metrics' and 'overall_accuracy'.
    """
    per_field_metrics = {}
    total_fields_count = 0
    correct_fields_count = 0

        # --- Standardize Ground Truth Date Column --- ADDED ---
    date_col = 'invoice_date_value'
    standardized_dates = []
    if date_col in ground_truth_df.columns:
        logging.info(f"Standardizing ground truth column: {date_col}")
        for date_str in ground_truth_df[date_col]:
            if pd.isna(date_str) or date_str == '':
                standardized_dates.append(None) # Keep None/empty as is
                continue
            try:
                # Parse robustly
                parsed_dt = parse_date_robust(str(date_str))
                if parsed_dt:
                    # Convert to ISO format YYYY-MM-DD
                    standardized_dates.append(parsed_dt.isoformat())
                else:
                    standardized_dates.append(None) # Parsing failed
                    logging.warning(f"Ground truth date parsing failed for: '{date_str}'. Storing as None.")
            except Exception as e:
                standardized_dates.append(None)
                logging.error(f"Error parsing ground truth date '{date_str}': {e}", exc_info=True)
        # Replace the original column or create a new one for comparison
        ground_truth_df[f"{date_col}_standardized"] = standardized_dates
        logging.info(f"Finished standardizing ground truth dates.")
    else:
        logging.warning(f"Ground truth date column '{date_col}' not found. Date metrics might be inaccurate.")
        # Create an empty standardized column to prevent errors later if needed
        ground_truth_df[f"{date_col}_standardized"] = None
    # --- End Date Standardization ---

    # Ensure ground truth columns match expected format
    gt_cols_map = {col: col.replace('_value', '') for col in ground_truth_df.columns if col.endswith('_value')}
    schema_fields_to_compare = [key for key in SCHEMA_KEYS if key not in ['line_items', '_pdf_page_start', '_pdf_page_end']]

    # Initialize metrics counters for each field
    for field_key in schema_fields_to_compare:
        per_field_metrics[field_key] = {'TP': 0, 'FP': 0, 'FN': 0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}

    # Convert extracted results to a DataFrame for easier joining (optional, can iterate)
    # Be careful with list/dict columns like line_items if converting to DF
    # Let's iterate through extracted results and match with ground truth

    processed_gt_indices = set() # Track processed ground truth rows

    for i, extracted_item in enumerate(extracted_results):
        filename = extracted_item.get('_filename')
        inv_index = extracted_item.get('_invoice_index_in_pdf')

        if not filename or inv_index is None:
            logging.warning(f"Skipping metrics for extracted item {i}: Missing filename or index.")
            continue

        # Find corresponding row(s) in ground truth
        gt_match = ground_truth_df[
            (ground_truth_df['filename'] == filename) &
            (ground_truth_df['invoice_index'] == inv_index)
        ]

        if gt_match.empty:
            logging.warning(f"No ground truth match found for: {filename}, Invoice Index: {inv_index}")
            # Treat all non-empty extracted fields as FP for this item? Or ignore?
            # Ignoring unmatched extracted items for now, focusing on matched pairs.
            continue

        if len(gt_match) > 1:
            logging.warning(f"Multiple ground truth matches found for: {filename}, Index: {inv_index}. Using first match.")
        gt_row = gt_match.iloc[0]
        processed_gt_indices.add(gt_row.name) # Mark this GT row as processed


        # Compare each field
        for field_key in schema_fields_to_compare:
            gt_col_name = field_key + '_value'
            if gt_col_name not in gt_row.index:
                 logging.warning(f"Ground truth column '{gt_col_name}' not found.")
                 continue

            actual_value = gt_row[gt_col_name]
            extracted_data = extracted_item.get(field_key, {}) # Get the {'value': ..., 'confidence': ..., 'trusted': ...} dict
            extracted_value = extracted_data.get('value') # Extract the actual value

            # --- TP, FP, FN Logic ---
            is_match = compare_values(extracted_value, actual_value, field_key)
            actual_is_present = not (pd.isna(actual_value) or actual_value == '')
            extracted_is_present = extracted_value is not None and extracted_value != ''


            if is_match and actual_is_present: # Correctly extracted existing value
                per_field_metrics[field_key]['TP'] += 1
                correct_fields_count += 1
            elif extracted_is_present and not actual_is_present: # Extracted something that shouldn't be there
                per_field_metrics[field_key]['FP'] += 1
            elif extracted_is_present and actual_is_present and not is_match: # Extracted wrong value
                per_field_metrics[field_key]['FP'] += 1 # Counts as FP (extracted something incorrect)
                per_field_metrics[field_key]['FN'] += 1 # Also counts as FN (missed the correct value)
            elif not extracted_is_present and actual_is_present: # Failed to extract existing value
                per_field_metrics[field_key]['FN'] += 1
            # Case: Both are empty/None - Handled by compare_values returning True, but doesn't increment TP/FP/FN unless actual_is_present.

            # Increment total fields count only if the ground truth had a value
            if actual_is_present:
                 total_fields_count += 1


    # --- Handle Unmatched Ground Truth Rows ---
    # Any GT rows not matched correspond to invoices the script failed to find/segment correctly.
    # All fields in these unmatched GT rows count as FN.
    unmatched_gt = ground_truth_df.drop(index=list(processed_gt_indices))
    for idx, gt_row in unmatched_gt.iterrows():
        for field_key in schema_fields_to_compare:
             gt_col_name = field_key + '_value'
             if gt_col_name in gt_row.index:
                 actual_value = gt_row[gt_col_name]
                 actual_is_present = not (pd.isna(actual_value) or actual_value == '')
                 if actual_is_present:
                     per_field_metrics[field_key]['FN'] += 1
                     total_fields_count += 1 # Count these fields as they existed in GT


    # --- Calculate P, R, F1 per field ---
    for field_key in schema_fields_to_compare:
        tp = per_field_metrics[field_key]['TP']
        fp = per_field_metrics[field_key]['FP']
        fn = per_field_metrics[field_key]['FN']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_field_metrics[field_key]['Precision'] = round(precision, 4)
        per_field_metrics[field_key]['Recall'] = round(recall, 4)
        per_field_metrics[field_key]['F1'] = round(f1, 4)

    # --- Calculate Overall Accuracy ---
    # Defined as: (Total Correctly Extracted Fields) / (Total Fields Present in Ground Truth)
    overall_accuracy = correct_fields_count / total_fields_count if total_fields_count > 0 else 0.0

    return {
        "per_field_metrics": per_field_metrics,
        "overall_accuracy": round(overall_accuracy, 4),
        "summary": {
             "total_ground_truth_fields_present": total_fields_count,
             "total_correctly_extracted_fields": correct_fields_count
        }
    }


# --- Main Execution (Updated) ---
if __name__ == "__main__":
    pdf_directory = "./testing"  # Default directory
    output_file_json = "invoice_extraction_hybrid_results.json"
    output_file_excel = "invoice_extraction_hybrid_results.xlsx"
    metrics_output_file = "extraction_metrics_report.json"

    if not os.path.isdir(pdf_directory):
        logging.error(f"Error: Directory not found: {pdf_directory}")
        exit()
    if not AWS_REGION:
         logging.warning("Warning: AWS_REGION is not set. Image-based PDFs will fail.")
         # exit() # Allow running without AWS for text-only tests

    all_results_flat = [] # Will store one dict per detected invoice
    for filename in sorted(os.listdir(pdf_directory)):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            invoice_results_list = extract_invoice_data_from_pdf(filepath)
            all_results_flat.extend(invoice_results_list)

    # --- Save Extraction Results ---
    try:
        with open(output_file_json, 'w', encoding='utf-8') as f:
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (datetime.date, datetime.datetime)): return obj.isoformat()
                    if isinstance(obj, defaultdict): return dict(obj)
                    # Handle potential numpy types if they sneak in
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                        np.int16, np.int32, np.int64, np.uint8,
                                        np.uint16, np.uint32, np.uint64)): return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
                    elif isinstance(obj, (np.ndarray,)): return obj.tolist() # Convert arrays to lists
                    elif isinstance(obj, np.bool_): return bool(obj)
                    elif isinstance(obj, (np.void)): return None # Handle numpy void type
                    try: return super().default(obj)
                    except TypeError: return str(obj)
            json.dump(all_results_flat, f, indent=4, ensure_ascii=False, cls=CustomEncoder)
        logging.info(f"\nExtraction complete. Results saved to {output_file_json}")
    except Exception as e: logging.error(f"\nError saving results to JSON: {e}", exc_info=True)

    # Save results to Excel (Keep existing logic)
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
             for key in SCHEMA_KEYS:
                 if key not in ['_pdf_page_start', '_pdf_page_end']:
                     data = res.get(key)
                     if key == 'line_items':
                         try: row[key] = json.dumps(data, cls=CustomEncoder) if data is not None else '[]'
                         except Exception: row[key] = '[]'
                     elif isinstance(data, dict): # Expected structure
                         row[f"{key}_value"] = data.get("value")
                         row[f"{key}_confidence"] = data.get("confidence")
                         row[f"{key}_trusted"] = data.get("trusted")
                     else:
                          row[f"{key}_value"] = None
                          row[f"{key}_confidence"] = 0.0
                          row[f"{key}_trusted"] = False
             excel_rows.append(row)
        df_excel = pd.DataFrame(excel_rows)
        cols_order = ['filename', 'invoice_index', 'page_start', 'page_end', 'error']
        value_cols, conf_cols, trust_cols, other_cols = [], [], [], []
        for skey in SCHEMA_KEYS:
             if skey not in ['_pdf_page_start', '_pdf_page_end']:
                 if skey == 'line_items': other_cols.append(skey)
                 else: value_cols.append(f"{skey}_value"); conf_cols.append(f"{skey}_confidence"); trust_cols.append(f"{skey}_trusted")
        cols_order.extend(value_cols); cols_order.extend(conf_cols); cols_order.extend(trust_cols); cols_order.extend(other_cols)
        final_cols = [c for c in cols_order if c in df_excel.columns]
        df_excel = df_excel[final_cols]
        df_excel.to_excel(output_file_excel, index=False, engine='openpyxl') # Specify engine
        logging.info(f"Results also saved to Excel: {output_file_excel}")
    except Exception as e: logging.error(f"Could not save results to Excel: {e}", exc_info=True)


    # --- Calculate and Save Metrics ---
    if os.path.exists(GROUND_TRUTH_CSV):
        logging.info(f"\nCalculating metrics against ground truth: {GROUND_TRUTH_CSV}")
        try:
            gt_dtypes = {
                'filename': str,
                'invoice_index': int, # Keep index as int if it's purely numeric
                'place_of_supply_value': str,  # Force place codes to string
                'place_of_origin_value': str,  # Force place codes to string
                'gstin_supplier_value': str,   # GSTINs should also be strings
                'gstin_recipient_value': str,
                'invoice_number_value': str,
                # Add any other columns that might look numeric but should be string
            }
            ground_truth_df = pd.read_csv(GROUND_TRUTH_CSV, dtype=gt_dtypes)
            logging.info(f"Ground truth dtypes for place codes: PoS={ground_truth_df['place_of_supply_value'].dtype}, PoO={ground_truth_df['place_of_origin_value'].dtype}") # Log dtype after loading
            # Basic validation of ground truth file structure
            if 'filename' not in ground_truth_df.columns or 'invoice_index' not in ground_truth_df.columns:
                 logging.error(f"Ground truth file '{GROUND_TRUTH_CSV}' is missing required columns 'filename' or 'invoice_index'. Skipping metrics.")
            else:
                 # Ensure invoice_index is integer
                 ground_truth_df['invoice_index'] = ground_truth_df['invoice_index'].astype(int)

                 metrics_results = calculate_metrics(all_results_flat, ground_truth_df)

                 # Save metrics report
                 with open(metrics_output_file, 'w', encoding='utf-8') as f:
                     json.dump(metrics_results, f, indent=4, cls=CustomEncoder)
                 logging.info(f"Metrics report saved to {metrics_output_file}")

                 # Print a summary to console
                 print("\n--- Extraction Metrics Summary ---")
                 print(f"Overall Accuracy (Correct Fields / Total GT Fields): {metrics_results['overall_accuracy']:.2%}")
                 print(f"Total Fields in Ground Truth (Present): {metrics_results['summary']['total_ground_truth_fields_present']}")
                 print(f"Total Fields Correctly Extracted: {metrics_results['summary']['total_correctly_extracted_fields']}")
                 print("\nPer-Field F1-Scores:")
                 # Sort fields by F1 score for better readability
                 sorted_fields = sorted(metrics_results['per_field_metrics'].items(), key=lambda item: item[1]['F1'], reverse=True)
                 for field, scores in sorted_fields:
                     print(f"  - {field:<25}: F1={scores['F1']:.4f} (P={scores['Precision']:.4f}, R={scores['Recall']:.4f}) (TP:{scores['TP']}, FP:{scores['FP']}, FN:{scores['FN']})")
                 print("---------------------------------")

        except FileNotFoundError:
            logging.warning(f"Ground truth file '{GROUND_TRUTH_CSV}' not found. Skipping metrics calculation.")
        except Exception as e:
            logging.error(f"Error during metrics calculation: {e}", exc_info=True)
    else:
        logging.warning(f"Ground truth file '{GROUND_TRUTH_CSV}' not found. Skipping metrics calculation.")

