import re 
import json
import os
import time
import argparse
import ast
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from visualize.generate_diagrams import generate_architecture_diagrams

# ===============================
# CONFIG
# ===============================

# Defaults (can be overridden by CLI)
INPUT_ROOT = Path("documentation2")
OUTPUT_ROOT = Path("architecture_output")
SOURCE_ROOT = None
BATCH_SIZE = 50  # Default batch size, can be overridden via CLI
LOG_FILE = None  # Optional log file path

# Get API key from environment (no hardcoded fallback for security)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable must be set. Do not hardcode API keys in source code.")

# ===============================
# LLM SETUP
# ===============================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY,
)

def call_llm(prompt: str) -> str:
    max_retries = 5
    base_delay = 10  # Start with 10 seconds 
    transient_errors = ["429", "RESOURCE_EXHAUSTED", "quota", "timeout", "connection", "network", "503", "502", "500"]

    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit or transient errors
            is_transient = any(err in error_str for err in transient_errors)
            
            if is_transient:
                # Exponential backoff for transient errors
                wait_time = base_delay * (2 ** attempt)
                logging.warning(f"LLM transient error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                print(f"[WARN] LLM transient error. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # Non-transient errors: retry a few times with backoff, then fail
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logging.warning(f"LLM error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    print(f"[WARN] LLM error: {e}. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logging.error(f"LLM failed after {max_retries} retries: {e}")
                    print(f"[ERROR] LLM failed after {max_retries} retries: {e}")
                    return ""
    
    logging.error(f"LLM failed after {max_retries} retries.")
    print(f"[ERROR] LLM failed after {max_retries} retries.")
    return ""

# ===============================
# CODE SCANNER (AST)
# ===============================

# ===============================
# CODE SCANNER (Polyglot)
# ===============================

class CodeScanner:
    
    # Simple regex patterns for other languages (Heuristic-based)
    # This is not a perfect parser, but good enough for architectural signals.
    LANGUAGE_PATTERNS = {
        ".js": {
            "class": r"class\s+(\w+)",
            "function": r"function\s+(\w+)|const\s+(\w+)\s*=\s*\(|(\w+)\s*:\s*function",
            "import": r"import\s+.*?from\s+['\"](.*?)['\"]|require\(['\"](.*?)['\"]",
        },
        ".ts": {
            "class": r"class\s+(\w+)",
            "function": r"function\s+(\w+)|const\s+(\w+)\s*=\s*\(|(\w+)\s*\(",
            "import": r"import\s+.*?from\s+['\"](.*?)['\"]",
        },
        ".java": {
            "class": r"class\s+(\w+)",
            "function": r"(?:public|private|protected).*?\s+(\w+)\s*\(",
            "import": r"import\s+([\w\.]+);",
        },
        ".cs": {
            "class": r"class\s+(\w+)",
            "function": r"(?:public|private|protected).*?\s+(\w+)\s*\(",
            "import": r"using\s+([\w\.]+);",
        },
        ".go": {
            "class": r"type\s+(\w+)\s+struct",
            "function": r"func\s+(\w+)\s*\(",
            "import": r"import\s+[\(]?\s*['\"](.*?)['\"]",
        },
        ".cpp": {
            "class": r"class\s+(\w+)",
            "function": r"\w+\s+(\w+)\s*\(", # Very broad, might be noisy
            "import": r"#include\s+[<\"](.*?)[>\"]",
        }
    }

    @staticmethod
    def scan_file(file_path: Path) -> str:
        """Scans a single file (Polyglot) and returns a summary of its structure."""
        if not file_path.exists():
            return ""
        
        ext = file_path.suffix.lower()
        
        # 1. Python Strategy (AST - Best quality)
        if ext == ".py":
            return CodeScanner._scan_python(file_path)
        
        # 2. Regex Strategy (Other languages)
        elif ext in CodeScanner.LANGUAGE_PATTERNS:
            return CodeScanner._scan_regex(file_path, ext)
            
        return ""

    @staticmethod
    def _scan_python(file_path: Path) -> str:
        try:
            code = file_path.read_text(encoding="utf-8")
            tree = ast.parse(code)
        except Exception as e:
            return f"Error parsing {file_path.name}: {e}"

        summary = []
        
        # Imports - Fixed IndexError by checking if names exist
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and node.names:
                imports.extend([alias.name for alias in node.names])
        from_imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
        all_deps = list(set(imports + from_imports))
        if all_deps:
            summary.append(f"Imports: {', '.join(all_deps[:10])}...")

        # Classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                summary.append(f"Class: {node.name}({', '.join(bases)})")
                
                # Methods
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                if methods:
                    summary.append(f"  Methods: {', '.join(methods[:10])}")

        # Functions (Top-level)
        funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        if funcs:
            summary.append(f"Functions: {', '.join(funcs[:5])}")

        return "\n".join(summary)

    @staticmethod
    def _scan_regex(file_path: Path, ext: str) -> str:
        try:
            code = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logging.warning(f"Failed to read file {file_path}: {e}")
            return ""

        patterns = CodeScanner.LANGUAGE_PATTERNS.get(ext, {})
        summary = []
        
        # Scan with Regex
        if "import" in patterns:
            deps = re.findall(patterns["import"], code)
            # Flatten tuples if finding multiple groups
            deps = [d if isinstance(d, str) else next((x for x in d if x), "") for d in deps]
            deps = list(set(d for d in deps if d))
            if deps:
                 summary.append(f"Imports: {', '.join(deps[:10])}...")

        if "class" in patterns:
            classes = re.findall(patterns["class"], code)
            for c in classes:
                # Handle regex groups
                c_name = c[0] if isinstance(c, tuple) else c
                summary.append(f"Class: {c_name}")
        
        if "function" in patterns:
            funcs = re.findall(patterns["function"], code)
            # Clean up tuple results from multiple groups
            clean_funcs = []
            for f in funcs:
                if isinstance(f, tuple):
                    clean_funcs.extend([x for x in f if x])
                else:
                    clean_funcs.append(f)
            
            if clean_funcs:
                summary.append(f"Functions: {', '.join(clean_funcs[:10])}")

        return "\n".join(summary)
    
    @staticmethod
    def find_source_file(md_path: Path, source_root: Path) -> Optional[Path]:
        """Tries to find the corresponding source file for a given .md file."""
        if not source_root:
            return None
        
        # Support multiple extensions for lookup
        extensions = [".py", ".js", ".ts", ".java", ".cs", ".go", ".cpp", ".h"]
        
        potential_names = [md_path.name.replace(".md", "")] # exact mapping name.py.md -> name.py
        
        # name.md -> name.py / name.js / name.java
        stem = md_path.stem
        for ext in extensions:
            potential_names.append(stem + ext)
            
        potential_names.append(stem) # folder match or exact name match
        
        # Fixed path resolution with error handling
        try:
            rel_path = md_path.parent.relative_to(INPUT_ROOT)
        except ValueError:
            # md_path is not under INPUT_ROOT, try direct lookup
            rel_path = Path("")
        
        for name in potential_names:
            candidate = source_root / rel_path / name
            if candidate.exists():
                return candidate
            
        return None

# ===============================
# MARKDOWN PARSING
# ===============================

SECTION_ALIASES = {
    "summary": ["Summary", "Overview", "Description"],
    "functional_explanation": ["Functional Explanation", "Functionality", "Behavior"],
    "external_dependencies": ["External Dependencies", "Dependencies"]
}

def extract_sections(md_text: str) -> dict:
    def grab_any(titles):
        for t in titles:
            m = re.search(rf"### {re.escape(t)}\n(.*?)(?=\n###|\Z)", md_text, re.S)
            if m:
                return m.group(1).strip()
        return ""

    def grab_list(titles):
        content = grab_any(titles)
        return [
            l.strip("* -").strip()
            for l in content.splitlines()
            if l.strip().startswith(("*", "-"))
        ] if content else []

    return {
        "summary": grab_any(SECTION_ALIASES["summary"]),
        "functional_explanation": grab_any(SECTION_ALIASES["functional_explanation"]),
        "external_dependencies": grab_list(SECTION_ALIASES["external_dependencies"]),
    }



def parse_md_file(path: Path) -> Optional[Dict[str, Any]]:
    """Parse a markdown file and extract sections. Returns None if file is empty or invalid."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Failed to read markdown file {path}: {e}")
        print(f"[ERROR] Failed to read {path}: {e}")
        return None
    
    sections = extract_sections(text)
    
    # Validate that file has meaningful content
    combined = (sections["summary"] + " " + sections["functional_explanation"]).strip()
    if len(combined) < 10:  # Very short content, likely empty or invalid
        logging.debug(f"Skipping empty/invalid markdown file: {path}")
        return None

    # Try to find and scan source code
    code_signals = ""
    if SOURCE_ROOT:
        src_file = CodeScanner.find_source_file(path, SOURCE_ROOT)
        if src_file:
            print(f"   + Scanned source: {src_file.name}")
            code_signals = CodeScanner.scan_file(src_file)

    return {
        "file": path.stem,
        "path": str(path),
        **sections,
        "code_signals": code_signals
    }

def parse_md_folder(folder: Path) -> List[Dict[str, Any]]:
    """Parse all markdown files in a folder, filtering out empty/invalid ones."""
    results = []
    for md in folder.glob("*.md"):
        parsed = parse_md_file(md)
        if parsed:  # Only include non-None results
            results.append(parsed)
    return results

# ===============================
# JSON SAFETY
# ===============================

def safe_json_parse(text: str) -> Dict[str, Any]:
    """Improved JSON parsing that handles complex nested JSON and multiple JSON objects."""
    if not text or not text.strip():
        return {}
    
    # Try to find JSON object boundaries more accurately
    # Look for balanced braces
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # Found a complete JSON object
                try:
                    json_str = text[start_idx:i+1]
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try next JSON object if this one fails
                    start_idx = -1
                    continue
    
    # Fallback: try simple regex (original method)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    
    return {}

def ensure_component_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "component_name": "Unknown",
        "architectural_description": "Unknown",
        "role_classification": "Unknown",
        "architectural_tier": "Unknown",
        "responsibilities": [],
        "non_responsibilities": [],
        "stateful_or_stateless": "Unknown",
        "owned_data_or_state": [],
        "allowed_incoming_interactions": [],
        "allowed_outgoing_interactions": [],
        "restricted_interactions": [],
        "external_systems_or_services": [],
        "folder_paths": [],
        "primary_files": [],
        "criticality_level": "Unknown",
        "impact_of_failure": "Unknown"
    }

    # apply base defaults
    for k, v in defaults.items():
        data.setdefault(k, v)
    
    # --- SMART DEFAULTS FOR STATIC/VENDOR ---
    # Heuristic: If the name suggests a static utility, fill in gaps instead of leaving "Unknown"
    name = data.get("component_name", "").lower()
    is_static = any(x in name for x in ["static", "assets", "css", "js", "img", "images", "vendor", "lib", "font", "style"])
    
    if is_static:
        if data["role_classification"] == "Unknown":
            data["role_classification"] = "Utility"
        if data["architectural_tier"] == "Unknown":
            data["architectural_tier"] = "Peripheral"
        if data["stateful_or_stateless"] == "Unknown":
            data["stateful_or_stateless"] = "Stateless"
        if data["criticality_level"] == "Unknown":
            data["criticality_level"] = "Medium"
        if data["impact_of_failure"] == "Unknown":
            data["impact_of_failure"] = "Visual degradation or reduced client-side interactivity"
        if not data["responsibilities"]:
             data["responsibilities"] = ["Serve static resources to the client"]

    return data

def ensure_system_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure system architecture JSON has all required keys with defaults."""
    defaults = {
        "system_name": "Unknown",
        "system_purpose": "Unknown",
        "high_level_description": "Unknown",
        "interaction_interfaces": [],
        "architectural_components": [],
        "data_entities": [],
        "primary_flows": [],
        "secondary_flows": [],
        "dependencies": [],
        "components_inside": [],
        "dependencies_outside": [],
        "constraints": "",
        "exclusions": ""
    }
    
    # Apply base defaults
    for k, v in defaults.items():
        data.setdefault(k, v)
    
    return data

# ===============================
# CHILD SUMMARY
# ===============================

def summarize_capability(responsibilities: List[str]) -> str:
    return responsibilities[0] if responsibilities else "General processing"

def summarize_children(children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for c in children:
        if not c or "data" not in c:
            continue
        data = c["data"]
        
        # System architecture might be flat or have "components" key
        if c.get("type") == "system":
             summary.append({
                "component_name": data.get("system_name", "Sub-System"),
                "role": "System",
                "capability": "High-level System Logic"
             })
             continue

        # Component architecture is typically { "FolderName": { ... } }
        if isinstance(data, dict):
            # Check if it is directly the component dict
            if "component_name" in data:
                 summary.append({
                    "component_name": data.get("component_name"),
                    "role": data.get("role_classification"),
                    "capability": summarize_capability(data.get("responsibilities", []))
                })
            else:
                # Nested wrapper keys
                for v in data.values():
                    if isinstance(v, dict) and "component_name" in v:
                        summary.append({
                            "component_name": v.get("component_name"),
                            "role": v.get("role_classification"),
                            "capability": summarize_capability(v.get("responsibilities", []))
                        })
    return summary

# ===============================
# BLUEPRINTS
# ===============================

COMPONENT_BLUEPRINT = """
## 1. Component Identity
- component_name (Must match folder name)
- architectural_description

## 2. Architectural Role
- role_classification (Interface / Processing / Storage / Orchestration / Utility)
- architectural_tier (Core / Peripheral)

## 3. Responsibility Scope
- responsibilities (list)
- non_responsibilities (list)

## 4. State & Data Ownership
- stateful_or_stateless
- owned_data_or_state (if any)

## 5. Interaction Rules
- allowed_incoming_interactions (list)
- allowed_outgoing_interactions (list)
- restricted_interactions (list)

## 6. External Touchpoints
- external_systems_or_services (list)
- purpose_of_each_touchpoint (list)

## 7. Implementation Mapping
- folder_paths (list)
- primary_files (list)
- supporting_files (list)
- responsibility_of_each_file (list)

## 8. Criticality & Failure Impact
- criticality_level
- impact_of_failure
"""

SYSTEM_BLUEPRINT = """
## 1. System Context
- system_name
- system_purpose
- high_level_description

## 2. Interaction Interfaces
- interfaces (list of objects with: type, identifier, trigger, relationship)

## 3. Architectural Components
- components (list of objects with: name, responsibility, type, criticality_level)

## 4. Data & State Model
- data_entities (list of objects with: name, owner, nature, storage)

## 5. Control & Interaction Flows
- primary_flows (list of objects with: purpose, initiating_interface, interaction_path)
- secondary_flows (list of objects with: purpose, initiating_interface, interaction_path)

## 6. External Dependencies
- dependencies (list of objects with: name, category, purpose, optionality)

## 7. System Boundary
- components_inside
- dependencies_outside

## 8. Constraints & Exclusions
- constraints
- exclusions
"""

# ===============================
# SCALABLE GENERATION 
# ===============================

def generate_rolling_architecture(name: str, files: List[Dict[str, Any]], children: List[Dict[str, Any]], 
                                  blueprint: str, type_hint: str = "component", batch_size: int = 50) -> Dict[str, Any]:
    """
    Handles generation using a Rolling Window strategy to support infinite files.
    """
    
    # 1. Chunking
    chunks = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    current_context: Dict[str, Any] = {}  # Starts empty
    failed_batches = 0
    
    total_chunks = len(chunks)
    
    for i, batch in enumerate(chunks):
        is_first = (i == 0)
        progress_pct = int((i + 1) / total_chunks * 100) if total_chunks > 0 else 0
        
        print(f"   > Processing Batch {i+1}/{total_chunks} ({len(batch)} files)... [{progress_pct}%]")
        
        # Prepare Batch Data
        batch_files_json = json.dumps(batch, separators=(',', ':'))
        batch_signals_json = json.dumps([f['code_signals'] for f in batch if f.get('code_signals')], separators=(',', ':'))
        
        # Construct Prompt
        if is_first:
            # Standard Initial Prompt
            prompt = f"""
Generate {type_hint.upper()} ARCHITECTURE JSON for "{name}".

Context Files (Batch {i+1}/{total_chunks}):
{batch_files_json}

## GROUND TRUTH CODE SIGNALS:
{batch_signals_json}

Child Components:
{json.dumps(summarize_children(children), separators=(',', ':'))}

BLUEPRINT:
{blueprint}

Rules:
- Return STRICT JSON only.
- If information is missing, use "Unknown".
- Infer "Utility" role for static/assets.
"""
        else:
            # Incremental Update Prompt
            prompt = f"""
UPDATE {type_hint.upper()} ARCHITECTURE JSON for "{name}".

PREVIOUS ARCHITECTURE CONTEXT:
{json.dumps(current_context, separators=(',', ':'))}

NEW FILES TO MERGE (Batch {i+1}/{total_chunks}):
{batch_files_json}

## GROUND TRUTH CODE SIGNALS:
{batch_signals_json}

TASK:
- Merge the new files into the existing architecture.
- Update responsibilities, interactions, and data models.
- Maintain consistency.

BLUEPRINT:
{blueprint}

Rules:
- Return the UPDATED STRICT JSON only.
"""

        # Call LLM
        response = call_llm(prompt)
        parsed = safe_json_parse(response)
        
        if parsed:
            # For component generation, we expect { "Folder": { ... } }
            # For system generation, we expect { ... }
            # We need to normalize current_context to be the inner data
            
            if type_hint == "component":
                if name in parsed:
                    parsed_data = parsed[name]
                    # Validate component name matches folder name
                    if parsed_data.get("component_name") != name:
                        logging.warning(f"Component name mismatch: expected '{name}', got '{parsed_data.get('component_name')}'. Correcting...")
                        parsed_data["component_name"] = name
                    current_context = ensure_component_keys(parsed_data)
                else:
                    # Fallback if LLM forgot the root key
                    parsed["component_name"] = name  # Ensure correct name
                    current_context = ensure_component_keys(parsed)
            else:
                current_context = parsed
            failed_batches = 0  # Reset counter on success
        else:
            failed_batches += 1
            logging.warning(f"Batch {i+1} failed to parse. Keeping previous context.")
            print(f"   [WARN] Batch {i+1} failed to parse. Keeping previous context.")
            
            # If too many batches fail in a row, warn user
            if failed_batches >= 3:
                logging.error(f"Multiple consecutive batches failed ({failed_batches}). Architecture may be incomplete.")
                print(f"   [ERROR] {failed_batches} consecutive batches failed. Architecture may be incomplete.")

    # Final Safety & Defaults
    if type_hint == "component":
         # If context is empty (LLM failed all batches), seed it with name
         if not current_context:
             logging.warning(f"All batches failed for component '{name}'. Generating minimal architecture.")
             print(f"   [WARN] All batches failed for '{name}'. Generating minimal architecture.")
             current_context = {"component_name": name}
         
         # Apply Smart Defaults
         current_context = ensure_component_keys(current_context)
         
         # Validate minimum quality - ensure we have more than just the name
         if len(str(current_context.get("architectural_description", ""))) < 10:
             logging.warning(f"Component '{name}' has minimal architecture data. Consider reviewing input files.")
         
         return {"type": "component", "data": {name: current_context}}
    else:
         # Apply system defaults and validation
         if not current_context:
             logging.warning(f"All batches failed for system '{name}'. Generating minimal architecture.")
             print(f"   [WARN] All batches failed for system '{name}'. Generating minimal architecture.")
             current_context = {"system_name": name}
         
         current_context = ensure_system_keys(current_context)
         
         # Validate minimum quality
         if len(str(current_context.get("system_purpose", ""))) < 10:
             logging.warning(f"System '{name}' has minimal architecture data. Consider reviewing input files.")
         
         return {"type": "system", "data": current_context}


def generate_component(folder_name: str, files: List[Dict[str, Any]], children: List[Dict[str, Any]], 
                       batch_size: int = 50) -> Optional[Dict[str, Any]]:
    # Signal checks - ignore low-info folders
    signal = sum(len(f.get("summary", "") + f.get("functional_explanation", "")) for f in files)
    if signal < 50 and not children:
        return None

    return generate_rolling_architecture(folder_name, files, children, COMPONENT_BLUEPRINT, "component", batch_size)

def generate_system(name: str, files: List[Dict[str, Any]], children: List[Dict[str, Any]], 
                    batch_size: int = 50) -> Dict[str, Any]:
    return generate_rolling_architecture(name, files, children, SYSTEM_BLUEPRINT, "system", batch_size)

# ===============================
# RECURSIVE WALK
# ===============================

def process_folder(current: Path, root: Path, batch_size: int = 50) -> Optional[Dict[str, Any]]:
    print(f"Processing: {current}")
    children = []
    
    # Process children first (Bottom-Up)
    try:
        dirs = [d for d in current.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"]
    except Exception as e:
        logging.error(f"Failed to list directory {current}: {e}")
        print(f"[ERROR] Failed to list directory {current}: {e}")
        return None
    
    for d in dirs:
        try:
            child = process_folder(d, root, batch_size)
            if child:
                children.append(child)
        except Exception as e:
            logging.error(f"Error processing {d}: {e}")
            print(f"[WARN] Error processing {d}: {e}")

    files = parse_md_folder(current)
    
    # Skip empty leaf nodes
    if not files and not children:
        return None

    is_root = (current == root)
    name = current.name

    if is_root:
        out_file = OUTPUT_ROOT / f"{name}.json" # Root usually sits at top
        
        # Check Cache with error handling
        if out_file.exists() and not FORCE_GENERATE:
            try:
                cached_data = json.loads(out_file.read_text(encoding="utf-8"))
                print(f"[SKIP] {name} already exists.")
                return {"type": "system", "data": cached_data}
            except Exception as e:
                logging.warning(f"Failed to read cached file {out_file}: {e}. Regenerating...")
                print(f"[WARN] Failed to read cached file. Regenerating...")

        arch = generate_system(name, files, children, batch_size)
    else:
        # Determine output path structure with error handling
        try:
            rel_path = current.relative_to(root.parent)
        except ValueError:
            # Fallback: use current directory name if relative path fails
            logging.warning(f"Could not compute relative path for {current}. Using direct path.")
            rel_path = Path(name)
        
        out_dir = OUTPUT_ROOT / rel_path
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{name}.json"

        # Check Cache with error handling
        if out_file.exists() and not FORCE_GENERATE:
            try:
                cached_data = json.loads(out_file.read_text(encoding="utf-8"))
                print(f"[SKIP] {name} already exists.")
                return {"type": "component", "data": cached_data}
            except Exception as e:
                logging.warning(f"Failed to read cached file {out_file}: {e}. Regenerating...")
                print(f"[WARN] Failed to read cached file. Regenerating...")

        arch = generate_component(name, files, children, batch_size)

    if arch:
        # Save specific data part for cleanliness, or whole arch?
        # User's reference saved arch["data"]
        # For non-root, we already calculated out_file. For root, we need to do it too.
        if is_root:
             out_dir = OUTPUT_ROOT
             out_dir.mkdir(parents=True, exist_ok=True)
             out_file = out_dir / f"{name}.json"
        
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(arch["data"], f, indent=2)
            print(f"Saved: {out_file}")
        except Exception as e:
            logging.error(f"Failed to save architecture to {out_file}: {e}")
            print(f"[ERROR] Failed to save {out_file}: {e}")

    return arch

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Architectural Analysis Engine")
    parser.add_argument("--input", default="documentation2", help="Input folder containing .md files")
    parser.add_argument("--output", default="architecture_output", help="Output folder for JSON artifacts")
    parser.add_argument("--source", default=None, help="Root folder of actual source code (for AST scanning)")
    parser.add_argument("--force", action="store_true", help="Force re-generation of all artifacts")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of files to process per batch")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    
    args = parser.parse_args()
    
    INPUT_ROOT = Path(args.input)
    OUTPUT_ROOT = Path(args.output)
    FORCE_GENERATE = args.force
    BATCH_SIZE = args.batch_size
    LOG_FILE = args.log_file

    # Setup logging
    log_level = logging.INFO
    handlers = [logging.StreamHandler()]
    if LOG_FILE:
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

    if args.source:
        SOURCE_ROOT = Path(args.source)
        if not SOURCE_ROOT.exists():
            logging.warning(f"Source path {SOURCE_ROOT} does not exist. Skipping code scanning.")
            print(f"[WARN] Source path {SOURCE_ROOT} does not exist. Skipping code scanning.")
            SOURCE_ROOT = None
    else:
        SOURCE_ROOT = None

    if not INPUT_ROOT.exists():
        logging.error(f"Input directory '{INPUT_ROOT}' does not exist.")
        print(f"Error: Input directory '{INPUT_ROOT}' does not exist.")
    else:
        print(f"Input: {INPUT_ROOT}")
        print(f"Source: {SOURCE_ROOT if SOURCE_ROOT else 'Not provided (Docs-only mode)'}")
        print(f"Batch Size: {BATCH_SIZE}")
        if LOG_FILE:
            print(f"Logging to: {LOG_FILE}")
        
        try:
            process_folder(INPUT_ROOT, INPUT_ROOT, BATCH_SIZE)
            print("Architecture generation complete.")

            generate_architecture_diagrams(call_llm)
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            logging.exception("Fatal error during architecture generation")
            print(f"[ERROR] Fatal error: {e}")
            raise







