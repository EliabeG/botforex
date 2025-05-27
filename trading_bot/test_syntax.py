import os
import ast
import sys

def check_syntax(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

errors = []
for root, dirs, files in os.walk('.'):
    # Skip virtual environments and cache
    dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'logs', 'data']]
    
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            success, error = check_syntax(filepath)
            if not success:
                errors.append(f"{filepath}: {error}")
                print(f"❌ {filepath}: {error}")
            else:
                print(f"✅ {filepath}")

if errors:
    print(f"\n❌ Found {len(errors)} syntax errors!")
    sys.exit(1)
else:
    print("\n✅ All Python files have valid syntax!")
