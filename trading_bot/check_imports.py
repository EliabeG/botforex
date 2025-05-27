import os
import ast
import sys

def get_imports(filepath):
    imports = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except:
        pass
    return imports

# Coletar todos os imports
all_imports = set()
local_modules = set()

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git']]
    
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            module_name = filepath[2:].replace('/', '.').replace('.py', '')
            local_modules.add(module_name.split('.')[0])
            all_imports.update(get_imports(filepath))

# Filtrar imports externos
external_imports = all_imports - local_modules - {'__future__'}

print("📦 External dependencies found in code:")
for imp in sorted(external_imports):
    print(f"  - {imp}")

# Verificar se estão no requirements.txt
print("\n📋 Checking requirements.txt...")
if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as f:
        requirements = f.read().lower()
    
    missing = []
    for imp in external_imports:
        if imp.lower() not in requirements:
            missing.append(imp)
    
    if missing:
        print(f"\n⚠️  Possibly missing from requirements.txt:")
        for m in missing:
            print(f"  - {m}")
else:
    print("❌ requirements.txt not found!")
