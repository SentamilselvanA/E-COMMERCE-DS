import nbformat, json

with open('DS_Day01_34_ECommerce_Customers.ipynb', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cells = [c for c in nb.cells if c.cell_type == 'code']
print('Total code cells:', len(code_cells))
print()

for i, cell in enumerate(code_cells):
    outputs = cell.get('outputs', [])
    parts = []
    for o in outputs:
        otype = o.get('output_type', '')
        if otype == 'stream':
            parts.append(''.join(o.get('text', [])))
        elif otype in ('execute_result', 'display_data'):
            data = o.get('data', {})
            parts.append(''.join(data.get('text/plain', [])))
    combined = ''.join(parts).strip()
    if combined:
        print(f'--- Cell {i+1} ---')
        print(combined[:400])
        print()
