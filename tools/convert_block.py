import re
import json

def parse_c_structs(c_filename):
    with open(c_filename, 'r') as f:
        c_code = f.read()

    # Regex for block struct
    block_re = re.compile(
        r'block\s+(\w+)\s*=\s*{\s*'
        r'\.width\s*=\s*(\d+),\s*'
        r'\.height\s*=\s*(\d+),\s*'
        r'\.centerX\s*=\s*(-?\d+),\s*'
        r'\.centerY\s*=\s*(-?\d+),\s*'
        r'\.pattern\s*=\s*{((?:\s*{[01,\s]+},?)+)\s*}'
        r'\s*};',
        re.MULTILINE
    )

    blocks = []
    for match in block_re.finditer(c_code):
        name = match.group(1)
        width = int(match.group(2))
        height = int(match.group(3))
        centerX = int(match.group(4))
        centerY = int(match.group(5))
        pattern_raw = match.group(6)

        # Extract pattern rows
        pattern_rows = re.findall(r'{([01,\s]+)}', pattern_raw)
        pattern = [
            [int(x) for x in row.replace(' ', '').split(',') if x]
            for row in pattern_rows
        ]

        blocks.append({
            "name": name,
            "width": width,
            "height": height,
            "centerX": centerX,
            "centerY": centerY,
            "pattern": pattern
        })

    return blocks

if __name__ == "__main__":
    blocks = parse_c_structs("src/blocks.c")
    with open("blocks.json", "w") as f:
        json.dump(blocks, f, indent=2)
    print("Blocks saved to blocks.json")