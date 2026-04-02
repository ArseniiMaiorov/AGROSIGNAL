#!/usr/bin/env python3
"""
Cleans up FieldActionsPanel.vue after composable extraction.
Removes duplicate script block between </script> and <style scoped>.
Run from any directory: python3 clean_panel.py
"""
import os

path = os.path.expanduser(
    "~/Documents/SUAI/Диплом/AGROSIGNAL/frontend/src/components/FieldActionsPanel.vue"
)

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Split on </script> — take the part before the first one (template + script)
# then append the <style scoped> block starting from the LAST <style scoped>
script_end_marker = "</script>"
style_start_marker = "<style scoped>"

first_script_end = content.find(script_end_marker)
if first_script_end == -1:
    print("ERROR: </script> not found")
    raise SystemExit(1)

last_style_start = content.rfind(style_start_marker)
if last_style_start == -1:
    print("ERROR: <style scoped> not found")
    raise SystemExit(1)

before = content[: first_script_end + len(script_end_marker)]
after = content[last_style_start:]

cleaned = before + "\n\n" + after

with open(path, "w", encoding="utf-8") as f:
    f.write(cleaned)

lines_before = content.count("\n")
lines_after = cleaned.count("\n")
print(f"Done. Lines: {lines_before} → {lines_after} (removed {lines_before - lines_after})")
