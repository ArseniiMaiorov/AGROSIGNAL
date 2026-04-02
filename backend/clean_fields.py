import os

filepath = '/home/arsenii-maiorov/Documents/SUAI/Диплом/AGROSIGNAL/backend/api/fields.py'
with open(filepath, 'r') as f:
    content = f.read()

startMarker1 = '        candidates = int(runtime.get("active_learning_candidates") or 0)'
endMarker1 = '@router.get("", response_model=FieldsListResponse)'

startIdx1 = content.find(startMarker1)
endIdx1 = content.find(endMarker1)

if startIdx1 != -1 and endIdx1 != -1 and endIdx1 > startIdx1:
    content = content[:startIdx1] + "\n\n" + content[endIdx1:]
    print("Removed block 1 successfully.")
else:
    print("Block 1 markers not found:", startIdx1, endIdx1)

startMarker2 = '@router.get("/status/{aoi_run_id}", response_model=RunStatus)'
startIdx2 = content.find(startMarker2)

if startIdx2 != -1:
    content = content[:startIdx2]
    print("Removed block 2 successfully.")
else:
    print("Block 2 marker not found:", startIdx2)

with open(filepath, 'w') as f:
    f.write(content)
print("File saved.")
