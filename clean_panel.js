const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'frontend/src/components/FieldActionsPanel.vue');
const content = fs.readFileSync(filePath, 'utf-8');

const scriptEnd = content.indexOf('</script>');
const lastStyleStart = content.lastIndexOf('<style scoped>');

if (scriptEnd !== -1 && lastStyleStart !== -1) {
    const before = content.slice(0, scriptEnd + 9);
    const after = content.slice(lastStyleStart);
    fs.writeFileSync(filePath, before + '\n\n' + after);
    console.log('Successfully cleaned up FieldActionsPanel.vue');
} else {
    console.log('Error: Could not find markers');
}
