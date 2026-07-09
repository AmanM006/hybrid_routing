import sys, re, glob, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Use the specific v10 pipeline run log
log_path = r'C:\Users\cheer\.gemini\antigravity\brain\2853899a-c256-4afd-94a7-ad38c004bada\.system_generated\tasks\task-1871.log'
print('Using log:', os.path.basename(log_path))

with open(log_path, encoding='utf-8', errors='replace') as f:
    log = f.read()

lines = log.splitlines()
task_logs = sorted([l for l in lines if 'TASK_LOG' in l and 'new' in l])
print('=== ALL TASK_LOG LINES ===')
for t in task_logs:
    print(t)

all_lats = [float(x) for x in re.findall(r'latency=([\d.]+)s', log)]
print()
print('Max latency:', max(all_lats) if all_lats else 'N/A', 's')
print('Avg latency:', round(sum(all_lats)/len(all_lats),1) if all_lats else 'N/A', 's')
fails = sum(1 for t in task_logs if 'validation=FAIL' in t)
print('FAIL:', fails, '/ PASS:', len(task_logs)-fails, '/', len(task_logs))

print()
print('=== HEALTHCHECK EVENTS ===')
for l in lines:
    if any(x in l for x in ['Healthcheck for', 'Pruned', 'UPDATED ROLE', 'SOFT_FAIL', 'DEAD (404']):
        print(l)
print()
print('=== WALL TIME ===')
for l in lines:
    if 'TOTAL WALL TIME' in l or 'Finished. Results' in l:
        print(l)
