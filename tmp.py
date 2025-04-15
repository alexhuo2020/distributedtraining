import json
import numpy as np
# Load the .trace.json file
file_path = "/Users/huo/Downloads/b1fb3ef0e538_87.1734665549614871437.pt.trace.json"
file_path2 = "/Users/huo/Downloads/b1fb3ef0e538_88.1734665549662972122.pt.trace.json"
with open(file_path, "r") as f:
    trace_data = json.load(f)
with open(file_path2, "r") as f:
    trace_data2 = json.load(f)
# Extract events
events1 = trace_data["traceEvents"]
events2 = trace_data2["traceEvents"]
gpu_events = [e for e in events1 if "gpu" in e.get("cat", "").lower()]
sgemm_events = []
for event in events1:
    if 'name' in event and 'sgemm' in event['name'].lower():
        sgemm_events.append(event)
sgemm_events_times = [e["ts"] for e in sgemm_events if e["ph"] == "X"]
print(sgemm_events_times)

gpu_events = [e for e in events2 if "gpu" in e.get("cat", "").lower()]
sgemm_events = []
for event in events2:
    if 'name' in event and 'sgemm' in event['name'].lower():
        sgemm_events.append(event)
sgemm_events_times = [e["ts"] for e in sgemm_events if e["ph"] == "X"]
print(sgemm_events_times)

# for events in [events1, events2]:
#     # Filter forward and backward computation events
#     forward_events = [e for e in events if "forward" in e.get("name", "").lower()]
#     backward_events = [e for e in events if "backward" in e.get("name", "").lower()]
#     # print(forward_events)
#     forward_start_times = [e["ts"] for e in forward_events if e["ph"] == "X"]
#     backward_start_times = [e["ts"] for e in backward_events if e["ph"] == "X"]
#     # Normalize timestamps (optional)
#     min_time = min(forward_start_times + backward_start_times)
#     forward_start_times = [ts - min_time for ts in forward_start_times]
#     backward_start_times = [ts - min_time for ts in backward_start_times]

#     # Print results
#     # print("Forward start times (ms):", np.array(forward_start_times)*1e-3)
#     # print("Backward start times (ms):", np.array(backward_start_times)*1e-3)
#     # Filter events that occur on the GPU
#     gpu_forward_events = [e for e in forward_events if "gpu" in e.get("cat", "").lower()]
#     gpu_backward_events = [e for e in backward_events if "gpu" in e.get("cat", "").lower()]

#     # Extract GPU-specific start times
#     gpu_forward_start_times = [e["ts"] for e in gpu_forward_events if e["ph"] == "X"]
#     gpu_backward_start_times = [e["ts"] for e in gpu_backward_events if e["ph"] == "X"]
#     gpu_forward_start_times = [ts - min_time for ts in gpu_forward_start_times]
#     gpu_backward_start_times = [ts - min_time for ts in gpu_backward_start_times]

#     # print("GPU Forward start times (ms):", np.array(gpu_forward_start_times)*1e-3)
#     # print("GPU Backward start times (ms):", np.array(gpu_backward_start_times)*1e-3)
#     # Extract GPU-specific forward events with names and start times
#     gpu_forward_events_info = [(e["name"], e["ts"]) for e in gpu_forward_events if e["ph"] == "X"]

#     # Extract GPU-specific backward events with names and start times
#     gpu_backward_events_info = [(e["name"], e["ts"]) for e in gpu_backward_events if e["ph"] == "X"]

#     # Print forward events
#     print("GPU Forward Events:")
#     for name, start_time in gpu_forward_events_info:
#         print(f"Event: {name}, Start Time: {start_time} μs")

#     # Print backward events
#     print("\nGPU Backward Events:")
#     for name, start_time in gpu_backward_events_info:
#         print(f"Event: {name}, Start Time: {start_time} μs")
