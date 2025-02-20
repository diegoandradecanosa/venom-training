import os
import json

def extract_and_sort_profiler_steps(trace_dir):

    profiler_steps = []
    kernels={}
    k=0

    # Iterate over all JSON files in the specified directory
    for filename in os.listdir(trace_dir):
        k+=1
        print(f'{k} / {len(os.listdir(trace_dir))}')
        if filename.endswith(".json"):
            file_path = os.path.join(trace_dir, filename)

            try:
                # Open and parse the JSON file
                with open(file_path, "r") as trace_file:
                    trace_data = json.load(trace_file)

                # Look for events matching 'ProfilerStep#<Number>'
                for event in trace_data.get("traceEvents", []):
                    if "name" in event and event["name"].startswith("ProfilerStep#"):
                        step_number = int(event["name"].split("#")[-1])
                        timestamp = event.get("ts")
                        duration = event.get("dur")
                        profiler_steps.append((step_number, timestamp, duration))
                    if "name" in event and ["gemm" in event["name"] or "spatha" in event["name"]][0]:
                        kernel_name = event["name"]
                        timestamp = event.get("ts")
                        duration = event.get("dur")
                        if kernel_name not in kernels:
                            kernels[kernel_name]=[]
                        kernels[kernel_name].append((timestamp, duration))
            except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
                print(f"Error reading or parsing file {file_path}: {e}")



    return profiler_steps, kernels

# Example usage
if __name__ == "__main__":
    trace_directory = "<results_directory>" #IMPORTANT: INTRODUCE THE DIRECTORY OF THE TRACES GENERATED, NOT THE DIRECTORY OF A SINGLE TRACE
    steps, kernels = extract_and_sort_profiler_steps(trace_directory)

    output_data = {"Steps": steps, "Kernels": kernels }


    output_file = "<results_name>.json"  # Replace with the desired output file name
    try:
        with open(output_file, "w") as outfile:
            json.dump(output_data, outfile, indent=4, separators=(",", ": "))
        print(f"Profiler steps saved to {output_file}")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
