using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CSharpOnnxBenchmark
{
    public class Program
    {
        public static int Main(string[] args)
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine("HDML C# ONNX RUNTIME INFERENCE VERIFICATION & BENCHMARK SUITE");
            Console.WriteLine("================================================================================");

            string modelPath = "deployment/unitree_a1_action_only.onnx";
            int contextLength = 20;
            int propDim = -1;
            int actionDim = -1;
            int iterations = 100;
            string inputJsonPath = "";
            string outputJsonPath = "";

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--model" && i + 1 < args.Length) modelPath = args[++i];
                else if (args[i] == "--context-length" && i + 1 < args.Length) contextLength = int.Parse(args[++i]);
                else if (args[i] == "--prop-dim" && i + 1 < args.Length) propDim = int.Parse(args[++i]);
                else if (args[i] == "--action-dim" && i + 1 < args.Length) actionDim = int.Parse(args[++i]);
                else if (args[i] == "--iters" && i + 1 < args.Length) iterations = int.Parse(args[++i]);
                else if ((args[i] == "--input" || args[i] == "--input-json") && i + 1 < args.Length) inputJsonPath = args[++i];
                else if (args[i] == "--dump-output" && i + 1 < args.Length) outputJsonPath = args[++i];
            }

            if (!File.Exists(modelPath))
            {
                Console.Error.WriteLine($"[ERROR] Model file not found: {modelPath}");
                return 1;
            }

            Console.WriteLine($"Loading ONNX Model : {modelPath}");
            var sessionOptions = new SessionOptions();
            sessionOptions.IntraOpNumThreads = Environment.ProcessorCount >= 4 ? 4 : 2;
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            using var session = new InferenceSession(modelPath, sessionOptions);

            Console.WriteLine("Inspecting Session Metadata:");
            Console.WriteLine($"  Input Count  : {session.InputMetadata.Count}");
            foreach (var kv in session.InputMetadata)
            {
                Console.WriteLine($"    Input  [{kv.Key}]: Shape=[{string.Join(", ", kv.Value.Dimensions)}] Type={kv.Value.ElementType}");
            }
            Console.WriteLine($"  Output Count : {session.OutputMetadata.Count}");
            foreach (var kv in session.OutputMetadata)
            {
                Console.WriteLine($"    Output [{kv.Key}]: Shape=[{string.Join(", ", kv.Value.Dimensions)}] Type={kv.Value.ElementType}");
            }

            // Auto-detect dimensions from metadata if not explicitly provided
            if (propDim <= 0 && session.InputMetadata.ContainsKey("states"))
            {
                var dims = session.InputMetadata["states"].Dimensions;
                if (dims.Length >= 3 && dims[2] > 0) propDim = dims[2];
            }
            if (propDim <= 0) propDim = 35; // Default fallback for Unitree A1

            if (actionDim <= 0 && session.InputMetadata.ContainsKey("actions"))
            {
                var dims = session.InputMetadata["actions"].Dimensions;
                if (dims.Length >= 3 && dims[2] > 0) actionDim = dims[2];
            }
            if (actionDim <= 0 && session.OutputMetadata.ContainsKey("action"))
            {
                var dims = session.OutputMetadata["action"].Dimensions;
                if (dims.Length >= 2 && dims[1] > 0) actionDim = dims[1];
            }
            if (actionDim <= 0) actionDim = 12; // Default fallback

            Console.WriteLine($"Configuration:");
            Console.WriteLine($"  Context Length : {contextLength}");
            Console.WriteLine($"  Prop Dim       : {propDim}");
            Console.WriteLine($"  Action Dim     : {actionDim}");
            Console.WriteLine($"  Iterations     : {iterations}");

            // Prepare Input Tensors
            DenseTensor<float> statesTensor;
            DenseTensor<float> rtgsTensor;
            DenseTensor<float> actionsTensor;
            DenseTensor<long> timestepsTensor;

            if (!string.IsNullOrEmpty(inputJsonPath) && File.Exists(inputJsonPath))
            {
                Console.WriteLine($"Reading verification input from: {inputJsonPath}");
                var jsonDoc = JsonDocument.Parse(File.ReadAllText(inputJsonPath));
                var root = jsonDoc.RootElement;

                var statesFlat = root.GetProperty("states").EnumerateArray().Select(x => x.GetSingle()).ToArray();
                var rtgsFlat = root.GetProperty("rtgs").EnumerateArray().Select(x => x.GetSingle()).ToArray();
                var actionsFlat = root.GetProperty("actions").EnumerateArray().Select(x => x.GetSingle()).ToArray();
                var timestepsFlat = root.GetProperty("timesteps").EnumerateArray().Select(x => x.GetInt64()).ToArray();

                statesTensor = new DenseTensor<float>(statesFlat, new[] { 1, contextLength, propDim });
                rtgsTensor = new DenseTensor<float>(rtgsFlat, new[] { 1, contextLength, 1 });
                actionsTensor = new DenseTensor<float>(actionsFlat, new[] { 1, contextLength, actionDim });
                timestepsTensor = new DenseTensor<long>(timestepsFlat, new[] { 1, contextLength });
            }
            else
            {
                var rng = new Random(42);
                statesTensor = new DenseTensor<float>(new[] { 1, contextLength, propDim });
                for (int t = 0; t < contextLength; t++)
                {
                    for (int d = 0; d < propDim; d++)
                    {
                        statesTensor[0, t, d] = (float)(rng.NextDouble() * 2.0 - 1.0);
                    }
                }

                rtgsTensor = new DenseTensor<float>(new[] { 1, contextLength, 1 });
                for (int t = 0; t < contextLength; t++)
                {
                    rtgsTensor[0, t, 0] = 0.0f;
                }

                actionsTensor = new DenseTensor<float>(new[] { 1, contextLength, actionDim });
                for (int t = 0; t < contextLength; t++)
                {
                    for (int d = 0; d < actionDim; d++)
                    {
                        actionsTensor[0, t, d] = (float)(rng.NextDouble() * 2.0 - 1.0);
                    }
                }

                timestepsTensor = new DenseTensor<long>(new[] { 1, contextLength });
                for (int t = 0; t < contextLength; t++)
                {
                    timestepsTensor[0, t] = t;
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("states", statesTensor),
                NamedOnnxValue.CreateFromTensor("rtgs", rtgsTensor),
                NamedOnnxValue.CreateFromTensor("actions", actionsTensor),
                NamedOnnxValue.CreateFromTensor("timesteps", timestepsTensor)
            };

            // Warmup iterations
            Console.WriteLine("Executing warmup iterations (10 runs)...");
            for (int w = 0; w < 10; w++)
            {
                using var outWarm = session.Run(inputs);
            }

            // Benchmark loop
            Console.WriteLine($"Benchmarking {iterations} iterations on CPU...");
            var latencies = new List<double>(iterations);
            float[] finalAction = null;

            for (int i = 0; i < iterations; i++)
            {
                var sw = Stopwatch.StartNew();
                using var results = session.Run(inputs);
                sw.Stop();
                latencies.Add(sw.Elapsed.TotalMilliseconds);

                if (i == 0)
                {
                    var actionTensor = results.First(r => r.Name == "action").AsTensor<float>();
                    finalAction = new float[actionDim];
                    for (int d = 0; d < actionDim; d++)
                    {
                        finalAction[d] = actionTensor[0, d];
                    }
                }
            }

            latencies.Sort();
            double avgMs = latencies.Average();
            double minMs = latencies.First();
            double maxMs = latencies.Last();
            double p50Ms = latencies[(int)(latencies.Count * 0.50)];
            double p95Ms = latencies[(int)(latencies.Count * 0.95)];
            double hz = 1000.0 / avgMs;

            // Resource consumption telemetry
            long memoryAllocatedBytes = GC.GetTotalMemory(true);
            double memoryAllocatedMb = memoryAllocatedBytes / (1024.0 * 1024.0);
            long processMemoryBytes = Process.GetCurrentProcess().WorkingSet64;
            double processMemoryMb = processMemoryBytes / (1024.0 * 1024.0);

            Console.WriteLine("\n--------------------------------------------------------------------------------");
            Console.WriteLine("BENCHMARK & RESOURCE CONSUMPTION (C# / .NET 8 / Microsoft.ML.OnnxRuntime):");
            Console.WriteLine("--------------------------------------------------------------------------------");
            Console.WriteLine($"  Average Latency    : {avgMs:F2} ms");
            Console.WriteLine($"  Median (P50)       : {p50Ms:F2} ms");
            Console.WriteLine($"  95th Percentile    : {p95Ms:F2} ms");
            Console.WriteLine($"  Min Latency        : {minMs:F2} ms");
            Console.WriteLine($"  Max Latency        : {maxMs:F2} ms");
            Console.WriteLine($"  Inference Rate     : {hz:F0} Hz");
            Console.WriteLine($"  GC Managed Heap    : {memoryAllocatedMb:F2} MB");
            Console.WriteLine($"  Working Set (RAM)  : {processMemoryMb:F2} MB");
            Console.WriteLine("--------------------------------------------------------------------------------");

            // Output action validation
            Console.WriteLine("\nAction Vector Sample:");
            bool hasNaN = false;
            for (int d = 0; d < finalAction.Length; d++)
            {
                if (float.IsNaN(finalAction[d]) || float.IsInfinity(finalAction[d]))
                {
                    hasNaN = true;
                }
                Console.Write($"{finalAction[d]:+0.0000;-0.0000} ");
                if ((d + 1) % 6 == 0) Console.WriteLine();
            }
            Console.WriteLine();

            if (hasNaN)
            {
                Console.Error.WriteLine("[FAIL] Output action tensor contains NaN or Inf!");
                return 1;
            }

            // Dump output JSON if requested
            if (!string.IsNullOrEmpty(outputJsonPath))
            {
                var outObj = new
                {
                    model_path = modelPath,
                    latency_ms = avgMs,
                    hz = hz,
                    action = finalAction
                };
                File.WriteAllText(outputJsonPath, JsonSerializer.Serialize(outObj, new JsonSerializerOptions { WriteIndented = true }));
                Console.WriteLine($"Saved output verification JSON to: {outputJsonPath}");
            }

            Console.WriteLine("[PASS] C# ONNX Runtime Execution Completed Successfully!");
            Console.WriteLine("================================================================================\n");
            return 0;
        }
    }
}
