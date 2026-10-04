using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Runtime.InteropServices;

namespace Neuraval.CLI
{
    public class SystemMonitor : IDisposable
    {
        private readonly Timer _timer;
        private readonly Process _process;
        private readonly DateTime _startTime;
        private readonly Stopwatch _stopwatch;
        private bool _disposed;

        private long _peakMemoryBytes;
        private TimeSpan _lastTotalProcessorTime;
        private DateTime _lastTime;
        private double _avgCpuUsage;
        private int _cpuSamples;
        private bool _gpuQueryFailed;

        public SystemMonitor(int updateIntervalSeconds = 5)
        {
            _process = Process.GetCurrentProcess();
            _startTime = DateTime.Now;
            _stopwatch = Stopwatch.StartNew();
            _peakMemoryBytes = 0;
            _avgCpuUsage = 0;
            _cpuSamples = 0;

            _lastTotalProcessorTime = _process.TotalProcessorTime;
            _lastTime = DateTime.Now;

            PrintSystemInfo();

            _timer = new Timer(UpdateStats, null, updateIntervalSeconds * 1000, updateIntervalSeconds * 1000);
        }

        private (int gpuUtilPercent, long memUsedMB, long memTotalMB)? QueryGpuStats()
        {
            if (_gpuQueryFailed)
            {
                return null;
            }

            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "nvidia-smi",
                    Arguments = "--query-gpu=utilization.gpu,memory.used,memory.total --format=csv",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var proc = Process.Start(psi);
                if (proc == null)
                {
                    _gpuQueryFailed = true;
                    return null;
                }

                string output = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit(2000);

                var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                if (lines.Length < 2)
                {
                    _gpuQueryFailed = true;
                    return null;
                }

                var parts = lines[1].Split(',');
                if (parts.Length < 3)
                {
                    _gpuQueryFailed = true;
                    return null;
                }

                int util = ParseFirstNumber(parts[0]);
                long memUsed = ParseFirstNumber(parts[1]);
                long memTotal = ParseFirstNumber(parts[2]);

                return (util, memUsed, memTotal);
            }
            catch
            {
                _gpuQueryFailed = true;
                return null;
            }
        }

        private static int ParseFirstNumber(string text)
        {
            var digits = new string(text.Trim().TakeWhile(c => char.IsDigit(c)).ToArray());
            return digits.Length > 0 ? int.Parse(digits) : 0;
        }

        private void PrintSystemInfo()
        {
            Console.WriteLine("===========================================");
            Console.WriteLine("          SYSTEM CONFIGURATION");
            Console.WriteLine("===========================================");
            Console.WriteLine($"  CPU Cores: {Environment.ProcessorCount}");
            Console.WriteLine($"  Threads Configured: {Neuraval.Core.Utils.Matematicas.GetNumThreads()}");
            Console.WriteLine($"  OS: {RuntimeInformation.OSDescription}");
            Console.WriteLine($"  Architecture: {RuntimeInformation.ProcessArchitecture}");
            Console.WriteLine($"  .NET Version: {Environment.Version}");
            bool gpuActive = Neuraval.Core.Utils.Matematicas.GpuEnabled;
            Console.WriteLine(gpuActive
                ? "  GPU Support: YES (CUDA activa)"
                : "  GPU Support: NO (CPU-only mode)");

            try
            {
                var gcInfo = GC.GetGCMemoryInfo();
                var totalMemoryMB = gcInfo.TotalAvailableMemoryBytes / 1024 / 1024;
                Console.WriteLine($"  Available Memory: ~{totalMemoryMB:N0} MB");
            }
            catch
            {
                Console.WriteLine($"  Available Memory: Unknown");
            }

            Console.WriteLine("===========================================");
            Console.WriteLine();
        }

        private void UpdateStats(object? state)
        {
            try
            {
                _process.Refresh();

                long currentMemoryBytes = _process.WorkingSet64;
                if (currentMemoryBytes > _peakMemoryBytes)
                {
                    _peakMemoryBytes = currentMemoryBytes;
                }

                double memoryMB = currentMemoryBytes / 1024.0 / 1024.0;

                var currentTime = DateTime.Now;
                var currentTotalProcessorTime = _process.TotalProcessorTime;

                double cpuUsage = 0;
                var timeDiff = (currentTime - _lastTime).TotalMilliseconds;

                if (timeDiff > 0)
                {
                    var processorTimeDiff = (currentTotalProcessorTime - _lastTotalProcessorTime).TotalMilliseconds;
                    cpuUsage = (processorTimeDiff / (timeDiff * Environment.ProcessorCount)) * 100;

                    cpuUsage = Math.Max(0, Math.Min(100, cpuUsage));

                    _avgCpuUsage = (_avgCpuUsage * _cpuSamples + cpuUsage) / (_cpuSamples + 1);
                    _cpuSamples++;
                }

                _lastTotalProcessorTime = currentTotalProcessorTime;
                _lastTime = currentTime;

                int threadCount = _process.Threads.Count;

                var elapsed = _stopwatch.Elapsed;

                string gpuInfo = "";
                if (Neuraval.Core.Utils.Matematicas.GpuEnabled)
                {
                    var gpuStats = QueryGpuStats();
                    if (gpuStats.HasValue)
                    {
                        var (util, memUsed, memTotal) = gpuStats.Value;
                        gpuInfo = $" | GPU: {util}% | VRAM: {memUsed}/{memTotal} MB";
                    }
                    else
                    {
                        gpuInfo = " | GPU: N/D";
                    }
                }

                Console.WriteLine($"[{elapsed:hh\\:mm\\:ss}] CPU: {cpuUsage:F1}% | RAM: {memoryMB:F0} MB | Threads: {threadCount}{gpuInfo}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Monitor error: {ex.Message}");
            }
        }

        public void PrintFinalStats()
        {
            _stopwatch.Stop();
            _process.Refresh();

            Console.WriteLine();
            Console.WriteLine("===========================================");
            Console.WriteLine("           TRAINING STATISTICS");
            Console.WriteLine("===========================================");
            Console.WriteLine($"  Total Time: {_stopwatch.Elapsed:hh\\:mm\\:ss}");
            Console.WriteLine($"  Peak Memory: {_peakMemoryBytes / 1024.0 / 1024.0:F0} MB");

            if (_cpuSamples > 0)
            {
                Console.WriteLine($"  Avg CPU Usage: {_avgCpuUsage:F1}%");
            }

            Console.WriteLine($"  Final Thread Count: {_process.Threads.Count}");

            Console.WriteLine($"  GC Collections:");
            Console.WriteLine($"    Gen 0: {GC.CollectionCount(0)}");
            Console.WriteLine($"    Gen 1: {GC.CollectionCount(1)}");
            Console.WriteLine($"    Gen 2: {GC.CollectionCount(2)}");

            var totalCpuTime = _process.TotalProcessorTime;
            Console.WriteLine($"  Total CPU Time: {totalCpuTime:hh\\:mm\\:ss}");

            Console.WriteLine("===========================================");
            Console.WriteLine();
        }

        public static void PrintCurrentStats()
        {
            var process = Process.GetCurrentProcess();
            process.Refresh();

            var memoryMB = process.WorkingSet64 / 1024.0 / 1024.0;
            var threads = process.Threads.Count;

            Console.WriteLine();
            Console.WriteLine("===========================================");
            Console.WriteLine("         CURRENT SYSTEM STATUS");
            Console.WriteLine("===========================================");
            Console.WriteLine($"  Current Memory: {memoryMB:F0} MB");
            Console.WriteLine($"  Active Threads: {threads}");
            Console.WriteLine($"  CPU Cores: {Environment.ProcessorCount}");
            Console.WriteLine($"  Configured Threads: {Neuraval.Core.Utils.Matematicas.GetNumThreads()}");
            Console.WriteLine($"  Total CPU Time: {process.TotalProcessorTime:hh\\:mm\\:ss}");

            var gcMemory = GC.GetTotalMemory(false) / 1024.0 / 1024.0;
            Console.WriteLine($"  Managed Memory: {gcMemory:F0} MB");
            Console.WriteLine($"  GC Gen 0: {GC.CollectionCount(0)}");
            Console.WriteLine($"  GC Gen 1: {GC.CollectionCount(1)}");
            Console.WriteLine($"  GC Gen 2: {GC.CollectionCount(2)}");

            Console.WriteLine("===========================================");
            Console.WriteLine();
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            _timer?.Dispose();
            _stopwatch?.Stop();

            _disposed = true;
        }
    }
}
