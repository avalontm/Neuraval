using System;
using System.Collections.Generic;
using System.IO;
using Neuraval.Core.Serialization.SafeTensors;
using Xunit;

namespace Neuraval.Tests
{
    public class SafeTensorsWriterReaderTests
    {
        private static string TempFilePath()
        {
            return Path.Combine(Path.GetTempPath(), $"safetensors_test_{Guid.NewGuid():N}.safetensors");
        }

        [Fact]
        public void WriteRead_SingleF32Tensor_RoundTripsExactly()
        {
            var path = TempFilePath();
            try
            {
                var original = new SafeTensorsEntry("weight", SafeTensorsDType.F32, new[] { 2, 2 }, new float[] { 1f, -2.5f, 3.25f, 100f });

                SafeTensorsWriter.Write(path, new[] { original });
                var file = SafeTensorsReader.Read(path);

                var loaded = file.Find("weight");
                Assert.NotNull(loaded);
                Assert.Equal(SafeTensorsDType.F32, loaded!.DType);
                Assert.Equal(original.Shape, loaded.Shape);
                Assert.Equal(original.Data, loaded.Data);
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Fact]
        public void WriteRead_MultipleTensors_PreservesAllOfThem()
        {
            var path = TempFilePath();
            try
            {
                var entries = new[]
                {
                    new SafeTensorsEntry("embed.weight", SafeTensorsDType.F32, new[] { 3 }, new float[] { 1f, 2f, 3f }),
                    new SafeTensorsEntry("layers.0.norm.weight", SafeTensorsDType.F32, new[] { 2, 2 }, new float[] { 0.5f, 0.5f, 0.5f, 0.5f })
                };

                SafeTensorsWriter.Write(path, entries);
                var file = SafeTensorsReader.Read(path);

                Assert.Equal(2, file.Tensors.Count);
                Assert.Equal(new float[] { 1f, 2f, 3f }, file.Find("embed.weight")!.Data);
                Assert.Equal(new float[] { 0.5f, 0.5f, 0.5f, 0.5f }, file.Find("layers.0.norm.weight")!.Data);
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Fact]
        public void WriteRead_F16Tensor_RoundTripsWithinPrecisionTolerance()
        {
            var path = TempFilePath();
            try
            {
                var original = new SafeTensorsEntry("weight", SafeTensorsDType.F16, new[] { 3 }, new float[] { 1.5f, -2.25f, 100f });

                SafeTensorsWriter.Write(path, new[] { original });
                var loaded = SafeTensorsReader.Read(path).Find("weight")!;

                for (int i = 0; i < original.Data.Length; i++)
                {
                    Assert.True(Math.Abs(original.Data[i] - loaded.Data[i]) < 0.01f);
                }
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Fact]
        public void WriteRead_BF16Tensor_RoundTripsWithinPrecisionTolerance()
        {
            var path = TempFilePath();
            try
            {
                var original = new SafeTensorsEntry("weight", SafeTensorsDType.BF16, new[] { 3 }, new float[] { 1.5f, -2.25f, 1000f });

                SafeTensorsWriter.Write(path, new[] { original });
                var loaded = SafeTensorsReader.Read(path).Find("weight")!;

                for (int i = 0; i < original.Data.Length; i++)
                {
                    Assert.True(Math.Abs(original.Data[i] - loaded.Data[i]) < Math.Abs(original.Data[i]) * 0.02f + 0.05f);
                }
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Fact]
        public void WriteRead_WithMetadata_RoundTripsMetadata()
        {
            var path = TempFilePath();
            try
            {
                var entries = new[] { new SafeTensorsEntry("weight", SafeTensorsDType.F32, new[] { 1 }, new float[] { 1f }) };
                var metadata = new Dictionary<string, string> { ["format"] = "navm-modern" };

                SafeTensorsWriter.Write(path, entries, metadata);
                var file = SafeTensorsReader.Read(path);

                Assert.Equal("navm-modern", file.Metadata["format"]);
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Fact]
        public void Find_UnknownName_ReturnsNull()
        {
            var path = TempFilePath();
            try
            {
                var entries = new[] { new SafeTensorsEntry("weight", SafeTensorsDType.F32, new[] { 1 }, new float[] { 1f }) };
                SafeTensorsWriter.Write(path, entries);

                var file = SafeTensorsReader.Read(path);
                Assert.Null(file.Find("does.not.exist"));
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Fact]
        public void Write_EmptyEntries_Throws()
        {
            var path = TempFilePath();
            Assert.Throws<ArgumentException>(() => SafeTensorsWriter.Write(path, Array.Empty<SafeTensorsEntry>()));
        }

        [Fact]
        public void Write_DuplicateNames_Throws()
        {
            var path = TempFilePath();
            var entries = new[]
            {
                new SafeTensorsEntry("weight", SafeTensorsDType.F32, new[] { 1 }, new float[] { 1f }),
                new SafeTensorsEntry("weight", SafeTensorsDType.F32, new[] { 1 }, new float[] { 2f })
            };

            Assert.Throws<ArgumentException>(() => SafeTensorsWriter.Write(path, entries));
        }

        [Fact]
        public void Write_NullEntries_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => SafeTensorsWriter.Write(TempFilePath(), null!));
        }

        [Fact]
        public void Write_CreatesMissingDirectory()
        {
            var directory = Path.Combine(Path.GetTempPath(), $"safetensors_dir_{Guid.NewGuid():N}");
            var path = Path.Combine(directory, "model.safetensors");

            try
            {
                var entries = new[] { new SafeTensorsEntry("weight", SafeTensorsDType.F32, new[] { 1 }, new float[] { 1f }) };
                SafeTensorsWriter.Write(path, entries);

                Assert.True(File.Exists(path));
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
            }
        }
    }
}
