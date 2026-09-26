using System;
using Neuraval.Core.Serialization.SafeTensors;
using Xunit;

namespace Neuraval.Tests
{
    public class SafeTensorsEntryTests
    {
        [Fact]
        public void Constructor_ValidArguments_ExposesThem()
        {
            var entry = new SafeTensorsEntry("weight", SafeTensorsDType.F32, new[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });

            Assert.Equal("weight", entry.Name);
            Assert.Equal(SafeTensorsDType.F32, entry.DType);
            Assert.Equal(new[] { 2, 3 }, entry.Shape);
            Assert.Equal(6, entry.Data.Length);
        }

        [Fact]
        public void Constructor_EmptyName_Throws()
        {
            Assert.Throws<ArgumentException>(() =>
                new SafeTensorsEntry(string.Empty, SafeTensorsDType.F32, new[] { 1 }, new float[] { 1f }));
        }

        [Fact]
        public void Constructor_NonPositiveDimension_Throws()
        {
            Assert.Throws<ArgumentException>(() =>
                new SafeTensorsEntry("weight", SafeTensorsDType.F32, new[] { 0, 3 }, new float[] { 1, 2, 3 }));
        }

        [Fact]
        public void Constructor_ShapeDataMismatch_Throws()
        {
            Assert.Throws<ArgumentException>(() =>
                new SafeTensorsEntry("weight", SafeTensorsDType.F32, new[] { 2, 3 }, new float[] { 1, 2, 3 }));
        }

        [Fact]
        public void Constructor_NullShape_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new SafeTensorsEntry("weight", SafeTensorsDType.F32, null!, new float[] { 1f }));
        }

        [Fact]
        public void Constructor_NullData_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new SafeTensorsEntry("weight", SafeTensorsDType.F32, new[] { 1 }, null!));
        }
    }
}
