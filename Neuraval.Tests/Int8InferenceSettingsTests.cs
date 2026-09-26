using Neuraval.Core.Quantization;
using Xunit;

namespace Neuraval.Tests
{
    public class Int8InferenceSettingsTests
    {
        [Fact]
        public void Default_IsDisabled()
        {
            Int8InferenceSettings.Disable();

            Assert.False(Int8InferenceSettings.EnableInt8Cpu);
        }

        [Fact]
        public void Enable_SetsFlagTrue()
        {
            Int8InferenceSettings.Enable();

            Assert.True(Int8InferenceSettings.EnableInt8Cpu);

            Int8InferenceSettings.Disable();
        }

        [Fact]
        public void Disable_SetsFlagFalse()
        {
            Int8InferenceSettings.Enable();
            Int8InferenceSettings.Disable();

            Assert.False(Int8InferenceSettings.EnableInt8Cpu);
        }
    }
}
