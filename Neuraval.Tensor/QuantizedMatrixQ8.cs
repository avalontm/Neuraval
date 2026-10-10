using System.Threading.Tasks;

namespace Neuraval.Tensor;

/// <summary>Block-quantized matrix used by low-memory CPU inference.</summary>
public sealed class QuantizedMatrixQ8
{
    private const int BlockSize = 32;
    private readonly sbyte[] _values;
    private readonly Half[] _scales;

    public int InputSize { get; }
    public int OutputSize { get; }

    private QuantizedMatrixQ8(int inputSize, int outputSize, sbyte[] values, Half[] scales)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        _values = values;
        _scales = scales;
    }

    /// <summary>Creates a matrix from the model's [input, output] layout.</summary>
    public static QuantizedMatrixQ8 FromInputOutput(float[,] matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        int input = matrix.GetLength(0), output = matrix.GetLength(1);
        return Create(input, output, (o, i) => matrix[i, o]);
    }

    /// <summary>Creates an output projection/embedding matrix in [output, input] layout.</summary>
    public static QuantizedMatrixQ8 FromOutputInput(float[,] matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        int output = matrix.GetLength(0), input = matrix.GetLength(1);
        return Create(input, output, (o, i) => matrix[o, i]);
    }

    /// <summary>Creates output-major weights directly from a flattened model-state array.</summary>
    public static QuantizedMatrixQ8 FromOutputMajor(float[] values, int outputSize, int inputSize)
    {
        if (outputSize <= 0 || inputSize <= 0 || values.Length != checked(outputSize * inputSize))
            throw new ArgumentException("Dimensiones Q8 incompatibles con el buffer de pesos.", nameof(values));
        return Create(inputSize, outputSize, (o, i) => values[o * inputSize + i]);
    }

    /// <summary>Creates [input, output] weights directly from a row-major flattened model-state array.</summary>
    public static QuantizedMatrixQ8 FromInputOutputMajor(float[] values, int inputSize, int outputSize)
    {
        if (inputSize <= 0 || outputSize <= 0 || values.Length != checked(inputSize * outputSize))
            throw new ArgumentException("Dimensiones Q8 incompatibles con el buffer de pesos.", nameof(values));
        return Create(inputSize, outputSize, (o, i) => values[i * outputSize + o]);
    }

    private static QuantizedMatrixQ8 Create(int inputSize, int outputSize, Func<int, int, float> getValue)
    {
        int blocksPerOutput = checked((inputSize + BlockSize - 1) / BlockSize);
        var values = new sbyte[checked(inputSize * outputSize)];
        var scales = new Half[checked(blocksPerOutput * outputSize)];

        for (int output = 0; output < outputSize; output++)
        {
            int outputOffset = output * inputSize;
            int scaleOffset = output * blocksPerOutput;
            for (int block = 0; block < blocksPerOutput; block++)
            {
                int first = block * BlockSize;
                int count = Math.Min(BlockSize, inputSize - first);
                float maxAbs = 0f;
                for (int i = 0; i < count; i++)
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(getValue(output, first + i)));

                float scale = maxAbs / 127f;
                scales[scaleOffset + block] = (Half)scale;
                if (scale == 0f)
                    continue;

                for (int i = 0; i < count; i++)
                {
                    int quantized = (int)MathF.Round(getValue(output, first + i) / scale);
                    values[outputOffset + first + i] = (sbyte)Math.Clamp(quantized, -127, 127);
                }
            }
        }

        return new QuantizedMatrixQ8(inputSize, outputSize, values, scales);
    }

    public Tensor Multiply(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Rank != 2 || input.Shape[1] != InputSize)
            throw new ArgumentException($"Q8 espera una matriz [filas,{InputSize}].", nameof(input));

        int rows = input.Shape[0];
        var output = new Tensor(new[] { rows, OutputSize }, input.Device, input.DType);
        int blocksPerOutput = (InputSize + BlockSize - 1) / BlockSize;

        void ComputeRow(int row)
        {
            int inputOffset = row * InputSize;
            int outputOffset = row * OutputSize;
            for (int column = 0; column < OutputSize; column++)
            {
                int weightOffset = column * InputSize;
                int scaleOffset = column * blocksPerOutput;
                float sum = 0f;
                for (int block = 0; block < blocksPerOutput; block++)
                {
                    int first = block * BlockSize;
                    int last = Math.Min(first + BlockSize, InputSize);
                    float blockSum = 0f;
                    for (int i = first; i < last; i++)
                        blockSum += input.Buffer[inputOffset + i] * _values[weightOffset + i];
                    sum += blockSum * (float)_scales[scaleOffset + block];
                }
                output.Buffer[outputOffset + column] = sum;
            }
        }

        if (rows <= 2)
            for (int row = 0; row < rows; row++) ComputeRow(row);
        else
            Parallel.For(0, rows, ComputeRow);

        return output;
    }

    public float[] GetOutputRow(int row)
    {
        if ((uint)row >= (uint)OutputSize)
            throw new ArgumentOutOfRangeException(nameof(row));
        int blocksPerOutput = (InputSize + BlockSize - 1) / BlockSize;
        int valuesOffset = row * InputSize;
        int scaleOffset = row * blocksPerOutput;
        var result = new float[InputSize];
        for (int i = 0; i < InputSize; i++)
            result[i] = _values[valuesOffset + i] * (float)_scales[scaleOffset + i / BlockSize];
        return result;
    }
}
