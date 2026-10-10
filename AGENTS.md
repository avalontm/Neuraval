# Repository Guidelines

## Project Structure & Module Organization

`Neuraval.sln` is the entry point for the .NET solution. Core model and training code lives in `Neuraval/`; shared contracts are in `Neuraval.Abstractions/`; tensor operations and CPU/CUDA backends are in `Neuraval.Tensor/` and `Neuraval.Cuda/`. Chat integrations and the CLI are in `Neuraval.ChatBot/` and `Neuraval.CLI/`. Evolution code is split between the reusable library (`Neuraval.Evolution/`) and Mario/BizHawk bridge (`Neuraval.Evolution.MarioBridge/`). `Neuraval.Samples.DinoGame/` is the graphical sample, and `Neuraval.Tests/` contains automated tests. Supporting material and runtime data are under `docs/`, `Data/`, and `lua/`; checkpoints and savestates are runtime artifacts.

## Build, Test, and Development Commands

Run commands from the repository root:

- `dotnet restore Neuraval.sln` restores solution dependencies.
- `dotnet build Neuraval.sln` builds all managed projects.
- `dotnet test Neuraval.Tests/Neuraval.Tests.csproj` runs the xUnit suite.
- `dotnet run --project Neuraval.CLI/Neuraval.CLI.csproj` starts the console application.
- `dotnet run --project Neuraval.Samples.DinoGame/Neuraval.Samples.DinoGame.csproj` starts the Windows/MonoGame sample.

The optional native CUDA component in `Neuraval.Cuda.Native/` has its own `CMakeLists.txt` and requires a CUDA-capable toolchain.

## Coding Style & Naming Conventions

Follow the existing C# style: four-space indentation, nullable reference types, implicit usings, PascalCase for public types and methods, and descriptive names. Keep namespaces and project references aligned with the module boundaries above. Name tests after the subject and behavior, using the `*Tests.cs` pattern (for example, `ChatModelFactoryTests.cs`). No repository-wide formatter or linter configuration is present; match nearby code.

## Testing Guidelines

Add focused xUnit tests in `Neuraval.Tests/` for behavior changes and regression fixes. Use `dotnet test Neuraval.Tests/Neuraval.Tests.csproj`; run the full solution build when changing shared APIs or project references. Tests commonly use descriptive fact/theory names and local fakes for external dependencies.

## Commit & Pull Request Guidelines

Recent commits use concise subjects such as `fix test unitarios` and `tensor implement`. Use a short imperative summary (for example, `Fix tokenizer boundary handling`) and keep unrelated changes separate. Pull requests should explain the behavior and motivation, list relevant build/test results, link related issues when available, and include screenshots for visible UI changes. Call out required model files, CUDA tooling, or emulator setup when they affect review or reproduction.

## Configuration & Generated Data

Do not commit credentials, large model weights, or machine-specific ROM/emulator savestates. Keep local training settings and generated checkpoints intentional, and document any new external runtime prerequisites.
