using System.Text.Json;
using System.Text.Json.Serialization;
using Neuraval.Abstractions;

namespace Neuraval.ChatBot.Persistence;

/// <summary>Stores local chat identity and conversation history for one named profile.</summary>
public sealed class ChatProfileStore
{
    private readonly string _directory;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter() }
    };

    public ChatProfileStore(string? directory = null)
    {
        _directory = directory ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Neuraval", "chat-profiles");
    }

    public ChatProfileMemory Load(string name)
    {
        string path = GetPath(name);
        if (!File.Exists(path))
            return new ChatProfileMemory(name);

        var profile = JsonSerializer.Deserialize<ChatProfileMemory>(File.ReadAllText(path), JsonOptions)
            ?? throw new InvalidDataException($"El perfil '{name}' está vacío o dañado.");
        profile.Name = name;
        profile.Messages ??= new List<StoredChatMessage>();
        return profile;
    }

    public void Save(ChatProfileMemory profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        Directory.CreateDirectory(_directory);
        string path = GetPath(profile.Name);
        string temporaryPath = path + ".tmp";
        File.WriteAllText(temporaryPath, JsonSerializer.Serialize(profile, JsonOptions));
        File.Move(temporaryPath, path, overwrite: true);
    }

    private string GetPath(string name)
    {
        if (string.IsNullOrWhiteSpace(name) || name is "." or ".."
            || name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0
            || name.Contains('/') || name.Contains('\\'))
            throw new ArgumentException("El nombre de perfil debe ser un nombre simple, sin rutas.", nameof(name));

        return Path.Combine(_directory, name + ".json");
    }
}

public sealed class ChatProfileMemory
{
    public string Name { get; set; }
    public string Identity { get; set; } = string.Empty;
    public List<StoredChatMessage> Messages { get; set; } = new();

    public ChatProfileMemory(string name) => Name = name;
}

public sealed class StoredChatMessage
{
    public ChatRole Role { get; set; }
    public string Content { get; set; } = string.Empty;

    public StoredChatMessage() { }
    public StoredChatMessage(ChatRole role, string content) { Role = role; Content = content; }
}
