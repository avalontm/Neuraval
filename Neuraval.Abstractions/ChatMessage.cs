namespace Neuraval.Abstractions
{
    public class ChatMessage
    {
        public ChatRole Role { get; }
        public string Content { get; }

        public ChatMessage(ChatRole role, string content)
        {
            Role = role;
            Content = content;
        }
    }
}
