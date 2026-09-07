using System;

namespace Neuraval.Abstractions
{
    public class ChatModelException : Exception
    {
        public ChatModelException(string message) : base(message)
        {
        }

        public ChatModelException(string message, Exception innerException) : base(message, innerException)
        {
        }
    }
}
