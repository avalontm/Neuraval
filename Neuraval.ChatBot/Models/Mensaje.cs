namespace Neuraval.ChatBot.Models
{
    public class Mensaje
    {
        public string Texto { get; set; }
        public bool EsUsuario { get; set; }
        public DateTime FechaHora { get; set; }
        public float Confianza { get; set; }

        public Mensaje()
        {
            Texto = string.Empty;
            EsUsuario = false;
            FechaHora = DateTime.Now;
            Confianza = 0.0f;
        }

        public Mensaje(string texto, bool esUsuario)
        {
            Texto = texto;
            EsUsuario = esUsuario;
            FechaHora = DateTime.Now;
            Confianza = 0.0f;
        }

        public Mensaje(string texto, bool esUsuario, float confianza)
        {
            Texto = texto;
            EsUsuario = esUsuario;
            FechaHora = DateTime.Now;
            Confianza = confianza;
        }

        public override string ToString()
        {
            string emisor = EsUsuario ? "User" : "Bot";
            return $"[{FechaHora:HH:mm:ss}] {emisor}: {Texto}";
        }

        public string ToStringWithConfidence()
        {
            string emisor = EsUsuario ? "User" : "Bot";
            if (EsUsuario)
            {
                return $"[{FechaHora:HH:mm:ss}] {emisor}: {Texto}";
            }
            return $"[{FechaHora:HH:mm:ss}] {emisor}: {Texto} (Confidence: {Confianza:P2})";
        }
    }
}