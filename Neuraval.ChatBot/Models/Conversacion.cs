namespace Neuraval.ChatBot.Models
{
    public class Conversacion
    {
        public string Id { get; set; }
        public List<Mensaje> Mensajes { get; set; }
        public DateTime FechaInicio { get; set; }
        public DateTime FechaUltimaActividad { get; set; }
        public bool EstaActiva { get; set; }

        public Conversacion()
        {
            Id = Guid.NewGuid().ToString();
            Mensajes = new List<Mensaje>();
            FechaInicio = DateTime.Now;
            FechaUltimaActividad = DateTime.Now;
            EstaActiva = true;
        }

        public Conversacion(string id)
        {
            Id = id;
            Mensajes = new List<Mensaje>();
            FechaInicio = DateTime.Now;
            FechaUltimaActividad = DateTime.Now;
            EstaActiva = true;
        }

        public void AgregarMensaje(Mensaje mensaje)
        {
            if (mensaje == null)
            {
                throw new ArgumentNullException(nameof(mensaje));
            }

            Mensajes.Add(mensaje);
            FechaUltimaActividad = DateTime.Now;
        }

        public void AgregarMensaje(string texto, bool esUsuario, float confianza = 0.0f)
        {
            var mensaje = new Mensaje(texto, esUsuario, confianza);
            AgregarMensaje(mensaje);
        }

        public Mensaje ObtenerUltimoMensaje()
        {
            return Mensajes.LastOrDefault();
        }

        public List<Mensaje> ObtenerMensajesUsuario()
        {
            return Mensajes.Where(m => m.EsUsuario).ToList();
        }

        public List<Mensaje> ObtenerMensajesBot()
        {
            return Mensajes.Where(m => !m.EsUsuario).ToList();
        }

        public int ContarMensajes()
        {
            return Mensajes.Count;
        }

        public void LimpiarConversacion()
        {
            Mensajes.Clear();
            FechaUltimaActividad = DateTime.Now;
        }

        public void FinalizarConversacion()
        {
            EstaActiva = false;
            FechaUltimaActividad = DateTime.Now;
        }

        public TimeSpan ObtenerDuracion()
        {
            return FechaUltimaActividad - FechaInicio;
        }

        public List<Mensaje> ObtenerUltimosNMensajes(int n)
        {
            return Mensajes.TakeLast(n).ToList();
        }

        public override string ToString()
        {
            return $"Conversation {Id} - Messages: {Mensajes.Count} - Active: {EstaActiva}";
        }

        public string ToDetailedString()
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"Conversation ID: {Id}");
            sb.AppendLine($"Start Time: {FechaInicio}");
            sb.AppendLine($"Last Activity: {FechaUltimaActividad}");
            sb.AppendLine($"Duration: {ObtenerDuracion()}");
            sb.AppendLine($"Messages: {Mensajes.Count}");
            sb.AppendLine($"Active: {EstaActiva}");
            sb.AppendLine();

            foreach (var mensaje in Mensajes)
            {
                sb.AppendLine(mensaje.ToStringWithConfidence());
            }

            return sb.ToString();
        }
    }
}