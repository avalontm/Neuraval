namespace Neuraval.ChatBot.Models
{
    public class DatoEntrenamiento
    {
        public string TextoEntrada { get; set; }
        public List<string> Palabras { get; set; }
        public int CategoriaRespuesta { get; set; }
        public string TextoRespuesta { get; set; }
        public List<string> Etiquetas { get; set; }

        public DatoEntrenamiento()
        {
            TextoEntrada = string.Empty;
            Palabras = new List<string>();
            CategoriaRespuesta = 0;
            TextoRespuesta = string.Empty;
            Etiquetas = new List<string>();
        }

        public DatoEntrenamiento(string textoEntrada, int categoriaRespuesta, string textoRespuesta)
        {
            TextoEntrada = textoEntrada;
            Palabras = new List<string>();
            CategoriaRespuesta = categoriaRespuesta;
            TextoRespuesta = textoRespuesta;
            Etiquetas = new List<string>();
        }

        public DatoEntrenamiento(List<string> palabras, int categoriaRespuesta)
        {
            TextoEntrada = string.Join(" ", palabras);
            Palabras = palabras;
            CategoriaRespuesta = categoriaRespuesta;
            TextoRespuesta = string.Empty;
            Etiquetas = new List<string>();
        }

        public void AgregarEtiqueta(string etiqueta)
        {
            if (!string.IsNullOrWhiteSpace(etiqueta) && !Etiquetas.Contains(etiqueta))
            {
                Etiquetas.Add(etiqueta);
            }
        }

        public void EliminarEtiqueta(string etiqueta)
        {
            Etiquetas.Remove(etiqueta);
        }

        public bool TieneEtiqueta(string etiqueta)
        {
            return Etiquetas.Contains(etiqueta);
        }

        public override string ToString()
        {
            return $"Input: '{TextoEntrada}' -> Category: {CategoriaRespuesta}";
        }

        public string ToDetailedString()
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"Input Text: {TextoEntrada}");
            sb.AppendLine($"Words: {string.Join(", ", Palabras)}");
            sb.AppendLine($"Response Category: {CategoriaRespuesta}");
            sb.AppendLine($"Response Text: {TextoRespuesta}");
            sb.AppendLine($"Tags: {string.Join(", ", Etiquetas)}");
            return sb.ToString();
        }
    }

    public class ConjuntoDatosEntrenamiento
    {
        public Dictionary<string, int> Vocabulario { get; set; }
        public List<string> Respuestas { get; set; }
        public List<DatoEntrenamiento> Datos { get; set; }

        public ConjuntoDatosEntrenamiento()
        {
            Vocabulario = new Dictionary<string, int>();
            Respuestas = new List<string>();
            Datos = new List<DatoEntrenamiento>();
        }

        public void AgregarDato(DatoEntrenamiento dato)
        {
            if (dato == null)
            {
                throw new ArgumentNullException(nameof(dato));
            }

            Datos.Add(dato);
        }

        public void AgregarPalabraVocabulario(string palabra)
        {
            if (!string.IsNullOrWhiteSpace(palabra) && !Vocabulario.ContainsKey(palabra))
            {
                Vocabulario[palabra] = Vocabulario.Count;
            }
        }

        public void AgregarRespuesta(string respuesta)
        {
            if (!string.IsNullOrWhiteSpace(respuesta) && !Respuestas.Contains(respuesta))
            {
                Respuestas.Add(respuesta);
            }
        }

        public int ObtenerTamanoVocabulario()
        {
            return Vocabulario.Count;
        }

        public int ObtenerNumeroRespuestas()
        {
            return Respuestas.Count;
        }

        public int ObtenerNumeroDatos()
        {
            return Datos.Count;
        }
    }
}