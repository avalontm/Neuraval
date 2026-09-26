namespace Neuraval.Core.Tokenizers
{
    /// <summary>
    /// Prefijo y sufijo literales que un chat template "por rol" antepone/agrega
    /// al contenido de un mensaje de ese rol.
    ///
    /// Se usa para formatos que no siguen el esquema genérico de ChatML
    /// (TurnStart + nombre de rol + separador + contenido + TurnEnd igual para
    /// todos los roles), como el de Mistral, donde el turno de usuario se envuelve
    /// en "[INST] ... [/INST]" y el de assistant simplemente termina en "&lt;/s&gt;",
    /// sin una etiqueta de rol visible.
    /// </summary>
    public sealed class ChatRoleTemplate
    {
        public string Prefix { get; set; } = string.Empty;

        public string Suffix { get; set; } = string.Empty;
    }
}
