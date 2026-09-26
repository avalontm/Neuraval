using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Neuraval.Abstractions;

namespace Neuraval.Core.Tokenizers
{
    /// <summary>
    /// Intérprete de un subconjunto de Jinja2 suficiente para renderizar la
    /// gran mayoría de los "chat_template" que traen los GGUF modernos en el
    /// metadato "tokenizer.chat_template" -- el mismo campo que usan
    /// transformers y llama.cpp para saber exactamente cómo armar el prompt
    /// de cada modelo (Llama, Mistral, Qwen/ChatML, Gemma, Phi, DeepSeek,
    /// Zephyr, etc.).
    ///
    /// Antes de esto, Neuraval solo tenía presets hardcodeados (ChatMl,
    /// Mistral) elegidos por arquitectura, lo cual es una aproximación que
    /// falla en cuanto un modelo usa una variante distinta del formato que
    /// "adivinamos". Leer y ejecutar el chat_template real del propio GGUF es
    /// lo que de verdad generaliza a "la mayoría de los modelos", porque cada
    /// checkpoint declara ahí exactamente el formato con el que fue
    /// entrenado/fine-tuneado.
    ///
    /// Esto NO es un motor Jinja completo. Deliberadamente no soporta:
    /// - Macros, imports, herencia de templates ({% extends %}, {% include %}).
    /// - Filtros de "tool calling" que cambian la forma de los datos
    ///   (selectattr/rejectattr/map/groupby/sort/unique/tojson).
    /// - El filtro de bucle inline "{% for x in seq if cond %}" (se ignora la
    ///   parte "if cond" salvo cuando en realidad es un condicional ternario
    ///   "A if cond else B" sobre el propio iterable).
    /// Si el template usa algo no soportado, se lanza
    /// <see cref="JinjaTemplateException"/> y el llamador (GgufChatModel)
    /// puede caer de nuevo al preset heurístico por arquitectura en vez de
    /// romper la carga del modelo.
    /// </summary>
    public static class JinjaChatTemplateEngine
    {
        public static string Render(
            string template,
            IReadOnlyList<ChatMessage> messages,
            bool addGenerationPrompt,
            string bosToken,
            string eosToken)
        {
            if (template == null)
                throw new ArgumentNullException(nameof(template));

            if (messages == null)
                throw new ArgumentNullException(nameof(messages));

            try
            {
                var nodes = JinjaParser.Parse(template);

                var root = new JinjaScope(null);
                root.SetLocal("messages", BuildMessages(messages));
                root.SetLocal("add_generation_prompt", addGenerationPrompt);
                root.SetLocal("bos_token", bosToken ?? string.Empty);
                root.SetLocal("eos_token", eosToken ?? string.Empty);

                var sb = new StringBuilder();
                JinjaEvaluator.ExecuteBlock(nodes, root, sb);
                return sb.ToString();
            }
            catch (JinjaTemplateException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new JinjaTemplateException("No se pudo renderizar el chat_template Jinja del GGUF: " + ex.Message, ex);
            }
        }

        private static List<object?> BuildMessages(IReadOnlyList<ChatMessage> messages)
        {
            var list = new List<object?>(messages.Count);

            foreach (var message in messages)
            {
                var dict = new Dictionary<string, object?>
                {
                    ["role"] = RoleName(message.Role),
                    ["content"] = message.Content ?? string.Empty
                };

                list.Add(dict);
            }

            return list;
        }

        private static string RoleName(ChatRole role)
        {
            return role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Assistant => "assistant",
                ChatRole.Tool => "tool",
                _ => role.ToString().ToLowerInvariant()
            };
        }
    }

    /// <summary>
    /// Se lanza cuando el chat_template Jinja de un GGUF usa una construcción
    /// que este motor (deliberadamente parcial) no soporta, o cuando el
    /// template está mal formado. El llamador puede capturarla para caer de
    /// nuevo a un preset heurístico en vez de romper la carga del modelo.
    /// </summary>
    public sealed class JinjaTemplateException : Exception
    {
        public JinjaTemplateException(string message) : base(message)
        {
        }

        public JinjaTemplateException(string message, Exception inner) : base(message, inner)
        {
        }
    }

    // =====================================================================
    // Valor "undefined" de Jinja: falsy, se imprime como cadena vacía, y es
    // lo que se devuelve al acceder a una variable/atributo que no existe
    // (en vez de lanzar una excepción), que es exactamente lo que necesitamos
    // para que ramas de templates sobre "tools"/"name" que ChatMessage no
    // tiene simplemente se salteen en vez de romper el render.
    // =====================================================================
    internal sealed class JinjaUndefined
    {
        public static readonly JinjaUndefined Instance = new();
        private JinjaUndefined()
        {
        }

        public override string ToString() => string.Empty;
    }

    internal sealed class JinjaBuiltinFunction
    {
        public string Name { get; }
        public JinjaBuiltinFunction(string name)
        {
            Name = name;
        }
    }

    internal sealed class JinjaScope
    {
        private readonly JinjaScope? _parent;
        private readonly Dictionary<string, object?> _vars = new();

        public JinjaScope(JinjaScope? parent)
        {
            _parent = parent;
        }

        public void SetLocal(string name, object? value)
        {
            _vars[name] = value;
        }

        public bool TryGet(string name, out object? value)
        {
            var scope = this;
            while (scope != null)
            {
                if (scope._vars.TryGetValue(name, out value))
                    return true;
                scope = scope._parent;
            }

            value = null;
            return false;
        }
    }

    // =====================================================================
    // Lexer de plantilla: separa el texto crudo en Texto / {{ Output }} /
    // {% Tag %}, descarta comentarios {# ... #}, y aplica el control de
    // espacios en blanco de Jinja: tanto los marcadores explícitos "-" como
    // el comportamiento por defecto que usa HuggingFace al renderizar
    // chat_template (trim_blocks + lstrip_blocks), sin el cual casi todos los
    // templates reales quedarían llenos de saltos de línea y espacios de más.
    // =====================================================================
    internal enum RawKind { Text, Output, Tag }

    internal sealed class RawNode
    {
        public RawKind Kind;
        public string Content = string.Empty;
    }

    internal static class JinjaTemplateLexer
    {
        private enum InternalKind { Text, Output, Tag, Comment }

        private sealed class InternalToken
        {
            public InternalKind Kind;
            public string Content = string.Empty;
            public bool TrimLeft;
            public bool TrimRight;
        }

        private static readonly Regex TokenRegex = new(
            @"\{\{(?<otl>-)?(?<out>.*?)(?<otr>-)?\}\}|\{%(?<ttl>-)?(?<tag>.*?)(?<ttr>-)?%\}|\{\#(?<ctl>-)?.*?(?<ctr>-)?\#\}",
            RegexOptions.Singleline | RegexOptions.Compiled);

        public static List<RawNode> LexTemplate(string template)
        {
            var tokens = new List<InternalToken>();
            int last = 0;

            foreach (Match m in TokenRegex.Matches(template))
            {
                if (m.Index > last)
                    tokens.Add(new InternalToken { Kind = InternalKind.Text, Content = template.Substring(last, m.Index - last) });

                if (m.Groups["out"].Success)
                {
                    tokens.Add(new InternalToken
                    {
                        Kind = InternalKind.Output,
                        Content = m.Groups["out"].Value,
                        TrimLeft = m.Groups["otl"].Success,
                        TrimRight = m.Groups["otr"].Success
                    });
                }
                else if (m.Groups["tag"].Success)
                {
                    tokens.Add(new InternalToken
                    {
                        Kind = InternalKind.Tag,
                        Content = m.Groups["tag"].Value,
                        TrimLeft = m.Groups["ttl"].Success,
                        TrimRight = m.Groups["ttr"].Success
                    });
                }
                else
                {
                    tokens.Add(new InternalToken
                    {
                        Kind = InternalKind.Comment,
                        TrimLeft = m.Groups["ctl"].Success,
                        TrimRight = m.Groups["ctr"].Success
                    });
                }

                last = m.Index + m.Length;
            }

            if (last < template.Length)
                tokens.Add(new InternalToken { Kind = InternalKind.Text, Content = template.Substring(last) });

            ApplyWhitespaceControl(tokens);

            var result = new List<RawNode>();
            foreach (var t in tokens)
            {
                if (t.Kind == InternalKind.Comment)
                    continue;

                var kind = t.Kind switch
                {
                    InternalKind.Text => RawKind.Text,
                    InternalKind.Output => RawKind.Output,
                    InternalKind.Tag => RawKind.Tag,
                    _ => RawKind.Text
                };

                result.Add(new RawNode { Kind = kind, Content = t.Content });
            }

            return result;
        }

        private static void ApplyWhitespaceControl(List<InternalToken> tokens)
        {
            // lstrip_blocks: si un "{% %}"/"{# #}" es lo único (aparte de
            // espacios/tabs) en su línea, se recorta esa corrida de
            // espacios/tabs del texto que lo precede.
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i].Kind != InternalKind.Tag && tokens[i].Kind != InternalKind.Comment)
                    continue;

                if (i > 0 && tokens[i - 1].Kind == InternalKind.Text)
                {
                    var text = tokens[i - 1].Content;
                    int cut = text.Length;
                    while (cut > 0 && (text[cut - 1] == ' ' || text[cut - 1] == '\t'))
                        cut--;

                    if (cut < text.Length && (cut == 0 || text[cut - 1] == '\n'))
                        tokens[i - 1].Content = text.Substring(0, cut);
                }
            }

            // trim_blocks: se recorta el primer salto de línea inmediatamente
            // después de un "{% %}"/"{# #}".
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i].Kind != InternalKind.Tag && tokens[i].Kind != InternalKind.Comment)
                    continue;

                if (i + 1 < tokens.Count && tokens[i + 1].Kind == InternalKind.Text)
                {
                    var text = tokens[i + 1].Content;
                    if (text.StartsWith("\r\n", StringComparison.Ordinal))
                        tokens[i + 1].Content = text.Substring(2);
                    else if (text.StartsWith("\n", StringComparison.Ordinal))
                        tokens[i + 1].Content = text.Substring(1);
                }
            }

            // Marcadores '-' explícitos: recortan TODO el espacio en blanco
            // adyacente (incluyendo saltos de línea), sin importar el tipo de
            // tag, y se aplican encima de lo anterior.
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i].Kind == InternalKind.Text)
                    continue;

                if (tokens[i].TrimLeft && i > 0 && tokens[i - 1].Kind == InternalKind.Text)
                    tokens[i - 1].Content = tokens[i - 1].Content.TrimEnd();

                if (tokens[i].TrimRight && i + 1 < tokens.Count && tokens[i + 1].Kind == InternalKind.Text)
                    tokens[i + 1].Content = tokens[i + 1].Content.TrimStart();
            }
        }
    }

    // =====================================================================
    // Lexer de expresiones: tokeniza el contenido de un "{{ ... }}" o de una
    // etiqueta "{% ... %}" en identificadores / números / strings / operadores.
    // =====================================================================
    internal enum TokType { Ident, Number, String, Op, End }

    internal sealed class Token
    {
        public TokType Type;
        public string Text = string.Empty;
        public object? Value;
    }

    internal static class JinjaExprLexer
    {
        private static readonly string[] MultiCharOps = { "==", "!=", "<=", ">=", "//" };

        public static List<Token> Tokenize(string content)
        {
            var tokens = new List<Token>();
            int i = 0;
            int n = content.Length;

            while (i < n)
            {
                char c = content[i];

                if (char.IsWhiteSpace(c))
                {
                    i++;
                    continue;
                }

                if (char.IsLetter(c) || c == '_')
                {
                    int start = i;
                    while (i < n && (char.IsLetterOrDigit(content[i]) || content[i] == '_'))
                        i++;
                    tokens.Add(new Token { Type = TokType.Ident, Text = content.Substring(start, i - start) });
                    continue;
                }

                if (char.IsDigit(c))
                {
                    int start = i;
                    bool isFloat = false;
                    while (i < n && char.IsDigit(content[i]))
                        i++;
                    if (i < n && content[i] == '.' && i + 1 < n && char.IsDigit(content[i + 1]))
                    {
                        isFloat = true;
                        i++;
                        while (i < n && char.IsDigit(content[i]))
                            i++;
                    }

                    string numText = content.Substring(start, i - start);
                    tokens.Add(new Token
                    {
                        Type = TokType.Number,
                        Text = numText,
                        Value = isFloat
                            ? double.Parse(numText, CultureInfo.InvariantCulture)
                            : (object)long.Parse(numText, CultureInfo.InvariantCulture)
                    });
                    continue;
                }

                if (c == '\'' || c == '"')
                {
                    char quote = c;
                    i++;
                    var sb = new StringBuilder();
                    while (i < n && content[i] != quote)
                    {
                        if (content[i] == '\\' && i + 1 < n)
                        {
                            char next = content[i + 1];
                            sb.Append(next switch
                            {
                                'n' => '\n',
                                't' => '\t',
                                'r' => '\r',
                                '\\' => '\\',
                                '\'' => '\'',
                                '"' => '"',
                                _ => next
                            });
                            i += 2;
                        }
                        else
                        {
                            sb.Append(content[i]);
                            i++;
                        }
                    }

                    if (i >= n)
                        throw new JinjaTemplateException("String sin cerrar en expresión Jinja: " + content);

                    i++; // comilla de cierre
                    var strValue = sb.ToString();
                    tokens.Add(new Token { Type = TokType.String, Text = strValue, Value = strValue });
                    continue;
                }

                bool matchedMulti = false;
                foreach (var op in MultiCharOps)
                {
                    if (i + op.Length <= n && string.CompareOrdinal(content, i, op, 0, op.Length) == 0)
                    {
                        tokens.Add(new Token { Type = TokType.Op, Text = op });
                        i += op.Length;
                        matchedMulti = true;
                        break;
                    }
                }

                if (matchedMulti)
                    continue;

                if ("()[]{}.,|:=<>+-*/%~".IndexOf(c) >= 0)
                {
                    tokens.Add(new Token { Type = TokType.Op, Text = c.ToString() });
                    i++;
                    continue;
                }

                throw new JinjaTemplateException($"Carácter inesperado '{c}' en expresión Jinja: {content}");
            }

            tokens.Add(new Token { Type = TokType.End, Text = string.Empty });
            return tokens;
        }
    }

    // =====================================================================
    // AST de expresiones.
    // =====================================================================
    internal enum ExprKind
    {
        Literal, Identifier, Attr, Subscript, Slice, Call, Filter,
        BinOp, Compare, And, Or, Not, Neg, Concat, In, IsTest, Ternary, ListLiteral
    }

    internal sealed class JinjaExpr
    {
        public ExprKind Kind;
        public object? LiteralValue;
        public string? Name;
        public JinjaExpr? Target;
        public JinjaExpr? Left;
        public JinjaExpr? Right;
        public JinjaExpr? Extra;
        public string? Op;
        public bool Negate;
        public List<(string? Name, JinjaExpr Value)>? CallArgs;
        public List<JinjaExpr>? Items;

        public static JinjaExpr Literal(object? value) => new() { Kind = ExprKind.Literal, LiteralValue = value };
        public static JinjaExpr Identifier(string name) => new() { Kind = ExprKind.Identifier, Name = name };
        public static JinjaExpr Attr(JinjaExpr target, string name) => new() { Kind = ExprKind.Attr, Target = target, Name = name };
        public static JinjaExpr Subscript(JinjaExpr target, JinjaExpr index) => new() { Kind = ExprKind.Subscript, Target = target, Left = index };
        public static JinjaExpr Slice(JinjaExpr target, JinjaExpr? start, JinjaExpr? stop) => new() { Kind = ExprKind.Slice, Target = target, Left = start, Right = stop };
        public static JinjaExpr Call(JinjaExpr callee, List<(string? Name, JinjaExpr Value)> args) => new() { Kind = ExprKind.Call, Target = callee, CallArgs = args };
        public static JinjaExpr Filter(JinjaExpr target, string name, List<(string? Name, JinjaExpr Value)> args) => new() { Kind = ExprKind.Filter, Target = target, Name = name, CallArgs = args };
        public static JinjaExpr BinOp(string op, JinjaExpr l, JinjaExpr r) => new() { Kind = ExprKind.BinOp, Op = op, Left = l, Right = r };
        public static JinjaExpr Compare(string op, JinjaExpr l, JinjaExpr r) => new() { Kind = ExprKind.Compare, Op = op, Left = l, Right = r };
        public static JinjaExpr And(JinjaExpr l, JinjaExpr r) => new() { Kind = ExprKind.And, Left = l, Right = r };
        public static JinjaExpr Or(JinjaExpr l, JinjaExpr r) => new() { Kind = ExprKind.Or, Left = l, Right = r };
        public static JinjaExpr Not(JinjaExpr x) => new() { Kind = ExprKind.Not, Target = x };
        public static JinjaExpr Neg(JinjaExpr x) => new() { Kind = ExprKind.Neg, Target = x };
        public static JinjaExpr Concat(JinjaExpr l, JinjaExpr r) => new() { Kind = ExprKind.Concat, Left = l, Right = r };
        public static JinjaExpr In(JinjaExpr l, JinjaExpr r, bool negate) => new() { Kind = ExprKind.In, Left = l, Right = r, Negate = negate };
        public static JinjaExpr IsTest(JinjaExpr target, string name, bool negate, List<(string? Name, JinjaExpr Value)> args) => new() { Kind = ExprKind.IsTest, Target = target, Name = name, Negate = negate, CallArgs = args };
        public static JinjaExpr Ternary(JinjaExpr value, JinjaExpr cond, JinjaExpr elseValue) => new() { Kind = ExprKind.Ternary, Left = value, Right = elseValue, Extra = cond };
        public static JinjaExpr ListLiteral(List<JinjaExpr> items) => new() { Kind = ExprKind.ListLiteral, Items = items };
    }

    // =====================================================================
    // Parser de expresiones (recursive descent, precedencia estilo Jinja2):
    // ternario < or < and < not < comparación/is/in < concat(~) < + - <
    // * / // % < unario < filtros(|) < postfijo(.[]()) < primario.
    // =====================================================================
    internal sealed class JinjaExpressionParser
    {
        private readonly List<Token> _tokens;
        private int _pos;

        public JinjaExpressionParser(List<Token> tokens)
        {
            _tokens = tokens;
            _pos = 0;
        }

        private Token Current => _tokens[_pos];
        private bool AtEnd => Current.Type == TokType.End;

        private bool CheckIdent(string text) => Current.Type == TokType.Ident && Current.Text == text;
        private bool CheckOp(string text) => Current.Type == TokType.Op && Current.Text == text;

        private bool MatchIdent(string text)
        {
            if (CheckIdent(text)) { _pos++; return true; }
            return false;
        }

        private bool MatchOp(string text)
        {
            if (CheckOp(text)) { _pos++; return true; }
            return false;
        }

        private void ExpectOp(string text)
        {
            if (!CheckOp(text))
                throw new JinjaTemplateException($"Se esperaba '{text}' en expresión Jinja, se encontró '{Current.Text}'.");
            _pos++;
        }

        private Token ExpectIdentAny()
        {
            if (Current.Type != TokType.Ident)
                throw new JinjaTemplateException($"Se esperaba un identificador en expresión Jinja, se encontró '{Current.Text}'.");
            var t = Current;
            _pos++;
            return t;
        }

        public JinjaExpr ParseExpression()
        {
            var expr = ParseTernary();
            if (!AtEnd)
                throw new JinjaTemplateException($"Token inesperado '{Current.Text}' en expresión Jinja.");
            return expr;
        }

        private JinjaExpr ParseTernary()
        {
            var value = ParseOr();

            if (MatchIdent("if"))
            {
                var cond = ParseOr();
                JinjaExpr elseValue = MatchIdent("else")
                    ? ParseTernary()
                    : JinjaExpr.Literal(JinjaUndefined.Instance);

                return JinjaExpr.Ternary(value, cond, elseValue);
            }

            return value;
        }

        private JinjaExpr ParseOr()
        {
            var left = ParseAnd();
            while (MatchIdent("or"))
                left = JinjaExpr.Or(left, ParseAnd());
            return left;
        }

        private JinjaExpr ParseAnd()
        {
            var left = ParseNot();
            while (MatchIdent("and"))
                left = JinjaExpr.And(left, ParseNot());
            return left;
        }

        private JinjaExpr ParseNot()
        {
            if (MatchIdent("not"))
                return JinjaExpr.Not(ParseNot());
            return ParseComparison();
        }

        private JinjaExpr ParseComparison()
        {
            var left = ParseConcat();

            while (true)
            {
                if (Current.Type == TokType.Op && Current.Text is "==" or "!=" or "<" or "<=" or ">" or ">=")
                {
                    string op = Current.Text;
                    _pos++;
                    left = JinjaExpr.Compare(op, left, ParseConcat());
                    continue;
                }

                if (CheckIdent("in"))
                {
                    _pos++;
                    left = JinjaExpr.In(left, ParseConcat(), negate: false);
                    continue;
                }

                if (CheckIdent("not"))
                {
                    int save = _pos;
                    _pos++;
                    if (CheckIdent("in"))
                    {
                        _pos++;
                        left = JinjaExpr.In(left, ParseConcat(), negate: true);
                        continue;
                    }
                    _pos = save;
                    break;
                }

                if (CheckIdent("is"))
                {
                    _pos++;
                    bool negate = MatchIdent("not");
                    var testToken = ExpectIdentAny();
                    var args = MatchOp("(") ? ParseArgList() : new List<(string? Name, JinjaExpr Value)>();
                    left = JinjaExpr.IsTest(left, testToken.Text, negate, args);
                    continue;
                }

                break;
            }

            return left;
        }

        private JinjaExpr ParseConcat()
        {
            var left = ParseAdditive();
            while (MatchOp("~"))
                left = JinjaExpr.Concat(left, ParseAdditive());
            return left;
        }

        private JinjaExpr ParseAdditive()
        {
            var left = ParseMultiplicative();
            while (CheckOp("+") || CheckOp("-"))
            {
                string op = Current.Text;
                _pos++;
                left = JinjaExpr.BinOp(op, left, ParseMultiplicative());
            }
            return left;
        }

        private JinjaExpr ParseMultiplicative()
        {
            var left = ParseUnary();
            while (CheckOp("*") || CheckOp("/") || CheckOp("//") || CheckOp("%"))
            {
                string op = Current.Text;
                _pos++;
                left = JinjaExpr.BinOp(op, left, ParseUnary());
            }
            return left;
        }

        private JinjaExpr ParseUnary()
        {
            if (MatchOp("-"))
                return JinjaExpr.Neg(ParseUnary());
            if (MatchOp("+"))
                return ParseUnary();
            return ParseFilterChain();
        }

        private JinjaExpr ParseFilterChain()
        {
            var left = ParsePostfix();
            while (MatchOp("|"))
            {
                var nameToken = ExpectIdentAny();
                var args = MatchOp("(") ? ParseArgList() : new List<(string? Name, JinjaExpr Value)>();
                left = JinjaExpr.Filter(left, nameToken.Text, args);
            }
            return left;
        }

        private JinjaExpr ParsePostfix()
        {
            var expr = ParsePrimary();

            while (true)
            {
                if (MatchOp("."))
                {
                    var name = ExpectIdentAny();
                    expr = JinjaExpr.Attr(expr, name.Text);
                    continue;
                }

                if (MatchOp("["))
                {
                    expr = ParseSubscript(expr);
                    continue;
                }

                if (MatchOp("("))
                {
                    expr = JinjaExpr.Call(expr, ParseArgList());
                    continue;
                }

                break;
            }

            return expr;
        }

        private JinjaExpr ParseSubscript(JinjaExpr target)
        {
            JinjaExpr? start = null;
            JinjaExpr? stop = null;
            bool isSlice = false;

            if (!CheckOp(":") && !CheckOp("]"))
                start = ParseTernary();

            if (MatchOp(":"))
            {
                isSlice = true;
                if (!CheckOp(":") && !CheckOp("]"))
                    stop = ParseTernary();

                // Un posible "step" (a[::2]) se consume pero se ignora: no es
                // común en chat_templates y así no rompemos el parseo del resto.
                if (MatchOp(":") && !CheckOp("]"))
                    ParseTernary();
            }

            ExpectOp("]");

            return isSlice ? JinjaExpr.Slice(target, start, stop) : JinjaExpr.Subscript(target, start!);
        }

        private List<(string? Name, JinjaExpr Value)> ParseArgList()
        {
            var args = new List<(string? Name, JinjaExpr Value)>();

            if (MatchOp(")"))
                return args;

            while (true)
            {
                string? name = null;

                if (Current.Type == TokType.Ident && _pos + 1 < _tokens.Count
                    && _tokens[_pos + 1].Type == TokType.Op && _tokens[_pos + 1].Text == "=")
                {
                    name = Current.Text;
                    _pos += 2;
                }

                args.Add((name, ParseTernary()));

                if (MatchOp(","))
                    continue;

                ExpectOp(")");
                break;
            }

            return args;
        }

        private JinjaExpr ParsePrimary()
        {
            var tok = Current;

            if (tok.Type == TokType.Number)
            {
                _pos++;
                return JinjaExpr.Literal(tok.Value);
            }

            if (tok.Type == TokType.String)
            {
                _pos++;
                return JinjaExpr.Literal(tok.Value);
            }

            if (tok.Type == TokType.Ident)
            {
                if (tok.Text is "true" or "True")
                {
                    _pos++;
                    return JinjaExpr.Literal(true);
                }

                if (tok.Text is "false" or "False")
                {
                    _pos++;
                    return JinjaExpr.Literal(false);
                }

                if (tok.Text is "none" or "None")
                {
                    _pos++;
                    return JinjaExpr.Literal(null);
                }

                _pos++;
                return JinjaExpr.Identifier(tok.Text);
            }

            if (CheckOp("("))
            {
                _pos++;
                var inner = ParseTernary();
                ExpectOp(")");
                return inner;
            }

            if (CheckOp("["))
            {
                _pos++;
                var items = new List<JinjaExpr>();
                if (!CheckOp("]"))
                {
                    while (true)
                    {
                        items.Add(ParseTernary());
                        if (MatchOp(","))
                            continue;
                        break;
                    }
                }
                ExpectOp("]");
                return JinjaExpr.ListLiteral(items);
            }

            throw new JinjaTemplateException($"No se pudo interpretar la expresión Jinja cerca de '{tok.Text}'.");
        }
    }

    // =====================================================================
    // AST de sentencias + parser de plantilla completo.
    // =====================================================================
    internal enum StmtKind { Text, Output, For, If, Set, BlockSet }

    internal sealed class StmtNode
    {
        public StmtKind Kind;
        public string? Text;
        public JinjaExpr? Expr;

        public string? LoopVar1;
        public string? LoopVar2;
        public JinjaExpr? IterableExpr;
        public List<StmtNode>? Body;
        public List<StmtNode>? ElseBody;

        public List<(JinjaExpr Condition, List<StmtNode> Body)>? Branches;

        public string? SetTargetName;
        public string? SetTargetAttr;
    }

    internal sealed class JinjaParser
    {
        private static readonly HashSet<string> NoStop = new();

        private readonly List<RawNode> _raw;
        private int _i;

        private JinjaParser(List<RawNode> raw)
        {
            _raw = raw;
            _i = 0;
        }

        public static List<StmtNode> Parse(string template)
        {
            var raw = JinjaTemplateLexer.LexTemplate(template);
            var parser = new JinjaParser(raw);
            var nodes = parser.ParseBlock(NoStop);

            if (parser._i < parser._raw.Count)
                throw new JinjaTemplateException("Se encontró una etiqueta Jinja de cierre sin apertura correspondiente.");

            return nodes;
        }

        private bool AtEnd => _i >= _raw.Count;

        private RawNode CurrentOrThrow()
        {
            if (AtEnd)
                throw new JinjaTemplateException("Bloque Jinja sin cerrar: se llegó al final del template antes de encontrar la etiqueta de cierre esperada.");
            return _raw[_i];
        }

        private static string FirstWord(string content)
        {
            var trimmed = content.Trim();
            int idx = 0;
            while (idx < trimmed.Length && !char.IsWhiteSpace(trimmed[idx]))
                idx++;
            return trimmed.Substring(0, idx);
        }

        private List<StmtNode> ParseBlock(HashSet<string> stopKeywords)
        {
            var result = new List<StmtNode>();

            while (!AtEnd)
            {
                var node = _raw[_i];

                if (node.Kind == RawKind.Text)
                {
                    if (node.Content.Length > 0)
                        result.Add(new StmtNode { Kind = StmtKind.Text, Text = node.Content });
                    _i++;
                    continue;
                }

                if (node.Kind == RawKind.Output)
                {
                    var expr = new JinjaExpressionParser(JinjaExprLexer.Tokenize(node.Content)).ParseExpression();
                    result.Add(new StmtNode { Kind = StmtKind.Output, Expr = expr });
                    _i++;
                    continue;
                }

                string trimmedTag = node.Content.Trim();
                string keyword = FirstWord(trimmedTag);

                if (stopKeywords.Contains(keyword))
                    return result;

                switch (keyword)
                {
                    case "for":
                        result.Add(ParseFor(trimmedTag));
                        continue;

                    case "if":
                        result.Add(ParseIf(trimmedTag));
                        continue;

                    case "set":
                        result.Add(ParseSet(trimmedTag));
                        continue;

                    case "break":
                    case "continue":
                        // No soportado (poco común en chat_templates): se
                        // ignora en vez de romper el parseo del resto.
                        _i++;
                        continue;

                    default:
                        throw new JinjaTemplateException($"Etiqueta Jinja no soportada: '{{% {trimmedTag} %}}'.");
                }
            }

            return result;
        }

        private StmtNode ParseFor(string trimmedTag)
        {
            _i++; // consumir la etiqueta 'for ...'

            var tokens = JinjaExprLexer.Tokenize(trimmedTag);
            int pos = 0;

            ExpectIdentToken(tokens, ref pos, "for");
            string var1 = ExpectIdentName(tokens, ref pos);
            string? var2 = null;

            if (tokens[pos].Type == TokType.Op && tokens[pos].Text == ",")
            {
                pos++;
                var2 = ExpectIdentName(tokens, ref pos);
            }

            ExpectIdentToken(tokens, ref pos, "in");

            // Nota: no soportamos el filtro inline "{% for x in y if cond %}";
            // se toma todo lo que sigue a 'in' como la expresión iterable
            // completa (si esa cola es en realidad un ternario "A if c else B"
            // funciona igual que siempre).
            var remaining = tokens.GetRange(pos, tokens.Count - pos);
            var iterableExpr = new JinjaExpressionParser(remaining).ParseExpression();

            var forNode = new StmtNode
            {
                Kind = StmtKind.For,
                LoopVar1 = var1,
                LoopVar2 = var2,
                IterableExpr = iterableExpr
            };

            forNode.Body = ParseBlock(new HashSet<string> { "endfor", "else" });

            var terminator = CurrentOrThrow();
            if (FirstWord(terminator.Content.Trim()) == "else")
            {
                _i++;
                forNode.ElseBody = ParseBlock(new HashSet<string> { "endfor" });
                CurrentOrThrow();
            }

            _i++; // consumir 'endfor'
            return forNode;
        }

        private StmtNode ParseIf(string trimmedTag)
        {
            var ifNode = new StmtNode { Kind = StmtKind.If, Branches = new List<(JinjaExpr, List<StmtNode>)>() };
            string currentTagText = trimmedTag;

            while (true)
            {
                _i++; // consumir la etiqueta 'if'/'elif' actual

                var condTokens = JinjaExprLexer.Tokenize(StripLeadingKeyword(currentTagText));
                var condExpr = new JinjaExpressionParser(condTokens).ParseExpression();

                var body = ParseBlock(new HashSet<string> { "elif", "else", "endif" });
                ifNode.Branches.Add((condExpr, body));

                var terminator = CurrentOrThrow();
                string terminatorKeyword = FirstWord(terminator.Content.Trim());

                if (terminatorKeyword == "elif")
                {
                    currentTagText = terminator.Content.Trim();
                    continue;
                }

                if (terminatorKeyword == "else")
                {
                    _i++;
                    ifNode.ElseBody = ParseBlock(new HashSet<string> { "endif" });
                    CurrentOrThrow();
                }

                _i++; // consumir 'endif'
                break;
            }

            return ifNode;
        }

        private static string StripLeadingKeyword(string tagText)
        {
            var trimmed = tagText.Trim();
            int idx = trimmed.IndexOf(' ');
            return idx < 0 ? string.Empty : trimmed.Substring(idx + 1);
        }

        private StmtNode ParseSet(string trimmedTag)
        {
            var tokens = JinjaExprLexer.Tokenize(trimmedTag);
            int pos = 0;
            ExpectIdentToken(tokens, ref pos, "set");

            int eqIdx = -1;
            int depth = 0;
            for (int j = pos; j < tokens.Count; j++)
            {
                var t = tokens[j];
                if (t.Type == TokType.End)
                    break;
                if (t.Type == TokType.Op && (t.Text == "(" || t.Text == "[" || t.Text == "{"))
                    depth++;
                else if (t.Type == TokType.Op && (t.Text == ")" || t.Text == "]" || t.Text == "}"))
                    depth--;
                else if (depth == 0 && t.Type == TokType.Op && t.Text == "=")
                {
                    eqIdx = j;
                    break;
                }
            }

            if (eqIdx < 0)
            {
                // Forma de bloque: {% set nombre %} ... {% endset %}
                string blockName = ExpectIdentName(tokens, ref pos);
                _i++; // consumir la etiqueta 'set'
                var body = ParseBlock(new HashSet<string> { "endset" });
                CurrentOrThrow();
                _i++; // consumir 'endset'

                return new StmtNode { Kind = StmtKind.BlockSet, SetTargetName = blockName, Body = body };
            }

            var targetTokens = tokens.GetRange(pos, eqIdx - pos);
            var valueTokens = tokens.GetRange(eqIdx + 1, tokens.Count - eqIdx - 1);

            string targetName;
            string? targetAttr = null;

            if (targetTokens.Count == 1 && targetTokens[0].Type == TokType.Ident)
            {
                targetName = targetTokens[0].Text;
            }
            else if (targetTokens.Count == 3
                     && targetTokens[0].Type == TokType.Ident
                     && targetTokens[1].Type == TokType.Op && targetTokens[1].Text == "."
                     && targetTokens[2].Type == TokType.Ident)
            {
                targetName = targetTokens[0].Text;
                targetAttr = targetTokens[2].Text;
            }
            else
            {
                throw new JinjaTemplateException("Forma de '{% set %}' no soportada: solo se admite 'set nombre = ...' o 'set ns.attr = ...'.");
            }

            var valueExpr = new JinjaExpressionParser(valueTokens).ParseExpression();

            _i++; // consumir la etiqueta 'set'

            return new StmtNode
            {
                Kind = StmtKind.Set,
                SetTargetName = targetName,
                SetTargetAttr = targetAttr,
                Expr = valueExpr
            };
        }

        private static void ExpectIdentToken(List<Token> tokens, ref int pos, string text)
        {
            if (tokens[pos].Type != TokType.Ident || tokens[pos].Text != text)
                throw new JinjaTemplateException($"Se esperaba '{text}' en una etiqueta Jinja.");
            pos++;
        }

        private static string ExpectIdentName(List<Token> tokens, ref int pos)
        {
            if (tokens[pos].Type != TokType.Ident)
                throw new JinjaTemplateException("Se esperaba un identificador en una etiqueta Jinja.");
            var name = tokens[pos].Text;
            pos++;
            return name;
        }
    }

    // =====================================================================
    // Evaluador: ejecuta el AST de sentencias contra un JinjaScope y produce
    // el texto final.
    // =====================================================================
    internal static class JinjaEvaluator
    {
        public static void ExecuteBlock(List<StmtNode> nodes, JinjaScope scope, StringBuilder sb)
        {
            foreach (var node in nodes)
                ExecuteNode(node, scope, sb);
        }

        private static void ExecuteNode(StmtNode node, JinjaScope scope, StringBuilder sb)
        {
            switch (node.Kind)
            {
                case StmtKind.Text:
                    sb.Append(node.Text);
                    break;

                case StmtKind.Output:
                    sb.Append(JinjaRuntime.Stringify(Eval(node.Expr!, scope)));
                    break;

                case StmtKind.For:
                    ExecuteFor(node, scope, sb);
                    break;

                case StmtKind.If:
                    ExecuteIf(node, scope, sb);
                    break;

                case StmtKind.Set:
                    ExecuteSet(node, scope);
                    break;

                case StmtKind.BlockSet:
                    var inner = new StringBuilder();
                    ExecuteBlock(node.Body!, scope, inner);
                    scope.SetLocal(node.SetTargetName!, inner.ToString());
                    break;
            }
        }

        private static void ExecuteFor(StmtNode node, JinjaScope scope, StringBuilder sb)
        {
            var iterableValue = Eval(node.IterableExpr!, scope);
            var items = JinjaRuntime.ToIterable(iterableValue);

            if (items.Count == 0)
            {
                if (node.ElseBody != null)
                    ExecuteBlock(node.ElseBody, scope, sb);
                return;
            }

            for (int i = 0; i < items.Count; i++)
            {
                var child = new JinjaScope(scope);
                var item = items[i];

                if (node.LoopVar2 != null && item is List<object?> pair && pair.Count == 2)
                {
                    child.SetLocal(node.LoopVar1!, pair[0]);
                    child.SetLocal(node.LoopVar2, pair[1]);
                }
                else
                {
                    child.SetLocal(node.LoopVar1!, item);
                    if (node.LoopVar2 != null)
                        child.SetLocal(node.LoopVar2, JinjaUndefined.Instance);
                }

                var loopInfo = new Dictionary<string, object?>
                {
                    ["index"] = (long)(i + 1),
                    ["index0"] = (long)i,
                    ["first"] = i == 0,
                    ["last"] = i == items.Count - 1,
                    ["length"] = (long)items.Count,
                    ["revindex"] = (long)(items.Count - i),
                    ["revindex0"] = (long)(items.Count - i - 1)
                };
                child.SetLocal("loop", loopInfo);

                ExecuteBlock(node.Body!, child, sb);
            }
        }

        private static void ExecuteIf(StmtNode node, JinjaScope scope, StringBuilder sb)
        {
            foreach (var (condition, body) in node.Branches!)
            {
                if (JinjaRuntime.ToBool(Eval(condition, scope)))
                {
                    ExecuteBlock(body, scope, sb);
                    return;
                }
            }

            if (node.ElseBody != null)
                ExecuteBlock(node.ElseBody, scope, sb);
        }

        private static void ExecuteSet(StmtNode node, JinjaScope scope)
        {
            var value = Eval(node.Expr!, scope);

            if (node.SetTargetAttr == null)
            {
                scope.SetLocal(node.SetTargetName!, value);
                return;
            }

            if (!scope.TryGet(node.SetTargetName!, out var target) || target is not Dictionary<string, object?> dict)
                throw new JinjaTemplateException($"'{node.SetTargetName}' no es un namespace válido para asignar '.{node.SetTargetAttr}'.");

            dict[node.SetTargetAttr] = value;
        }

        public static object? Eval(JinjaExpr expr, JinjaScope scope)
        {
            switch (expr.Kind)
            {
                case ExprKind.Literal:
                    return expr.LiteralValue;

                case ExprKind.Identifier:
                    if (scope.TryGet(expr.Name!, out var value))
                        return value;
                    return JinjaRuntime.TryGetBuiltinFunction(expr.Name!, out var fn) ? fn : JinjaUndefined.Instance;

                case ExprKind.Attr:
                    return JinjaRuntime.GetMember(Eval(expr.Target!, scope), expr.Name!);

                case ExprKind.Subscript:
                    return JinjaRuntime.GetIndex(Eval(expr.Target!, scope), Eval(expr.Left!, scope));

                case ExprKind.Slice:
                {
                    var target = Eval(expr.Target!, scope);
                    var start = expr.Left != null ? Eval(expr.Left, scope) : null;
                    var stop = expr.Right != null ? Eval(expr.Right, scope) : null;
                    return JinjaRuntime.GetSlice(target, start, stop);
                }

                case ExprKind.Call:
                    return JinjaRuntime.InvokeCall(expr, scope);

                case ExprKind.Filter:
                {
                    var target = Eval(expr.Target!, scope);
                    var args = EvalArgs(expr.CallArgs, scope);
                    return JinjaRuntime.ApplyFilter(expr.Name!, target, args);
                }

                case ExprKind.BinOp:
                    return JinjaRuntime.BinOp(expr.Op!, Eval(expr.Left!, scope), Eval(expr.Right!, scope));

                case ExprKind.Compare:
                    return JinjaRuntime.Compare(expr.Op!, Eval(expr.Left!, scope), Eval(expr.Right!, scope));

                case ExprKind.And:
                {
                    var l = Eval(expr.Left!, scope);
                    return JinjaRuntime.ToBool(l) ? Eval(expr.Right!, scope) : l;
                }

                case ExprKind.Or:
                {
                    var l = Eval(expr.Left!, scope);
                    return JinjaRuntime.ToBool(l) ? l : Eval(expr.Right!, scope);
                }

                case ExprKind.Not:
                    return !JinjaRuntime.ToBool(Eval(expr.Target!, scope));

                case ExprKind.Neg:
                    return JinjaRuntime.Negate(Eval(expr.Target!, scope));

                case ExprKind.Concat:
                    return JinjaRuntime.Stringify(Eval(expr.Left!, scope)) + JinjaRuntime.Stringify(Eval(expr.Right!, scope));

                case ExprKind.In:
                {
                    var l = Eval(expr.Left!, scope);
                    var r = Eval(expr.Right!, scope);
                    bool result = JinjaRuntime.Contains(r, l);
                    return expr.Negate ? !result : result;
                }

                case ExprKind.IsTest:
                {
                    var target = Eval(expr.Target!, scope);
                    var args = EvalArgs(expr.CallArgs, scope);
                    bool result = JinjaRuntime.EvalTest(expr.Name!, target, args);
                    return expr.Negate ? !result : result;
                }

                case ExprKind.Ternary:
                {
                    var cond = Eval(expr.Extra!, scope);
                    return JinjaRuntime.ToBool(cond) ? Eval(expr.Left!, scope) : Eval(expr.Right!, scope);
                }

                case ExprKind.ListLiteral:
                    return expr.Items!.Select(item => Eval(item, scope)).ToList();

                default:
                    throw new JinjaTemplateException($"Nodo de expresión Jinja no soportado: {expr.Kind}");
            }
        }

        private static List<(string? Name, object? Value)> EvalArgs(List<(string? Name, JinjaExpr Value)>? args, JinjaScope scope)
        {
            if (args == null || args.Count == 0)
                return new List<(string?, object?)>();

            return args.Select(a => (a.Name, Eval(a.Value, scope))).ToList();
        }
    }

    // =====================================================================
    // Runtime: operaciones sobre los valores dinámicos (object?) que produce
    // el evaluador -- acceso a miembros/índices, verdad/falsedad al estilo
    // Python, aritmética, comparaciones, filtros, tests y funciones builtin.
    // =====================================================================
    internal static class JinjaRuntime
    {
        public static object? GetMember(object? target, string name)
        {
            if (target is Dictionary<string, object?> dict)
                return dict.TryGetValue(name, out var v) ? v : JinjaUndefined.Instance;

            if (name is "length" or "count")
            {
                if (target is List<object?> list) return (long)list.Count;
                if (target is string s) return (long)s.Length;
            }

            return JinjaUndefined.Instance;
        }

        public static object? GetIndex(object? target, object? index)
        {
            if (target is List<object?> list)
            {
                long i = ToLong(index);
                if (i < 0) i += list.Count;
                if (i < 0 || i >= list.Count) return JinjaUndefined.Instance;
                return list[(int)i];
            }

            if (target is Dictionary<string, object?> dict)
            {
                var key = Stringify(index);
                return dict.TryGetValue(key, out var v) ? v : JinjaUndefined.Instance;
            }

            if (target is string s)
            {
                long i = ToLong(index);
                if (i < 0) i += s.Length;
                if (i < 0 || i >= s.Length) return JinjaUndefined.Instance;
                return s[(int)i].ToString();
            }

            return JinjaUndefined.Instance;
        }

        public static object? GetSlice(object? target, object? start, object? stop)
        {
            if (target is List<object?> list)
            {
                int count = list.Count;
                int s = NormalizeSliceIndex(start, count, 0);
                int e = NormalizeSliceIndex(stop, count, count);
                if (e < s) e = s;
                return list.GetRange(s, e - s);
            }

            if (target is string str)
            {
                int count = str.Length;
                int s = NormalizeSliceIndex(start, count, 0);
                int e = NormalizeSliceIndex(stop, count, count);
                if (e < s) e = s;
                return str.Substring(s, e - s);
            }

            return JinjaUndefined.Instance;
        }

        private static int NormalizeSliceIndex(object? value, int count, int defaultValue)
        {
            if (value == null || value is JinjaUndefined)
                return defaultValue;

            long i = ToLong(value);
            if (i < 0) i += count;
            if (i < 0) i = 0;
            if (i > count) i = count;
            return (int)i;
        }

        public static List<object?> ToIterable(object? value)
        {
            if (value is List<object?> list) return list;
            if (value is Dictionary<string, object?> dict) return dict.Keys.Select(k => (object?)k).ToList();
            if (value is string s) return s.Select(c => (object?)c.ToString()).ToList();
            return new List<object?>();
        }

        public static bool ToBool(object? value)
        {
            return value switch
            {
                null => false,
                JinjaUndefined => false,
                bool b => b,
                long l => l != 0,
                double d => d != 0,
                string s => s.Length > 0,
                List<object?> list => list.Count > 0,
                Dictionary<string, object?> dict => dict.Count > 0,
                _ => true
            };
        }

        public static string Stringify(object? value)
        {
            return value switch
            {
                null => string.Empty,
                JinjaUndefined => string.Empty,
                bool b => b ? "True" : "False",
                long l => l.ToString(CultureInfo.InvariantCulture),
                double d => d.ToString(CultureInfo.InvariantCulture),
                string s => s,
                List<object?> list => "[" + string.Join(", ", list.Select(Stringify)) + "]",
                Dictionary<string, object?> dict => "{" + string.Join(", ", dict.Select(kv => $"'{kv.Key}': {Stringify(kv.Value)}")) + "}",
                _ => value.ToString() ?? string.Empty
            };
        }

        private static long ToLong(object? value)
        {
            return value switch
            {
                long l => l,
                double d => (long)d,
                bool b => b ? 1 : 0,
                string s when long.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) => parsed,
                _ => 0
            };
        }

        private static double ToDouble(object? value)
        {
            return value switch
            {
                long l => l,
                double d => d,
                bool b => b ? 1 : 0,
                string s when double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) => parsed,
                _ => 0
            };
        }

        private static bool IsNumber(object? value) => value is long or double;
        private static bool IsBothInt(object? l, object? r) => l is long && r is long;

        public static object? BinOp(string op, object? l, object? r)
        {
            if (op == "+")
            {
                if (l is string ls && r is string rs) return ls + rs;
                if (l is List<object?> ll && r is List<object?> rl)
                {
                    var combined = new List<object?>(ll);
                    combined.AddRange(rl);
                    return combined;
                }
                return IsBothInt(l, r) ? (long)(ToLong(l) + ToLong(r)) : ToDouble(l) + ToDouble(r);
            }

            return op switch
            {
                "-" => IsBothInt(l, r) ? (object)(ToLong(l) - ToLong(r)) : ToDouble(l) - ToDouble(r),
                "*" => IsBothInt(l, r) ? (object)(ToLong(l) * ToLong(r)) : ToDouble(l) * ToDouble(r),
                "/" => ToDouble(l) / ToDouble(r),
                "//" => (object)(long)Math.Floor(ToDouble(l) / ToDouble(r)),
                "%" => IsBothInt(l, r) ? (object)(ToLong(l) % ToLong(r)) : ToDouble(l) % ToDouble(r),
                _ => throw new JinjaTemplateException($"Operador aritmético no soportado: '{op}'.")
            };
        }

        public static object Negate(object? value)
        {
            if (value is long l) return -l;
            return -ToDouble(value);
        }

        public static bool Compare(string op, object? l, object? r)
        {
            if (op == "==") return ValuesEqual(l, r);
            if (op == "!=") return !ValuesEqual(l, r);

            if (l is string ls && r is string rs)
            {
                int cmp = string.CompareOrdinal(ls, rs);
                return op switch
                {
                    "<" => cmp < 0,
                    "<=" => cmp <= 0,
                    ">" => cmp > 0,
                    ">=" => cmp >= 0,
                    _ => throw new JinjaTemplateException($"Operador de comparación no soportado: '{op}'.")
                };
            }

            double dl = ToDouble(l);
            double dr = ToDouble(r);
            return op switch
            {
                "<" => dl < dr,
                "<=" => dl <= dr,
                ">" => dl > dr,
                ">=" => dl >= dr,
                _ => throw new JinjaTemplateException($"Operador de comparación no soportado: '{op}'.")
            };
        }

        public static bool ValuesEqual(object? l, object? r)
        {
            if (l is JinjaUndefined) l = null;
            if (r is JinjaUndefined) r = null;

            if (l == null || r == null) return l == null && r == null;
            if (l is string ls && r is string rs) return ls == rs;
            if (IsNumber(l) && IsNumber(r)) return ToDouble(l) == ToDouble(r);
            if (l is bool lb && r is bool rb) return lb == rb;

            return Equals(l, r);
        }

        public static bool Contains(object? container, object? value)
        {
            if (container is List<object?> list) return list.Any(item => ValuesEqual(item, value));
            if (container is string s && value is string sub) return s.Contains(sub, StringComparison.Ordinal);
            if (container is Dictionary<string, object?> dict && value is string key) return dict.ContainsKey(key);
            return false;
        }

        public static bool EvalTest(string name, object? target, List<(string? Name, object? Value)> args)
        {
            return name switch
            {
                "defined" => target is not JinjaUndefined,
                "undefined" => target is JinjaUndefined,
                "none" => target is null,
                "string" => target is string,
                "number" => IsNumber(target),
                "boolean" => target is bool,
                "mapping" => target is Dictionary<string, object?>,
                "iterable" => target is List<object?> or string or Dictionary<string, object?>,
                "sequence" => target is List<object?> or string,
                "sameas" => args.Count > 0 && ReferenceEquals(target, args[0].Value),
                "equalto" or "eq" => args.Count > 0 && ValuesEqual(target, args[0].Value),
                "even" => IsNumber(target) && ToLong(target) % 2 == 0,
                "odd" => IsNumber(target) && ToLong(target) % 2 != 0,
                _ => false
            };
        }

        public static object? ApplyFilter(string name, object? target, List<(string? Name, object? Value)> args)
        {
            switch (name)
            {
                case "trim":
                case "strip":
                    return Stringify(target).Trim();
                case "upper":
                    return Stringify(target).ToUpperInvariant();
                case "lower":
                    return Stringify(target).ToLowerInvariant();
                case "capitalize":
                {
                    var s = Stringify(target);
                    return s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s.Substring(1).ToLowerInvariant();
                }
                case "title":
                    return CultureInfo.InvariantCulture.TextInfo.ToTitleCase(Stringify(target).ToLowerInvariant());
                case "length":
                case "count":
                    return (long)CountOf(target);
                case "default":
                case "d":
                {
                    bool treatFalsyAsDefault = args.Count > 1 && ToBool(args[1].Value);
                    bool isMissing = target is JinjaUndefined || (treatFalsyAsDefault && !ToBool(target));
                    return isMissing && args.Count > 0 ? args[0].Value : target;
                }
                case "join":
                {
                    var sep = args.Count > 0 ? Stringify(args[0].Value) : string.Empty;
                    return string.Join(sep, ToIterable(target).Select(Stringify));
                }
                case "first":
                {
                    var list = ToIterable(target);
                    return list.Count > 0 ? list[0] : JinjaUndefined.Instance;
                }
                case "last":
                {
                    var list = ToIterable(target);
                    return list.Count > 0 ? list[^1] : JinjaUndefined.Instance;
                }
                case "reverse":
                {
                    if (target is string s) return new string(s.Reverse().ToArray());
                    var copy = new List<object?>(ToIterable(target));
                    copy.Reverse();
                    return copy;
                }
                case "list":
                    return ToIterable(target);
                case "int":
                    return ToLong(target);
                case "float":
                    return ToDouble(target);
                case "string":
                    return Stringify(target);
                case "replace":
                    return args.Count < 2 ? target : Stringify(target).Replace(Stringify(args[0].Value), Stringify(args[1].Value));
                case "abs":
                    return target is long l ? Math.Abs(l) : (object)Math.Abs(ToDouble(target));
                case "truncate":
                {
                    var s = Stringify(target);
                    int len = args.Count > 0 ? (int)ToLong(args[0].Value) : 255;
                    return s.Length <= len ? s : s.Substring(0, len) + "...";
                }
                default:
                    throw new JinjaTemplateException($"Filtro Jinja no soportado: '{name}'.");
            }
        }

        private static int CountOf(object? value)
        {
            return value switch
            {
                string s => s.Length,
                List<object?> list => list.Count,
                Dictionary<string, object?> dict => dict.Count,
                _ => 0
            };
        }

        public static bool TryGetBuiltinFunction(string name, out object? function)
        {
            if (name is "range" or "namespace" or "strftime_now" or "raise_exception")
            {
                function = new JinjaBuiltinFunction(name);
                return true;
            }

            function = null;
            return false;
        }

        public static object? InvokeCall(JinjaExpr callExpr, JinjaScope scope)
        {
            var calleeExpr = callExpr.Target!;
            var args = callExpr.CallArgs?.Select(a => (a.Name, Value: JinjaEvaluator.Eval(a.Value, scope))).ToList()
                       ?? new List<(string? Name, object? Value)>();

            // Llamada a método: `expr.metodo(args)` (p.ej. `message['content'].strip()`).
            if (calleeExpr.Kind == ExprKind.Attr)
            {
                var target = JinjaEvaluator.Eval(calleeExpr.Target!, scope);
                return InvokeMethod(target, calleeExpr.Name!, args);
            }

            if (calleeExpr.Kind == ExprKind.Identifier)
            {
                var callee = JinjaEvaluator.Eval(calleeExpr, scope);
                if (callee is JinjaBuiltinFunction builtin)
                    return InvokeBuiltin(builtin.Name, args);
            }

            throw new JinjaTemplateException("Llamada a función/método Jinja no soportada.");
        }

        private static object? InvokeMethod(object? target, string method, List<(string? Name, object? Value)> args)
        {
            if (target is string s)
            {
                return method switch
                {
                    "strip" or "trim" => s.Trim(),
                    "lstrip" => s.TrimStart(),
                    "rstrip" => s.TrimEnd(),
                    "upper" => s.ToUpperInvariant(),
                    "lower" => s.ToLowerInvariant(),
                    "title" => CultureInfo.InvariantCulture.TextInfo.ToTitleCase(s.ToLowerInvariant()),
                    "startswith" => args.Count > 0 && s.StartsWith(Stringify(args[0].Value), StringComparison.Ordinal),
                    "endswith" => args.Count > 0 && s.EndsWith(Stringify(args[0].Value), StringComparison.Ordinal),
                    "replace" => args.Count >= 2 ? s.Replace(Stringify(args[0].Value), Stringify(args[1].Value)) : s,
                    _ => throw new JinjaTemplateException($"Método de string Jinja no soportado: '{method}'.")
                };
            }

            if (target is Dictionary<string, object?> dict)
            {
                switch (method)
                {
                    case "items":
                        return dict.Select(kv => (object?)new List<object?> { kv.Key, kv.Value }).ToList();
                    case "keys":
                        return dict.Keys.Select(k => (object?)k).ToList();
                    case "values":
                        return dict.Values.ToList();
                    case "get":
                    {
                        var key = args.Count > 0 ? Stringify(args[0].Value) : string.Empty;
                        if (dict.TryGetValue(key, out var v)) return v;
                        return args.Count > 1 ? args[1].Value : JinjaUndefined.Instance;
                    }
                    default:
                        throw new JinjaTemplateException($"Método de dict Jinja no soportado: '{method}'.");
                }
            }

            if (target is List<object?> list)
            {
                if (method == "append")
                {
                    list.Add(args.Count > 0 ? args[0].Value : null);
                    return null;
                }
                throw new JinjaTemplateException($"Método de lista Jinja no soportado: '{method}'.");
            }

            throw new JinjaTemplateException($"No se puede invocar el método '{method}' sobre este valor.");
        }

        private static object? InvokeBuiltin(string name, List<(string? Name, object? Value)> args)
        {
            switch (name)
            {
                case "range":
                {
                    long start = 0, stop, step = 1;
                    if (args.Count == 1) { stop = ToLong(args[0].Value); }
                    else if (args.Count == 2) { start = ToLong(args[0].Value); stop = ToLong(args[1].Value); }
                    else if (args.Count >= 3) { start = ToLong(args[0].Value); stop = ToLong(args[1].Value); step = ToLong(args[2].Value); }
                    else { stop = 0; }

                    var result = new List<object?>();
                    if (step == 0) return result;
                    if (step > 0)
                        for (long v = start; v < stop; v += step) result.Add(v);
                    else
                        for (long v = start; v > stop; v += step) result.Add(v);
                    return result;
                }

                case "namespace":
                {
                    var dict = new Dictionary<string, object?>();
                    foreach (var (argName, value) in args)
                    {
                        if (argName != null)
                            dict[argName] = value;
                    }
                    return dict;
                }

                case "strftime_now":
                {
                    var format = args.Count > 0 ? Stringify(args[0].Value) : "%Y-%m-%d";
                    return FormatStrftime(DateTime.UtcNow, format);
                }

                case "raise_exception":
                {
                    var message = args.Count > 0 ? Stringify(args[0].Value) : "raise_exception() sin mensaje";
                    throw new JinjaTemplateException("El chat_template pidió abortar: " + message);
                }

                default:
                    throw new JinjaTemplateException($"Función Jinja no soportada: '{name}'.");
            }
        }

        private static string FormatStrftime(DateTime now, string format)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < format.Length; i++)
            {
                if (format[i] == '%' && i + 1 < format.Length)
                {
                    char code = format[i + 1];
                    sb.Append(code switch
                    {
                        'Y' => now.Year.ToString("D4"),
                        'm' => now.Month.ToString("D2"),
                        'd' => now.Day.ToString("D2"),
                        'H' => now.Hour.ToString("D2"),
                        'M' => now.Minute.ToString("D2"),
                        'S' => now.Second.ToString("D2"),
                        'B' => now.ToString("MMMM", CultureInfo.InvariantCulture),
                        'b' => now.ToString("MMM", CultureInfo.InvariantCulture),
                        'A' => now.ToString("dddd", CultureInfo.InvariantCulture),
                        'a' => now.ToString("ddd", CultureInfo.InvariantCulture),
                        '%' => "%",
                        _ => "%" + code
                    });
                    i++;
                }
                else
                {
                    sb.Append(format[i]);
                }
            }
            return sb.ToString();
        }
    }
}
