using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace Neuraval.Evolution.MarioBridge
{
    // Excepcion dedicada para "se rompio la conexion con BizHawk" (timeout,
    // socket cerrado, error de red al escribir). Separarla de
    // InvalidOperationException permite que quien llama (Program.cs) sepa
    // con certeza cuando vale la pena reconectar y reintentar, en vez de
    // tener que adivinar por el mensaje de texto de una excepcion generica.
    public sealed class SnesBridgeConnectionLostException : Exception
    {
        public SnesBridgeConnectionLostException(string message, Exception? inner = null)
            : base(message, inner)
        {
        }
    }

    public sealed class SnesBridgeConnection : IDisposable
    {
        private const string ResetCommand = "RESET";
        private const string StopCommand = "STOP";
        private const string CaptureCommand = "CAPTURE";
        private const string TurboCommand = "TURBO";

        private readonly TcpListener _listener;
        private readonly byte[] _buffer = new byte[1 << 16];
        private readonly StringBuilder _pending = new StringBuilder();
        private TcpClient? _client;
        private NetworkStream? _stream;

        public SnesBridgeConnection(IPAddress address, int port)
        {
            _listener = new TcpListener(address, port);
        }

        // shouldAbort se consulta periodicamente mientras se espera la
        // PRIMERA conexion, igual que en Reconnect(). Sin esto,
        // AcceptTcpClient() bloquea sin forma de cancelarlo: Ctrl+C sigue
        // disparando el CancelKeyPress y seteando stopRequested, pero el
        // proceso nunca llega a leerlo porque esta trabado en el accept, asi
        // que el mensaje de "deteniendo..." se repite en cada intento sin
        // que el programa realmente termine.
        public void WaitForBizHawk(Func<bool>? shouldAbort = null)
        {
            _listener.Start();

            while (!_listener.Pending())
            {
                if (shouldAbort != null && shouldAbort())
                {
                    throw new OperationCanceledException("Se cancelo la espera de conexion con BizHawk.");
                }

                Thread.Sleep(250);
            }

            AcceptNext();
        }

        // Se llama cuando ReceiveState()/SendXxx() detecto que se perdio la
        // conexion (BizHawk/Lua se reinicio, se cerro, o se colgo mas de
        // ReceiveTimeoutMs). Descarta el socket viejo y se queda esperando
        // que un NUEVO cliente (el script de Lua reiniciado) se conecte al
        // mismo listener, que nunca dejo de escuchar. No hace falta volver
        // a levantar el proceso de entrenamiento entero: la generacion que
        // se estaba corriendo se puede reintentar apenas vuelva a haber
        // conexion.
        //
        // shouldAbort se consulta periodicamente mientras se espera, para
        // poder salir limpio si el usuario pidio detener el entrenamiento
        // (Ctrl+C) mientras BizHawk seguia desconectado.
        public void Reconnect(Func<bool>? shouldAbort = null)
        {
            _stream?.Dispose();
            _client?.Dispose();
            _stream = null;
            _client = null;
            _pending.Clear();

            Console.WriteLine("MarioBridge: conexion con BizHawk perdida. Esperando reconexion (reinicia el script de Lua)...");
            var lastNotice = DateTime.UtcNow;

            while (!_listener.Pending())
            {
                if (shouldAbort != null && shouldAbort())
                {
                    throw new OperationCanceledException("Se cancelo la espera de reconexion con BizHawk.");
                }

                if (DateTime.UtcNow - lastNotice > TimeSpan.FromSeconds(15))
                {
                    Console.WriteLine("MarioBridge: sigo esperando que BizHawk se reconecte en " + PortForLog() + "...");
                    lastNotice = DateTime.UtcNow;
                }

                Thread.Sleep(250);
            }

            AcceptNext();
            Console.WriteLine("MarioBridge: BizHawk reconectado.");
        }

        private string PortForLog()
        {
            return _listener.LocalEndpoint?.ToString() ?? "el puerto configurado";
        }

        public SnesState ReceiveState()
        {
            if (_stream == null)
            {
                AcceptNext();
            }

            var state = SnesState.Parse(ReadLine());

            // Efecto secundario de solo lectura: no cambia el estado ni la
            // logica de fitness/encoding, solo deja constancia en disco de
            // cualquier sprite que el ROM real muestre y que todavia no
            // tenga nombre en MarioSpriteNames.
            MarioUnknownSpriteLogger.Track(state);

            return state;
        }

        public void SendAction(SnesAction action)
        {
            SendMessage(action.ToWireFormat());
        }

        // levelIndex es opcional para no romper a nadie que siga llamando
        // SendReset() sin argumentos: sin indice, Lua recarga el nivel que
        // ya tenia cargado (ver applyAction en mario_bridge.lua). Con
        // indice, le pide a Lua que cambie de savestate antes de recargar.
        public void SendReset(int? levelIndex = null)
        {
            SendMessage(levelIndex.HasValue ? $"{ResetCommand}:{levelIndex.Value}" : ResetCommand);
        }

        public void SendCapture(int levelIndex)
        {
            SendMessage($"{CaptureCommand}:{levelIndex}");
        }

        // Activa/desactiva velocidad maxima en BizHawk (client.speedmode).
        // Pensado para entrenamiento headless por NEAT, donde no hace falta
        // ver el juego en tiempo real; no usar durante --capture, porque ahi
        // se necesita velocidad normal para poder jugar a mano.
        public void SendTurbo(bool enabled)
        {
            SendMessage($"{TurboCommand}:{(enabled ? 1 : 0)}");
        }

        public void SendStop()
        {
            if (_stream != null)
            {
                SendMessage(StopCommand);
            }
        }

        // Si BizHawk deja de mandar datos por cualquier motivo (se pausa, el
        // Lua entra en un loop bloqueante que no vuelve a tocar el socket,
        // se pierde el foco, etc.), sin timeout el stream.Read() de abajo se
        // queda esperando para siempre: el proceso sigue "vivo" pero nunca
        // vuelve a guardar un checkpoint, y no aparece ningun error que lo
        // explique. Con esto, despues de 60s sin datos se tira una
        // excepcion clara en vez de colgarse en silencio.
        private const int ReceiveTimeoutMs = 60_000;

        private void AcceptNext()
        {
            _client = _listener.AcceptTcpClient();
            _stream = _client.GetStream();
            _stream.ReadTimeout = ReceiveTimeoutMs;
            _pending.Clear();
        }

        private string ReadLine()
        {
            var stream = _stream ?? throw new InvalidOperationException("La conexion con BizHawk no esta lista.");

            while (true)
            {
                var text = _pending.ToString();
                var spaceIndex = text.IndexOf(' ');

                if (spaceIndex > 0 && int.TryParse(text.Substring(0, spaceIndex), out var length))
                {
                    var totalNeeded = spaceIndex + 1 + length;

                    if (text.Length >= totalNeeded)
                    {
                        var payload = text.Substring(spaceIndex + 1, length).Trim();
                        _pending.Clear();
                        _pending.Append(text.Substring(totalNeeded));
                        return payload;
                    }
                }

                int bytesRead;
                try
                {
                    bytesRead = stream.Read(_buffer, 0, _buffer.Length);
                }
                catch (System.IO.IOException ex)
                {
                    throw new SnesBridgeConnectionLostException(
                        $"No llego ningun dato de BizHawk en {ReceiveTimeoutMs / 1000} segundos. " +
                        "Lo mas probable es que el script Lua se haya quedado trabado esperando algo " +
                        "(un cartel de dialogo que no se cierra, el emulador pausado, o perdio el foco), " +
                        "o que se haya reiniciado el script/emulador. " +
                        "El ultimo checkpoint guardado antes de esto sigue intacto.",
                        ex);
                }
                catch (ObjectDisposedException ex)
                {
                    throw new SnesBridgeConnectionLostException(
                        "El socket con BizHawk ya estaba cerrado al intentar leer.", ex);
                }

                if (bytesRead == 0)
                {
                    throw new SnesBridgeConnectionLostException(
                        "BizHawk cerro la conexion (probablemente se reinicio el script o el emulador).");
                }

                _pending.Append(Encoding.ASCII.GetString(_buffer, 0, bytesRead));
            }
        }

        private void SendMessage(string message)
        {
            var stream = _stream ?? throw new InvalidOperationException("La conexion con BizHawk no esta lista.");
            var payload = Encoding.ASCII.GetBytes($"{message.Length} {message}");

            try
            {
                stream.Write(payload, 0, payload.Length);
            }
            catch (System.IO.IOException ex)
            {
                throw new SnesBridgeConnectionLostException(
                    "Se perdio la conexion con BizHawk al enviar un comando (probablemente se reinicio el script o el emulador).", ex);
            }
            catch (ObjectDisposedException ex)
            {
                throw new SnesBridgeConnectionLostException(
                    "El socket con BizHawk ya estaba cerrado al intentar enviar un comando.", ex);
            }
        }

        public void Dispose()
        {
            // Best-effort: si la conexion ya esta rota no tiene sentido que
            // Dispose() explote por eso, solo queremos liberar los recursos.
            try
            {
                SendStop();
            }
            catch (SnesBridgeConnectionLostException)
            {
                // Ignorado a proposito: la conexion ya estaba perdida.
            }

            _stream?.Dispose();
            _client?.Dispose();
            _listener.Stop();
        }
    }
}
