using System;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class SnesBridgeConnection : IDisposable
    {
        private const string ResetCommand = "RESET";
        private const string StopCommand = "STOP";

        private readonly TcpListener _listener;
        private readonly byte[] _buffer = new byte[1 << 16];
        private readonly StringBuilder _pending = new StringBuilder();
        private TcpClient? _client;
        private NetworkStream? _stream;

        public SnesBridgeConnection(IPAddress address, int port)
        {
            _listener = new TcpListener(address, port);
        }

        public void WaitForBizHawk()
        {
            _listener.Start();
            AcceptNext();
        }

        public SnesState ReceiveState()
        {
            if (_stream == null)
            {
                AcceptNext();
            }

            return SnesState.Parse(ReadLine());
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
                    throw new InvalidOperationException(
                        $"No llego ningun dato de BizHawk en {ReceiveTimeoutMs / 1000} segundos. " +
                        "Lo mas probable es que el script Lua se haya quedado trabado esperando algo " +
                        "(un cartel de dialogo que no se cierra, el emulador pausado, o perdio el foco). " +
                        "El ultimo checkpoint guardado antes de esto sigue intacto.",
                        ex);
                }

                if (bytesRead == 0)
                {
                    throw new InvalidOperationException("BizHawk cerro la conexion.");
                }

                _pending.Append(Encoding.ASCII.GetString(_buffer, 0, bytesRead));
            }
        }

        private void SendMessage(string message)
        {
            var stream = _stream ?? throw new InvalidOperationException("La conexion con BizHawk no esta lista.");
            var payload = Encoding.ASCII.GetBytes($"{message.Length} {message}");
            stream.Write(payload, 0, payload.Length);
        }

        public void Dispose()
        {
            SendStop();
            _stream?.Dispose();
            _client?.Dispose();
            _listener.Stop();
        }
    }
}
