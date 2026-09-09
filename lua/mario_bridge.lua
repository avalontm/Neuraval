-- Detecta la carpeta donde vive ESTE script (mario_bridge.lua) usando
-- debug.getinfo, en vez de depender de "./" (que BizHawk resuelve relativo
-- a su propio directorio de trabajo, casi nunca el del proyecto Neuraval)
-- o de que el usuario configure una variable de entorno a mano. Por
-- default los savestates (DP1.state, etc.) van a vivir al lado de
-- mario_bridge.lua, en lua/ -- una ubicacion fija y predecible sin
-- hardcodear ninguna ruta absoluta.
--
-- debug.getinfo(1, "S").source devuelve algo como "@D:\ruta\lua\mario_bridge.lua"
-- (el "@" indica que vino de un archivo, no de un string cargado en memoria)
-- SOLO si BizHawk abrio el script con una ruta absoluta. Si se abrio con una
-- ruta relativa (accesos directos viejos, entradas "recientes" de un .luases
-- movido, etc.), source tambien va a ser relativo, y la carpeta que sacamos
-- de ahi hereda ese problema -- ver isAbsolutePath()/resolveSavestateDir()
-- mas abajo para el chequeo correspondiente.
local function isAbsolutePath(path)
    return path:match("^%a:[/\\]") ~= nil  -- Windows: "C:\..." o "C:/..."
        or path:match("^[/\\][/\\]") ~= nil  -- UNC: "\\\\servidor\\recurso"
        or path:match("^/") ~= nil  -- Unix/Mac
end

local function scriptDirectory()
    local info = debug.getinfo(1, "S")
    local source = info and info.source or nil
    console.log("MarioBridge: debug.getinfo().source = " .. tostring(source))
    if source == nil or source:sub(1, 1) ~= "@" then
        return nil
    end

    local path = source:sub(2)
    -- Se banca separadores / y \\ (BizHawk corre en Windows/Linux/Mac).
    return path:match("^(.*[/\\])")
end

-- NEURAVAL_SAVESTATE_DIR sigue funcionando como override explicito para
-- quien prefiera guardar los savestates en otro lado; si no esta definida,
-- se usa la carpeta del script. Si ninguno de los dos resuelve (caso raro:
-- version de BizHawk sin soporte de debug.getinfo con "source") se cae a
-- "./" como ultimo recurso. En los tres casos se valida que la carpeta
-- resultante sea una ruta ABSOLUTA -- ese es el motivo mas comun por el
-- que DP1.state "no se encuentra" a pesar de existir en disco.
local function resolveSavestateDir()
    local envDir = os.getenv("NEURAVAL_SAVESTATE_DIR")
    local dir, dirSource

    if envDir ~= nil and envDir ~= "" then
        dir, dirSource = envDir, "NEURAVAL_SAVESTATE_DIR"
    else
        local ok, scriptDir = pcall(scriptDirectory)
        if ok and scriptDir ~= nil then
            dir, dirSource = scriptDir, "la carpeta de este script (debug.getinfo)"
        end
    end

    if dir == nil then
        console.log("MarioBridge: no se pudo detectar automaticamente la carpeta de este script " ..
            "(y NEURAVAL_SAVESTATE_DIR no esta definida); usando el directorio de trabajo actual " ..
            "(\"./\") como ultimo recurso, que puede no coincidir con donde arranco BizHawk. " ..
            "Defini NEURAVAL_SAVESTATE_DIR a mano si esto pasa.")
        return "./"
    end

    if not isAbsolutePath(dir) then
        console.log("MarioBridge: ATENCION - la carpeta resuelta via " .. dirSource .. " (\"" .. dir ..
            "\") es RELATIVA, no absoluta. BizHawk la va a interpretar contra su propio directorio de " ..
            "trabajo actual (normalmente la carpeta de EmuHawk.exe), casi nunca la carpeta real de " ..
            "mario_bridge.lua -- esta es la causa tipica de que DP1.state \"no se encuentre\" aunque el " ..
            "archivo exista donde deberia. Solucion: en el Lua Console usa \"Script > Open Script...\" y " ..
            "elegi el archivo desde el dialogo (eso siempre carga con ruta absoluta) en vez de un acceso " ..
            "directo/entrada reciente con ruta relativa, o definí NEURAVAL_SAVESTATE_DIR con la ruta " ..
            "absoluta completa a la carpeta que tiene DP1.state.")
    end

    return dir
end

local SAVESTATE_DIR = resolveSavestateDir()
if not SAVESTATE_DIR:match("[/\\]$") then
    SAVESTATE_DIR = SAVESTATE_DIR .. "/"
end

local SAVESTATE_FILES = {
    [0] = SAVESTATE_DIR .. "DP1.state",
}

-- Chequeo de arranque: si algun .state configurado arriba no existe en
-- disco, savestate.load() falla en silencio cada vez que se intenta
-- resetear (BizHawk solo tira un "could not find file" suelto por consola,
-- sin contexto), y el nivel queda trabado reintentando el load para
-- siempre sin que quede claro por que. Esto avisa UNA vez, apenas carga el
-- script, con la ruta completa que se va a intentar usar, para diagnosticar
-- esto antes de perder tiempo jugando.
for levelIndex, path in pairs(SAVESTATE_FILES) do
    local file = io.open(path, "rb")
    if file == nil then
        console.log("MarioBridge: ATENCION - no se encuentra el savestate del nivel " .. tostring(levelIndex) ..
            " en \"" .. path .. "\" (carpeta resuelta: " .. SAVESTATE_DIR ..
            (os.getenv("NEURAVAL_SAVESTATE_DIR") ~= nil and " -- via NEURAVAL_SAVESTATE_DIR" or " -- carpeta de este script") ..
            "). Los resets a este nivel van a fallar en silencio hasta que crees ese archivo ahi. Ver README, seccion \"Configuracion de savestates\".")
    else
        file:close()
    end
end

local currentLevelIndex = 0
local captureMode = false

local RESET_COMMAND = "RESET"
local STOP_COMMAND = "STOP"
local CAPTURE_COMMAND = "CAPTURE"
local TURBO_COMMAND = "TURBO"

-- 6400 es un valor deliberadamente alto: distintas versiones de BizHawk
-- soportan distintos topes internos para client.speedmode (el menu grafico
-- limita a 400%, pero la API de Lua acepta valores mas altos y BizHawk los
-- recorta el solo al maximo real que soporte esa version). Pedir de mas no
-- rompe nada, solo se satura al techo disponible.
local TURBO_SPEED_PERCENT = 6400
local NORMAL_SPEED_PERCENT = 100
local turboEnabled = false

-- Flags del overlay de "vision" declarados aca arriba (junto con
-- turboEnabled) para que setTurbo() pueda resetear el aviso de
-- "no se dibuja por turbo" cuando el turbo se apaga y se vuelve a
-- prender mas adelante en la sesion.
local visionOverlayEnabled = true
local visionToggleKeyWasDown = false
local visionDrawWarned = false
local visionClearWarned = false

-- Envuelto en pcall porque el nombre exacto de estas funciones de la API de
-- Lua de BizHawk cambio entre versiones. Si alguna no existe en tu build,
-- esto lo avisa por consola en vez de tirar abajo todo el script.
local function setTurbo(enabled)
    turboEnabled = enabled

    local speedOk, speedErr = pcall(function()
        client.speedmode(enabled and TURBO_SPEED_PERCENT or NORMAL_SPEED_PERCENT)
    end)
    if not speedOk then
        console.log("MarioBridge: no se pudo cambiar la velocidad con client.speedmode (" .. tostring(speedErr) .. ").")
    end

    -- El audio suele ser el cuello de botella real para llegar a velocidad
    -- maxima (ver documentacion de BizHawk sobre turbo). Si el metodo no
    -- existe en esta version, se ignora sin romper el script.
    if client.SetSoundOn ~= nil then
        pcall(function() client.SetSoundOn(not enabled) end)
    end

    console.log("MarioBridge: turbo " .. (enabled and "activado" or "desactivado") .. ".")
end

-- Si el script se detiene por cualquier motivo (STOP, cerrar BizHawk,
-- recargar el script), siempre se vuelve a velocidad normal. Sin esto,
-- BizHawk podria quedar en modo turbo despues de que el entrenamiento
-- termine, lo cual es confuso si despues alguien quiere jugar a mano.
event.onexit(function() setTurbo(false) end)

-- Radio 8 = grid de 17x17 tiles (272x272 px) centrado en Mario. La
-- pantalla de SMW mide 256x224 px (16x14 tiles), asi que este radio
-- cubre el ancho completo justo (128px de cada lado) y se pasa un
-- poco de alto a proposito, para garantizar que Mario "vea" toda la
-- pantalla sin importar donde este parado dentro del scroll de camara.
-- DEBE coincidir exactamente con SnesState.GridRadius en el lado C#
-- (Neuraval.Evolution.MarioBridge/SnesState.cs): cambia el tamano de
-- la capa de entrada de la red, asi que un desajuste rompe el
-- protocolo o desalinea los tiles silenciosamente.
local GRID_RADIUS = 8
local BUTTON_NAMES = { "A", "B", "X", "Y", "Up", "Down", "Left", "Right", "L", "R", "Select", "Start" }
local MESSAGE_BOX_ADDR = 0x1426
local MESSAGE_BOX_HOLD_FRAMES = 4
local MESSAGE_BOX_RELEASE_FRAMES = 4
local MAX_DISMISS_ATTEMPTS = 90
local LEVEL_END_ADDR = 0x1493
local MANUAL_RESET_KEY = "Insert"
local manualResetKeyWasDown = false

-- Ventana de gracia tras cualquier savestate.load(): la RAM restaurada puede
-- traer "congelados" los flags de muerte (0x71) o fin de nivel (0x1493) si el
-- .state se guardo en un frame donde esos flags todavia no habian sido
-- limpiados por el juego (por ejemplo, un savestate tomado un instante antes
-- de que el motor reinicie esos bytes). Sin esta ventana, buildState() reporta
-- inmediatamente "muerto"/"nivel completo" en el primer frame post-reset, el
-- bridge en C# manda otro RESET, y el nivel se reinicia en bucle sin que el
-- agente llegue a jugar ni un frame -- exactamente el sintoma de "se reinicia
-- muy rapido" reportado.
local RESET_GRACE_FRAMES = 10
local resetGraceFramesRemaining = 0

local function beginResetGrace()
    resetGraceFramesRemaining = RESET_GRACE_FRAMES
end

local function currentSavestateFile()
    local path = SAVESTATE_FILES[currentLevelIndex]
    if path == nil then
        return SAVESTATE_FILES[0]
    end
    return path
end

local function readS8(address)
    local value = memory.readbyte(address)
    if value > 0x7F then
        value = value - 0x100
    end
    return value
end

local function marioPosition()
    return memory.read_s16_le(0x94), memory.read_s16_le(0x96)
end

local function marioSubSpeed()
    return memory.readbyte(0x7A)
end

local function marioVelocity()
    return readS8(0x7B), readS8(0x7D)
end

local function isMarioDead()
    return memory.readbyte(0x71) == 0x09
end

local function livesRemaining()
    return memory.readbyte(0x0DBE) + 1
end

local function marioDirection()
    return memory.readbyte(0x76)
end

local function marioBlocked()
    return memory.readbyte(0x77)
end

local function marioSubpixel()
    return memory.readbyte(0x13DA), memory.readbyte(0x13DC)
end

local function marioAirState()
    return memory.readbyte(0x72)
end

local function marioDucking()
    return memory.readbyte(0x73)
end

local function marioClimbing()
    return memory.readbyte(0x74)
end

local function marioWater()
    return memory.readbyte(0x75)
end

local function marioPMeter()
    return memory.readbyte(0x13E4)
end

local function marioTakeoff()
    return memory.readbyte(0x149F)
end

local function marioHurt()
    return memory.readbyte(0x1496)
end

local function marioCape()
    return memory.readbyte(0x14A5)
end

local function marioPowTimers()
    return memory.readbyte(0x14AD), memory.readbyte(0x14AE)
end

local function doorExitCounter()
    return memory.readbyte(0x141A)
end

local function itemMemory()
    return memory.readbyte(0x13BE)
end

local function currentPlayerState()
    return memory.readbyte(0x0DA0), memory.readbyte(0x0DB3), memory.readbyte(0x0DBF)
end

local function marioPowerup()
    return memory.readbyte(0x19)
end

local function cameraPosition()
    return memory.read_s16_le(0x1462), memory.read_s16_le(0x1464)
end

local function gameState()
    return memory.readbyte(0x0100), memory.readbyte(0x0D9B), memory.readbyte(0x13BF)
end

local function marioCollectibles()
    return memory.readbyte(0x0DB6), memory.readbyte(0x0DBC)
end

local function marioReservedItemBox()
    return memory.readbyte(0x0DC2)
end

local function marioCarryFlags()
    return memory.readbyte(0x1470), memory.readbyte(0x148F)
end

local function midwayCheckpointFlags()
    return memory.readbyte(0x13CE), memory.readbyte(0x13CD)
end

local function yoshiCoinsCollected()
    return memory.readbyte(0x1420)
end

local function marioControllerCopies()
    return memory.readbyte(0x0DA2), memory.readbyte(0x0DA4)
end

local function secondPlayerControllers()
    return memory.readbyte(0x0DA3), memory.readbyte(0x0DA5), memory.readbyte(0x0DA7), memory.readbyte(0x0DA9)
end

local function controllerState()
    return memory.readbyte(0x0015), memory.readbyte(0x0016), memory.readbyte(0x0017), memory.readbyte(0x0018)
end

local function frameCounters()
    return memory.readbyte(0x13), memory.readbyte(0x14), memory.readbyte(0x1F2)
end

local function isMessageBoxActive()
    return memory.readbyte(MESSAGE_BOX_ADDR) ~= 0
end

local function isLevelComplete()
    return memory.readbyte(LEVEL_END_ADDR) ~= 0
end

-- Version "efectiva" de isMarioDead/isLevelComplete que usa buildState() para
-- decidir que le reporta a C#: durante la ventana de gracia post-reset, se
-- fuerza a false aunque la memoria diga lo contrario. isMarioDead()/
-- isLevelComplete() crudas se dejan intactas para quien las necesite sin
-- filtrar (no se usan en otro lado hoy, pero evita romper el contrato).
local loggedGraceSuppression = false

local function isMarioDeadEffective()
    if resetGraceFramesRemaining > 0 then
        if isMarioDead() and not loggedGraceSuppression then
            loggedGraceSuppression = true
            console.log("MarioBridge: flag de muerte activo justo tras un savestate.load(); suprimido por ventana de gracia (" .. resetGraceFramesRemaining .. " frames restantes).")
        end
        return false
    end
    return isMarioDead()
end

local function isLevelCompleteEffective()
    if resetGraceFramesRemaining > 0 then
        if isLevelComplete() and not loggedGraceSuppression then
            loggedGraceSuppression = true
            console.log("MarioBridge: flag de nivel completo activo justo tras un savestate.load(); suprimido por ventana de gracia (" .. resetGraceFramesRemaining .. " frames restantes).")
        end
        return false
    end
    return isLevelComplete()
end

local function releaseAllButtons()
    local controller = {}
    for _, name in ipairs(BUTTON_NAMES) do
        controller["P1 " .. name] = false
    end
    joypad.set(controller)
end

local function manualResetPulse()
    local keys = input.get()
    local isDown = keys[MANUAL_RESET_KEY] == true
    local pulse = isDown and not manualResetKeyWasDown
    manualResetKeyWasDown = isDown
    return pulse
end

local function dismissMessageBox()
    local attempts = 0
    while isMessageBoxActive() do
        attempts = attempts + 1
        if attempts > MAX_DISMISS_ATTEMPTS then
            console.log("MarioBridge: cartel de dialogo no se cerro; forzando reset por savestate.")
            releaseAllButtons()
            savestate.load(currentSavestateFile())
            beginResetGrace()
            return true
        end

        local controller = {}
        for _, name in ipairs(BUTTON_NAMES) do
            controller["P1 " .. name] = false
        end
        controller["P1 Start"] = true
        controller["P1 A"] = true
        joypad.set(controller)
        for _ = 1, MESSAGE_BOX_HOLD_FRAMES do
            emu.frameadvance()
        end

        releaseAllButtons()
        for _ = 1, MESSAGE_BOX_RELEASE_FRAMES do
            emu.frameadvance()
        end
    end

    return false
end

-- Map16 Low Byte Table: $7E:C800 (offset 0xC800 en el dominio WRAM de BizHawk).
-- Map16 High Byte Table: $7F:C800 (offset 0x1C800 en el dominio WRAM: banco $7F
-- empieza en 0x10000, entonces 0x10000 + 0xC800 = 0x1C800). Antes esto leia
-- 0x1C800/0x1D800 (offset +0x1000 entre ambas), que caia dos veces dentro de
-- la tabla de high byte y nunca tocaba la de low byte real -- por eso ningun
-- tile (moneda, bloque de moneda, solido) se identificaba bien en el overlay,
-- mientras que sprites/Mario si dibujaban porque usan direcciones de banco $7E
-- que ya eran correctas.
local MAP16_LOW_BYTE_TABLE = 0xC800
local MAP16_HIGH_BYTE_TABLE = 0x1C800

local function getTileFull(marioX, marioY, dx, dy)
    local x = math.floor((marioX + dx) / 16)
    local y = math.floor((marioY + dy) / 16)
    local idx = math.floor(x / 0x10) * 0x1B0 + y * 0x10 + x % 0x10
    local lo = memory.readbyte(MAP16_LOW_BYTE_TABLE + idx)
    local hi = memory.readbyte(MAP16_HIGH_BYTE_TABLE + idx)
    return hi * 256 + lo
end

local function getTile(marioX, marioY, dx, dy)
    return getTileFull(marioX, marioY, dx, dy) % 256
end

-- OJO: estos IDs de Map16 (byte bajo) son especificos del tileset/nivel
-- cargado, no un estandar universal de SMW -- hay que verificarlos contra
-- el ROM real, no asumirlos. 0x25 se saco de esta lista porque se
-- confirmo visualmente que en DP1 corresponde a terreno solido comun
-- (colina de pasto), no a una moneda: se pintaba amarillo sobre toda una
-- ladera sin monedas visibles, lo que ademas contaminaba buildCoinSignals()
-- con una "moneda" falsa pegada al agente, empujandolo a saltar contra la
-- pared en vez de avanzar. 0x2B, 0x5B, 0x6B quedan porque no mostraron ese
-- problema, pero conviene reconfirmarlos igual: pararse al lado de una
-- moneda real en pantalla y leer el numero hex que dibuja el overlay sobre
-- ese tile (la etiqueta "ID Map16 crudo" que ya trae la leyenda).
-- Confirmado contra la tabla oficial $7E009C ("Generated Map16 tile") del RAM
-- map de smwcentral (bin.smwcentral.net/u/1686/ram.txt):
--   01/02 -> tile 0x25 = "empty" (vacio) -- por eso se saco antes de esta lista.
--   06    -> tile 0x2B = "coin"          -- confirmado, se mantiene.
--   0A    -> tile 0x11B = "multiple coin turnblock" (low byte 0x1B)
--   0B    -> tile 0x123 = "multiple coin q block"   (low byte 0x23)
-- 0x5B y 0x6B no aparecen en ninguna tabla oficial y no hay motivo tecnico
-- para que existan: en SMW la animacion de la moneda es a nivel de VRAM
-- (ExAnimation) sobre UN SOLO Map16 ID, no cambia de ID por frame. Se sacan
-- hasta poder confirmarlos empiricamente (pararse al lado de una moneda real
-- con el overlay prendido y leer el ID que muestra).
local COIN_TILE_LOW_BYTES = { [0x2B] = true }
-- Confirmados por la tabla $7E009C: 0x1B = "multiple coin turnblock" (de 0x11B),
-- 0x23 = "multiple coin q block" (de 0x123).
local COIN_BLOCK_LOW_BYTES = { [0x1B] = true, [0x23] = true }
local DIALOG_TILE_FULL = { [0x0104] = true, [0x0105] = true, [0x0106] = true, [0x0107] = true }
local PIPE_ENTRANCE_TILE_FULL = { [0x0137] = true, [0x0138] = true }

local function buildTileGrid(marioX, marioY)
    local tiles = {}
    for dy = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
        for dx = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
            local tile = getTile(marioX, marioY, dx, dy)
            tiles[#tiles + 1] = tostring(tile)
        end
    end
    return table.concat(tiles, ",")
end

local function buildCoinSignals(marioX, marioY)
    local coinCount = 0
    local coinDx, coinDy, coinBestSq = 0, 0, nil
    local blockCount = 0
    local blockDx, blockDy, blockBestSq = 0, 0, nil
    for dy = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
        for dx = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
            local tile = getTile(marioX, marioY, dx, dy)
            local distSq = dx * dx + dy * dy
            if COIN_TILE_LOW_BYTES[tile] and (coinBestSq == nil or distSq < coinBestSq) then
                coinCount = coinCount + 1
                coinBestSq = distSq
                coinDx = math.floor((marioX + dx) / 16) * 16 + 8 - marioX
                coinDy = math.floor((marioY + dy) / 16) * 16 + 8 - marioY
            end
            if COIN_BLOCK_LOW_BYTES[tile] and (blockBestSq == nil or distSq < blockBestSq) then
                blockCount = blockCount + 1
                blockBestSq = distSq
                blockDx = math.floor((marioX + dx) / 16) * 16 + 8 - marioX
                blockDy = math.floor((marioY + dy) / 16) * 16 + 8 - marioY
            end
        end
    end
    return coinCount .. ";" .. coinDx .. ";" .. coinDy .. ";" .. blockCount .. ";" .. blockDx .. ";" .. blockDy
end

local function buildDialogSignals(marioX, marioY)
    local dialogCount = 0
    local dialogDx, dialogDy, dialogBestSq = 0, 0, nil
    for dy = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
        for dx = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
            local tile = getTileFull(marioX, marioY, dx, dy)
            local distSq = dx * dx + dy * dy
            if DIALOG_TILE_FULL[tile] and (dialogBestSq == nil or distSq < dialogBestSq) then
                dialogCount = dialogCount + 1
                dialogBestSq = distSq
                dialogDx = math.floor((marioX + dx) / 16) * 16 + 8 - marioX
                dialogDy = math.floor((marioY + dy) / 16) * 16 + 8 - marioY
            end
        end
    end
    return dialogCount .. ";" .. dialogDx .. ";" .. dialogDy
end

local function buildPipeSignals(marioX, marioY)
    local pipeCount = 0
    local pipeDx, pipeDy, pipeBestSq = 0, 0, nil
    for dy = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
        for dx = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
            local tile = getTileFull(marioX, marioY, dx, dy)
            local distSq = dx * dx + dy * dy
            if PIPE_ENTRANCE_TILE_FULL[tile] and (pipeBestSq == nil or distSq < pipeBestSq) then
                pipeCount = pipeCount + 1
                pipeBestSq = distSq
                pipeDx = math.floor((marioX + dx) / 16) * 16 + 8 - marioX
                pipeDy = math.floor((marioY + dy) / 16) * 16 + 8 - marioY
            end
        end
    end
    return pipeCount .. ";" .. pipeDx .. ";" .. pipeDy
end

local function buildCliffSignals(marioX, marioY)
    local feetRow = math.floor((marioY + 16) / 16)
    local gaps = {}
    local inGap = false
    local gapStart = 0
    for col = 0, GRID_RADIUS do
        local tx = math.floor(marioX / 16) + col
        local open = true
        for row = feetRow, feetRow + 4 do
            local lo = memory.readbyte(MAP16_LOW_BYTE_TABLE + math.floor(tx / 0x10) * 0x1B0 + row * 0x10 + tx % 0x10)
            if lo ~= 0 then
                open = false
                break
            end
        end
        if open and not inGap then
            inGap = true
            gapStart = col
        elseif not open and inGap then
            gaps[#gaps + 1] = { start = gapStart, width = col - gapStart }
            inGap = false
        end
    end
    if inGap then
        gaps[#gaps + 1] = { start = gapStart, width = GRID_RADIUS + 1 - gapStart }
    end
    local signals = {}
    for i = 1, 2 do
        if gaps[i] ~= nil then
            signals[#signals + 1] = gaps[i].start
            signals[#signals + 1] = gaps[i].width
        else
            signals[#signals + 1] = 0
            signals[#signals + 1] = 0
        end
    end
    return table.concat(signals, ";")
end

local function isVerticalLevel()
    return memory.readbyte(0x1412) ~= 0 and 1 or 0
end

local function isSolidTile(tx, ty)
    return memory.readbyte(MAP16_LOW_BYTE_TABLE + math.floor(tx / 0x10) * 0x1B0 + ty * 0x10 + tx % 0x10) ~= 0
end

local function buildWallSignals(marioX, marioY)
    local marioCol = math.floor(marioX / 16)
    local marioRow = math.floor(marioY / 16)
    local feetRow = math.floor((marioY + 16) / 16)

    local wallDistance = 0
    for col = 1, GRID_RADIUS do
        local tx = marioCol + col
        if isSolidTile(tx, marioRow - 1) or isSolidTile(tx, marioRow) or isSolidTile(tx, feetRow) then
            wallDistance = col
            break
        end
    end

    local aboveDistance = 0
    for row = 1, GRID_RADIUS do
        if isSolidTile(marioCol, marioRow - row) then
            aboveDistance = row
            break
        end
    end

    local belowDistance = 0
    for row = 1, GRID_RADIUS do
        if isSolidTile(marioCol, feetRow + row) then
            belowDistance = row
            break
        end
    end

    return wallDistance .. ";" .. aboveDistance .. ";" .. belowDistance
end

local function buildSpriteList()
    local sprites = {}
    for slot = 0, 11 do
        local status = memory.readbyte(0x14C8 + slot)
        if status ~= 0 then
            local x = memory.readbyte(0xE4 + slot) + memory.readbyte(0x14E0 + slot) * 256
            local y = memory.readbyte(0xD8 + slot) + memory.readbyte(0x14D4 + slot) * 256
            local vx = readS8(0xB6 + slot)
            local vy = readS8(0xAA + slot)
            local direction = memory.readbyte(0x157C + slot)
            local blocked = memory.readbyte(0x1588 + slot)
            local offscreen = memory.readbyte(0x15A0 + slot)
            local spriteType = memory.readbyte(0x9E + slot)
            local subX = memory.readbyte(0x14F8 + slot)
            local subY = memory.readbyte(0x14EC + slot)
            local stun = memory.readbyte(0x1540 + slot)
            local props = memory.readbyte(0x167A + slot)
            local misc1 = memory.readbyte(0x1504 + slot)
            local misc2 = memory.readbyte(0x151C + slot)
            local misc3 = memory.readbyte(0x1528 + slot)
            local offscreenFull = memory.readbyte(0x15C4 + slot)
            local eaten = memory.readbyte(0x15D0 + slot)
            local objectInteraction = memory.readbyte(0x15DC + slot)
            local spinTimer = memory.readbyte(0x15AC + slot)
            sprites[#sprites + 1] = x .. "," .. y .. "," .. spriteType .. "," .. vx .. "," .. vy .. "," .. direction .. "," .. blocked .. "," .. offscreen .. "," .. subX .. "," .. subY .. "," .. status .. "," .. stun .. "," .. props .. "," .. misc1 .. "," .. misc2 .. "," .. misc3 .. "," .. offscreenFull .. "," .. eaten .. "," .. objectInteraction .. "," .. spinTimer
        end
    end
    return table.concat(sprites, ";")
end

local function buildClusterList()
    local clusters = {}
    for slot = 0, 19 do
        local x = memory.readbyte(0x1E16 + slot) + memory.readbyte(0x1E3E + slot) * 256
        local y = memory.readbyte(0x1E02 + slot) + memory.readbyte(0x1E2A + slot) * 256
        if x ~= 0 or y ~= 0 then
            clusters[#clusters + 1] = x .. "," .. y
        end
    end
    return table.concat(clusters, ";")
end

local function backgroundLayerPositions()
    return memory.read_s16_le(0x1E), memory.read_s16_le(0x20), memory.read_s16_le(0x22), memory.read_s16_le(0x24)
end

local function isGrounded(marioVY)
    return marioVY == 0
end

local function buildState(levelComplete, manualReset)
    local marioX, marioY = marioPosition()
    local marioVX, marioVY = marioVelocity()
    local subX, subY = marioSubpixel()
    local cameraX, cameraY = cameraPosition()
    local layer2X, layer2Y, layer3X, layer3Y = backgroundLayerPositions()
    local gameMode, levelMode, translevel = gameState()
    local coins, itemBox = marioCollectibles()
    local p1, p2, p3, p4 = controllerState()
    local gameFrame, spriteFrame, lag = frameCounters()
    local subSpeed = marioSubSpeed()
    local reservedItem = marioReservedItemBox()
    local c1Copy, c2Copy = marioControllerCopies()
    local p2c1, p2c2, p2c1Prev, p2c2Prev = secondPlayerControllers()
    local bluePow, silverPow = marioPowTimers()
    local doorExit = doorExitCounter()
    local itemMemoryValue = itemMemory()
    local currentPlayer, character, currentPlayerCoins = currentPlayerState()
    local dead = isMarioDeadEffective() and "1" or "0"
    local lives = livesRemaining()
    local carryingFlag, holdingObjectFlag = marioCarryFlags()
    local midwayFlag, midwaySuppressed = midwayCheckpointFlags()
    local yoshiCoins = yoshiCoinsCollected()
    local tiles = buildTileGrid(marioX, marioY)
    local sprites = buildSpriteList()
    local clusters = buildClusterList()
    local grounded = isGrounded(marioVY) and "1" or "0"
    local powerup = marioPowerup()

    return table.concat({
        gameFrame,
        marioX,
        marioY,
        marioVX,
        marioVY,
        dead,
        lives,
        tiles,
        sprites,
        clusters,
        grounded,
        levelComplete and "1" or "0",
        manualReset and "1" or "0",
        powerup,
        currentLevelIndex,
        marioDirection(),
        marioBlocked(),
        subX,
        subY,
        marioAirState(),
        marioDucking(),
        marioClimbing(),
        marioWater(),
        marioPMeter(),
        marioTakeoff(),
        marioHurt(),
        marioCape(),
        bluePow,
        silverPow,
        doorExit,
        cameraX,
        cameraY,
        layer2X,
        layer2Y,
        layer3X,
        layer3Y,
        gameMode,
        levelMode,
        translevel,
        currentPlayer,
        character,
        itemMemoryValue,
        currentPlayerCoins,
        coins,
        itemBox,
        p1,
        p2,
        p3,
        p4,
        subSpeed,
        reservedItem,
        c1Copy,
        c2Copy,
        spriteFrame,
        lag,
        emu.framecount(),
        p2c1,
        p2c1Prev,
        p2c2,
        p2c2Prev,
        buildCoinSignals(marioX, marioY),
        buildDialogSignals(marioX, marioY),
        buildCliffSignals(marioX, marioY),
        carryingFlag,
        holdingObjectFlag,
        (midwayFlag ~= 0 and midwaySuppressed == 0) and 1 or 0,
        yoshiCoins,
        buildWallSignals(marioX, marioY),
        isVerticalLevel(),
        buildPipeSignals(marioX, marioY)
    }, "|")
end

local function loadLevel(index)
    currentLevelIndex = index
    savestate.load(currentSavestateFile())
    beginResetGrace()
end

-- ============================================================
-- Overlay de "vision" del agente: dibuja sobre la pantalla de
-- BizHawk la misma cuadricula de tiles (GRID_RADIUS=8 -> 17x17,
-- centrada en Mario, cubre toda la pantalla) que buildTileGrid()
-- le manda a la red,
-- coloreada por tipo de tile, mas los sprites/enemigos y las
-- senales de pared/precipicio (buildWallSignals) que tambien
-- recibe el modelo. Es solo para depurar visualmente, no afecta
-- el entrenamiento ni el protocolo con C#.
--
-- Toggle: tecla "V" del teclado (no choca con los botones de
-- SNES). Se dibuja siempre que este activado, incluso con turbo
-- prendido -- a velocidades muy altas es probable que se vea
-- borroso o parpadee, pero eso lo deja ver el usuario.
--
-- Los nombres exactos de gui.drawRectangle/drawText/drawLine
-- pueden variar segun la version de BizHawk (algunas viejas usan
-- gui.drawBox en vez de gui.drawRectangle), asi que todo el
-- dibujo va envuelto en pcall: si algo no existe, se avisa UNA
-- vez por consola con el error exacto en vez de romper el script.
-- ============================================================
local VISION_TOGGLE_KEY = "V"

local TILE_SOLID_FILL = 0x50808080
local TILE_COIN_FILL = 0x90FFD700
local TILE_COINBLOCK_FILL = 0x90FFA500
local TILE_DIALOG_FILL = 0x90A020F0
local TILE_PIPE_FILL = 0x9000C000
local MARIO_CELL_BORDER = 0xFF00FFFF
local GRID_LINE_COLOR = 0x30FFFFFF
local SPRITE_BOX_COLOR = 0xFFFF0000
local SPRITE_TEXT_COLOR = 0xFFFF6060
local CLUSTER_BOX_COLOR = 0xFF3080FF
local WALL_SIGNAL_COLOR = 0xFFFF00FF
local LEGEND_BG = 0xC0000000
local LEGEND_TEXT = 0xFFFFFFFF
local TILE_ID_LABEL_COLOR = 0xFFFFFFFF

-- Debounce por tiempo REAL (no por frames): bajo turbo el loop corre
-- muchisimo mas rapido que a velocidad normal, y a esa velocidad se
-- alcanza a leer el rebote mecanico normal de la tecla (el "bounce"
-- fisico del contacto, que dura pocos milisegundos y normalmente pasa
-- desapercibido a 60fps) como varias pulsaciones separadas. Usamos
-- os.clock() en vez de contar frames porque un debounce en frames
-- significaria una ventana de tiempo real distinta segun la velocidad
-- del turbo (a 6400% muchos frames pasan en milisegundos). Si os.clock
-- no esta disponible en esta build de BizHawk, se cae de nuevo al
-- comportamiento sin debounce en vez de romper el toggle.
local VISION_TOGGLE_DEBOUNCE_SECONDS = 0.15
local visionLastToggleClock = nil
local visionClockAvailable = true

local function visionTogglePulse()
    local keys = input.get()
    local isDown = keys[VISION_TOGGLE_KEY] == true
    local rawPulse = isDown and not visionToggleKeyWasDown
    visionToggleKeyWasDown = isDown

    if not rawPulse then
        return false
    end

    if not visionClockAvailable then
        return true
    end

    local ok, now = pcall(os.clock)
    if not ok then
        visionClockAvailable = false
        return true
    end

    if visionLastToggleClock ~= nil and (now - visionLastToggleClock) < VISION_TOGGLE_DEBOUNCE_SECONDS then
        return false
    end

    visionLastToggleClock = now
    return true
end

local function tileFillColor(tileFull, tileLow)
    if COIN_TILE_LOW_BYTES[tileLow] then
        return TILE_COIN_FILL
    end
    if COIN_BLOCK_LOW_BYTES[tileLow] then
        return TILE_COINBLOCK_FILL
    end
    if DIALOG_TILE_FULL[tileFull] then
        return TILE_DIALOG_FILL
    end
    if PIPE_ENTRANCE_TILE_FULL[tileFull] then
        return TILE_PIPE_FILL
    end
    if tileLow ~= 0 then
        return TILE_SOLID_FILL
    end
    return nil
end

local function drawVisionOverlay(marioX, marioY, cameraX, cameraY)
    -- Cuadricula: un rectangulo de 16x16 por celda, alineado igual
    -- que getTile()/getTileFull(), coloreado segun el tipo de tile.
    for dy = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
        for dx = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
            local tileFull = getTileFull(marioX, marioY, dx, dy)
            local tileLow = tileFull % 256
            local cellWorldX = math.floor((marioX + dx) / 16) * 16
            local cellWorldY = math.floor((marioY + dy) / 16) * 16
            local screenX = cellWorldX - cameraX
            local screenY = cellWorldY - cameraY

            gui.drawRectangle(screenX, screenY, 16, 16, GRID_LINE_COLOR, tileFillColor(tileFull, tileLow))

            -- Etiqueta de diagnostico: el ID crudo de Map16 (2 hex digits,
            -- el mismo tileLow que se compara contra COIN_TILE_LOW_BYTES /
            -- COIN_BLOCK_LOW_BYTES / etc) sobre cualquier tile no vacio.
            -- Sirve para detectar de un vistazo cuando un tile se pinta gris
            -- "solido" generico en vez de su color especifico -- significa
            -- que su ID no esta en ninguna de las tablas de clasificacion de
            -- arriba y hay que sumarlo. No afecta al entrenamiento, es solo
            -- para leer el numero desde una captura de pantalla.
            if tileLow ~= 0 then
                gui.drawText(screenX + 1, screenY + 4, string.format("%02X", tileLow), TILE_ID_LABEL_COLOR, nil, 8)
            end

            if dx == 0 and dy == 0 then
                gui.drawRectangle(screenX, screenY, 16, 16, MARIO_CELL_BORDER, nil)
            end
        end
    end

    -- Sprites/enemigos: mismos slots y direcciones de memoria que
    -- buildSpriteList(), etiquetados con su tipo en hex.
    for slot = 0, 11 do
        local status = memory.readbyte(0x14C8 + slot)
        if status ~= 0 then
            local x = memory.readbyte(0xE4 + slot) + memory.readbyte(0x14E0 + slot) * 256
            local y = memory.readbyte(0xD8 + slot) + memory.readbyte(0x14D4 + slot) * 256
            local spriteType = memory.readbyte(0x9E + slot)
            local screenX = x - cameraX
            local screenY = y - cameraY
            gui.drawRectangle(screenX, screenY, 16, 16, SPRITE_BOX_COLOR, nil)
            gui.drawText(screenX, screenY - 10, string.format("%02X", spriteType), SPRITE_TEXT_COLOR, nil, 8)
        end
    end

    -- Sprites "cluster": algunos enemigos de SMW (los que van en fila
    -- o multi-segmento) no usan la tabla de sprites normal de arriba,
    -- sino esta tabla separada (mismas direcciones que buildClusterList()).
    -- Se dibujan en azul para diferenciarlos de los sprites comunes.
    for slot = 0, 19 do
        local x = memory.readbyte(0x1E16 + slot) + memory.readbyte(0x1E3E + slot) * 256
        local y = memory.readbyte(0x1E02 + slot) + memory.readbyte(0x1E2A + slot) * 256
        if x ~= 0 or y ~= 0 then
            local screenX = x - cameraX
            local screenY = y - cameraY
            gui.drawRectangle(screenX, screenY, 16, 16, CLUSTER_BOX_COLOR, nil)
        end
    end

    -- Senales de pared/techo/piso mas cercano (buildWallSignals),
    -- dibujadas como lineas desde Mario hacia donde detecta el limite.
    local wallDistance, aboveDistance, belowDistance = string.match(
        buildWallSignals(marioX, marioY), "(%d+);(%d+);(%d+)"
    )
    local marioScreenX = marioX - cameraX
    local marioScreenY = marioY - cameraY
    if tonumber(wallDistance) > 0 then
        gui.drawLine(marioScreenX + 16, marioScreenY + 8, marioScreenX + 16 + tonumber(wallDistance) * 16, marioScreenY + 8, WALL_SIGNAL_COLOR)
    end
    if tonumber(aboveDistance) > 0 then
        gui.drawLine(marioScreenX + 8, marioScreenY, marioScreenX + 8, marioScreenY - tonumber(aboveDistance) * 16, WALL_SIGNAL_COLOR)
    end
    if tonumber(belowDistance) > 0 then
        gui.drawLine(marioScreenX + 8, marioScreenY + 16, marioScreenX + 8, marioScreenY + 16 + tonumber(belowDistance) * 16, WALL_SIGNAL_COLOR)
    end

    -- Leyenda fija en la esquina superior izquierda.
    gui.drawRectangle(2, 2, 152, 100, LEGEND_BG, LEGEND_BG)
    gui.drawText(6, 4, "Vision del agente (V)", LEGEND_TEXT, nil, 8)
    gui.drawText(6, 16, "Gris = solido", TILE_SOLID_FILL, nil, 8)
    gui.drawText(6, 26, "Amarillo = moneda", TILE_COIN_FILL, nil, 8)
    gui.drawText(6, 36, "Naranja = bloque moneda", TILE_COINBLOCK_FILL, nil, 8)
    gui.drawText(6, 46, "Morado = dialogo", TILE_DIALOG_FILL, nil, 8)
    gui.drawText(6, 56, "Verde = tuberia", TILE_PIPE_FILL, nil, 8)
    gui.drawText(6, 66, "Rojo = sprite/enemigo", SPRITE_BOX_COLOR, nil, 8)
    gui.drawText(6, 76, "Azul = sprite cluster", CLUSTER_BOX_COLOR, nil, 8)
    gui.drawText(6, 86, "Numero = ID Map16 crudo", TILE_ID_LABEL_COLOR, nil, 8)
end

local function applyAction(response)
    if response == STOP_COMMAND then
        return true
    end

    local resetIndex = tonumber(string.match(response, "^RESET:(%d+)$"))
    if resetIndex ~= nil then
        captureMode = false
        loadLevel(resetIndex)
        return false
    end

    if response == RESET_COMMAND then
        captureMode = false
        savestate.load(currentSavestateFile())
        beginResetGrace()
        return false
    end

    local captureIndex = tonumber(string.match(response, "^CAPTURE:(%d+)$"))
    if captureIndex ~= nil then
        captureMode = true
        loadLevel(captureIndex)
        return false
    end

    local turboValue = string.match(response, "^" .. TURBO_COMMAND .. ":(%d)$")
    if turboValue ~= nil then
        setTurbo(turboValue == "1")
        return false
    end

    if captureMode then
        return false
    end

    local controller = {}
    for _, name in ipairs(BUTTON_NAMES) do
        controller["P1 " .. name] = false
    end

    if response ~= "None" then
        for name in string.gmatch(response, "[^,]+") do
            controller["P1 " .. name] = true
        end
    end

    joypad.set(controller)
    return false
end

-- Cuantos frames seguidos vinieron fallando dentro del ciclo principal
-- (send/receive/applyAction). Sirve solo para no inundar la consola de
-- Lua si el error se repite muchos frames seguidos; el conteo se resetea
-- apenas un frame se procesa bien.
local consecutiveLoopErrors = 0
local MAX_LOOP_ERROR_LOGS = 5

while true do
    -- Todo el ciclo de leer estado + mandarlo por el socket + leer la
    -- respuesta + aplicar la accion va envuelto en pcall a proposito: si
    -- cualquiera de estas funciones (buildState/buildTileGrid/
    -- buildSpriteList/buildClusterList, etc.) tira un error de Lua no
    -- controlado en un frame puntual (un evento raro del juego que rompe
    -- un supuesto sobre la RAM), BizHawk mataba el script entero en
    -- silencio -- la conexion TCP quedaba "viva" del lado del socket
    -- pero nadie volvia a mandar datos nunca mas, y del lado de C# eso se
    -- ve como un timeout de 60s sin ninguna pista de la causa real. Con
    -- esto, en cambio, se loguea el error real y se salta ese frame (sin
    -- mandar nada por el socket ese ciclo), dejando que el juego y la
    -- captura sigan.
    local shouldStop = false
    local ok, errOrStop = pcall(function()
        local forcedReset = false
        if isMessageBoxActive() then
            forcedReset = dismissMessageBox()
        end

        local manualReset = manualResetPulse() or forcedReset
        comm.socketServerSend(buildState(isLevelCompleteEffective(), manualReset) .. "\n")
        local response = comm.socketServerResponse()
        return applyAction(response)
    end)

    if ok then
        consecutiveLoopErrors = 0
        shouldStop = errOrStop
    else
        consecutiveLoopErrors = consecutiveLoopErrors + 1
        if consecutiveLoopErrors <= MAX_LOOP_ERROR_LOGS then
            console.log("MarioBridge: error en el ciclo principal (frame " .. tostring(emu.framecount()) .. "), se salta este frame sin mandar datos: " .. tostring(errOrStop))
            if consecutiveLoopErrors == MAX_LOOP_ERROR_LOGS then
                console.log("MarioBridge: se silencian mas repeticiones de este error hasta que se recupere un frame OK.")
            end
        end
    end

    if shouldStop then
        console.log("MarioBridge: finalizado, deteniendo script.")
        break
    end

    if visionTogglePulse() then
        visionOverlayEnabled = not visionOverlayEnabled
        console.log("MarioBridge: overlay de vision " .. (visionOverlayEnabled and "activado" or "desactivado") .. ".")
    end

    -- Se limpia SIEMPRE, este activado o no el overlay. gui.draw* en
    -- BizHawk no se borra solo entre frames: si no llamamos esto antes
    -- de decidir si redibujamos, el ultimo frame dibujado (grid, sprites,
    -- leyenda) queda pegado en pantalla para siempre en cuanto se apaga
    -- el toggle. Va en pcall porque el nombre puede variar entre
    -- versiones de BizHawk (algunas viejas no exponen clearGraphics).
    local clearOk, clearErr = pcall(gui.clearGraphics)
    if not clearOk and not visionClearWarned then
        visionClearWarned = true
        console.log("MarioBridge: no se pudo limpiar el overlay de vision (" .. tostring(clearErr) .. "). Puede que tu version de BizHawk no tenga gui.clearGraphics.")
    end

    if visionOverlayEnabled then
        local marioX, marioY = marioPosition()
        local cameraX, cameraY = cameraPosition()
        local ok, err = pcall(drawVisionOverlay, marioX, marioY, cameraX, cameraY)
        if not ok and not visionDrawWarned then
            visionDrawWarned = true
            console.log("MarioBridge: no se pudo dibujar el overlay de vision (" .. tostring(err) .. "). Puede que tu version de BizHawk use otros nombres para gui.drawRectangle/gui.drawText/gui.drawLine (por ejemplo gui.drawBox en versiones viejas).")
        end
    end

    emu.frameadvance()

    if resetGraceFramesRemaining > 0 then
        resetGraceFramesRemaining = resetGraceFramesRemaining - 1
        if resetGraceFramesRemaining == 0 then
            loggedGraceSuppression = false
        end
    end
end