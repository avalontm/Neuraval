local SAVESTATE_FILE = "D:/_CODE_/BizHawk/DP1.state"
local RESET_COMMAND = "RESET"
local STOP_COMMAND = "STOP"
local GRID_RADIUS = 6
local BUTTON_NAMES = { "A", "B", "X", "Y", "Up", "Down", "Left", "Right", "L", "R", "Select", "Start" }

local function marioPosition()
    local x = memory.read_s16_le(0x94)
    local y = memory.read_s16_le(0x96)
    return x, y
end

local function marioVelocity()
    local vx = memory.read_s8(0x7B)
    local vy = memory.read_s8(0x7D)
    return vx, vy
end

local function isMarioDead()
    return memory.readbyte(0x71) == 0x09
end

local function livesRemaining()
    return memory.readbyte(0x0DBE) + 1
end

-- $7E:1426 = Message box trigger. 0 = ninguno, >0 = hay un cartel de dialogo
-- activo (mensaje de nivel, "gracias" de Yoshi, etc). Mientras esta activo el
-- juego queda esperando un boton para cerrarlo; la red nunca aprendio a
-- apretar Start en ese contexto (ni siquiera es una de sus salidas), asi que
-- el episodio se quedaba trabado ahi hasta el timeout/reset manual.
local MESSAGE_BOX_ADDR = 0x1426
local MESSAGE_BOX_HOLD_FRAMES = 4
local MESSAGE_BOX_RELEASE_FRAMES = 4

local function isMessageBoxActive()
    return memory.readbyte(MESSAGE_BOX_ADDR) ~= 0
end

local function releaseAllButtons()
    local controller = {}
    for _, name in ipairs(BUTTON_NAMES) do
        controller["P1 " .. name] = false
    end
    joypad.set(controller)
end

-- $7E:1493 = "activate end level flag": el juego lo pone en un valor
-- distinto de 0 (normalmente $FF) apenas Mario toca la meta (tape/orb) y
-- arranca la secuencia de fin de nivel. Lo usamos para avisarle a C# que
-- el episodio termino por completar el nivel (no por morir).
local LEVEL_END_ADDR = 0x1493

local function isLevelComplete()
    return memory.readbyte(LEVEL_END_ADDR) ~= 0
end

-- Reset manual "por las dudas": mantené la ventana de BizHawk enfocada y
-- apreta esta tecla si el agente se queda pegado en algo que las
-- detecciones automaticas (dialogo, muerte, fin de nivel) no cubren.
-- Corta el episodio actual como si Mario hubiera muerto, sin tener que
-- cerrar ni reiniciar el proceso de entrenamiento. Si esta tecla choca con
-- algun hotkey que ya uses en BizHawk, cambiala aca nomas.
local MANUAL_RESET_KEY = "Insert"
local manualResetKeyWasDown = false

local function manualResetPulse()
    local keys = input.get()
    local isDown = keys[MANUAL_RESET_KEY] == true
    local pulse = isDown and not manualResetKeyWasDown
    manualResetKeyWasDown = isDown
    return pulse
end
-- alterna sostener Start+A (los dos candidatos mas comunes para "avanzar
-- texto" en SMW) con soltarlos, para no depender de saber con certeza cual
-- de los dos es en esta build en particular. Estos frames no se le mandan
-- al agente (no cuentan como decision suya ni como parte del episodio).
--
-- Limite de seguridad: si despues de MAX_DISMISS_ATTEMPTS ciclos el cartel
-- sigue sin cerrarse (direccion de memoria rara, cartel que necesita otro
-- boton, o cualquier estado que no previmos), esto se colgaria para
-- siempre -- sin este limite, nunca vuelve a mandar nada por el socket, y
-- del lado de C# el entrenamiento queda congelado en silencio sin generar
-- mas checkpoints, indistinguible de "no pasa nada". En vez de eso, se
-- fuerza un reset por savestate (mismo mecanismo que RESET_COMMAND) para
-- que el entrenamiento se recupere solo.
local MAX_DISMISS_ATTEMPTS = 90 -- ~90 * (4+4) frames = ~12s a 60fps

local function dismissMessageBox()
    local attempts = 0
    while isMessageBoxActive() do
        attempts = attempts + 1
        if attempts > MAX_DISMISS_ATTEMPTS then
            console.log("MarioBridge: cartel de dialogo no se cerro despues de " .. MAX_DISMISS_ATTEMPTS ..
                " intentos; forzando reset por savestate para no colgar el entrenamiento.")
            releaseAllButtons()
            savestate.load(SAVESTATE_FILE)
            return
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
end

local function getTile(marioX, marioY, dx, dy)
    local x = math.floor((marioX + dx) / 16)
    local y = math.floor((marioY + dy) / 16)
    return memory.readbyte(0x1C800 + math.floor(x / 0x10) * 0x1B0 + y * 0x10 + x % 0x10)
end

local function buildTileGrid(marioX, marioY)
    local tiles = {}
    for dy = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
        for dx = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
            local tile = getTile(marioX, marioY, dx, dy)
            tiles[#tiles + 1] = tile ~= 0 and "1" or "0"
        end
    end
    return table.concat(tiles)
end

local function buildSpriteList()
    local sprites = {}
    for slot = 0, 11 do
        local status = memory.readbyte(0x14C8 + slot)
        if status ~= 0 then
            local x = memory.readbyte(0xE4 + slot) + memory.readbyte(0x14E0 + slot) * 256
            local y = memory.readbyte(0xD8 + slot) + memory.readbyte(0x14D4 + slot) * 256
            local spriteType = memory.readbyte(0x9E + slot)
            sprites[#sprites + 1] = x .. "," .. y .. "," .. spriteType
        end
    end
    return table.concat(sprites, ";")
end

local function buildState(levelComplete, manualReset)
    local marioX, marioY = marioPosition()
    local marioVX, marioVY = marioVelocity()
    local dead = isMarioDead() and "1" or "0"
    local lives = livesRemaining()
    local tiles = buildTileGrid(marioX, marioY)
    local sprites = buildSpriteList()

    return table.concat({
        emu.framecount(),
        marioX,
        marioY,
        marioVX,
        marioVY,
        dead,
        lives,
        tiles,
        sprites,
        levelComplete and "1" or "0",
        manualReset and "1" or "0"
    }, "|")
end

local function applyAction(response)
    if response == STOP_COMMAND then
        return true
    end

    if response == RESET_COMMAND then
        savestate.load(SAVESTATE_FILE)
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

while true do
    if isMessageBoxActive() then
        dismissMessageBox()
    end

    local manualReset = manualResetPulse()
    comm.socketServerSend(buildState(isLevelComplete(), manualReset) .. "\n")
    local response = comm.socketServerResponse()
    local shouldStop = applyAction(response)
    if shouldStop then
        console.log("MarioBridge: entrenamiento finalizado, deteniendo script.")
        break
    end
    emu.frameadvance()
end
