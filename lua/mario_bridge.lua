local function isAbsolutePath(path)
    return path:match("^%a:[/\\]") ~= nil
        or path:match("^[/\\][/\\]") ~= nil
        or path:match("^/") ~= nil
end

local function scriptDirectory()
    local info = debug.getinfo(1, "S")
    local source = info and info.source or nil
    if source == nil or source:sub(1, 1) ~= "@" then
        return nil
    end
    local path = source:sub(2)
    return path:match("^(.*[/\\])")
end

local function resolveSavestateDir()
    local envDir = os.getenv("NEURAVAL_SAVESTATE_DIR")
    local dir

    if envDir ~= nil and envDir ~= "" then
        dir = envDir
    else
        local ok, scriptDir = pcall(scriptDirectory)
        if ok and scriptDir ~= nil then
            dir = scriptDir
        end
    end

    if dir == nil then
        console.log("MarioBridge: no se pudo detectar la carpeta del script; usando \"./\".")
        return "./"
    end

    if not isAbsolutePath(dir) then
        console.log("MarioBridge: ATENCION - la carpeta resuelta (\"" .. dir .. "\") es relativa, puede fallar al cargar savestates.")
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

for levelIndex, path in pairs(SAVESTATE_FILES) do
    local file = io.open(path, "rb")
    if file == nil then
        console.log("MarioBridge: ATENCION - no se encuentra el savestate del nivel " .. tostring(levelIndex) .. " en \"" .. path .. "\".")
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

local TURBO_SPEED_PERCENT = 6400
local NORMAL_SPEED_PERCENT = 100
local turboEnabled = false

local visionOverlayEnabled = true
local visionToggleKeyWasDown = false
local visionDrawWarned = false
local visionClearWarned = false
local visionWasEnabled = false

local function setTurbo(enabled)
    turboEnabled = enabled

    local speedOk, speedErr = pcall(function()
        client.speedmode(enabled and TURBO_SPEED_PERCENT or NORMAL_SPEED_PERCENT)
    end)
    if not speedOk then
        console.log("MarioBridge: no se pudo cambiar la velocidad con client.speedmode (" .. tostring(speedErr) .. ").")
    end

    if client.SetSoundOn ~= nil then
        pcall(function() client.SetSoundOn(not enabled) end)
    end

    console.log("MarioBridge: turbo " .. (enabled and "activado" or "desactivado") .. ".")
end

event.onexit(function() setTurbo(false) end)

local GRID_RADIUS = 8
local BUTTON_NAMES = { "A", "B", "X", "Y", "Up", "Down", "Left", "Right", "L", "R", "Select", "Start" }
local MESSAGE_BOX_ADDR = 0x1426
local MESSAGE_BOX_HOLD_FRAMES = 4
local MESSAGE_BOX_RELEASE_FRAMES = 4
local MAX_DISMISS_ATTEMPTS = 90
local LEVEL_END_ADDR = 0x1493
local MANUAL_RESET_KEY = "Insert"
local manualResetKeyWasDown = false

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

local loggedGraceSuppression = false

local function isMarioDeadEffective()
    if resetGraceFramesRemaining > 0 then
        if isMarioDead() and not loggedGraceSuppression then
            loggedGraceSuppression = true
            console.log("MarioBridge: flag de muerte suprimido por ventana de gracia (" .. resetGraceFramesRemaining .. " frames restantes).")
        end
        return false
    end
    return isMarioDead()
end

local function isLevelCompleteEffective()
    if resetGraceFramesRemaining > 0 then
        if isLevelComplete() and not loggedGraceSuppression then
            loggedGraceSuppression = true
            console.log("MarioBridge: flag de nivel completo suprimido por ventana de gracia (" .. resetGraceFramesRemaining .. " frames restantes).")
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

local MAP16_LOW_BYTE_TABLE = 0xC800
local MAP16_HIGH_BYTE_TABLE = 0x1C800

local function tileIndex(tx, ty)
    return math.floor(tx / 0x10) * 0x1B0 + ty * 0x10 + tx % 0x10
end

local function getTileFull(marioX, marioY, dx, dy)
    local tx = math.floor((marioX + dx) / 16)
    local ty = math.floor((marioY + dy) / 16)
    local idx = tileIndex(tx, ty)
    local lo = memory.readbyte(MAP16_LOW_BYTE_TABLE + idx)
    local hi = memory.readbyte(MAP16_HIGH_BYTE_TABLE + idx)
    return hi * 256 + lo
end

local function isSolidTile(tx, ty)
    return memory.readbyte(MAP16_LOW_BYTE_TABLE + tileIndex(tx, ty)) ~= 0
end

local COIN_TILE_LOW_BYTES = { [0x2B] = true }
local COIN_BLOCK_LOW_BYTES = { [0x1B] = true, [0x23] = true }
local DIALOG_TILE_FULL = { [0x0104] = true, [0x0105] = true, [0x0106] = true, [0x0107] = true }
local PIPE_ENTRANCE_TILE_FULL = { [0x0137] = true, [0x0138] = true }

local VISION_GRID_SIDE = 2 * GRID_RADIUS + 1
local VISION_GRID_SIZE = VISION_GRID_SIDE * VISION_GRID_SIDE
local visionGridCache = {}
for i = 1, VISION_GRID_SIZE do
    visionGridCache[i] = { dx = 0, dy = 0, tileFull = 0, tileLow = 0 }
end
local visionGridCount = 0
local visionGridMarioX, visionGridMarioY = nil, nil

local function buildVisionData(marioX, marioY)
    visionGridCount = 0
    visionGridMarioX, visionGridMarioY = marioX, marioY

    local tiles = {}
    local tileCount = 0
    local coinCount, coinDx, coinDy, coinBestSq = 0, 0, 0, nil
    local blockCount, blockDx, blockDy, blockBestSq = 0, 0, 0, nil
    local dialogCount, dialogDx, dialogDy, dialogBestSq = 0, 0, 0, nil
    local pipeCount, pipeDx, pipeDy, pipeBestSq = 0, 0, 0, nil

    for dy = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
        for dx = -GRID_RADIUS * 16, GRID_RADIUS * 16, 16 do
            local tileFull = getTileFull(marioX, marioY, dx, dy)
            local tileLow = tileFull % 256

            tileCount = tileCount + 1
            tiles[tileCount] = tostring(tileLow)

            visionGridCount = visionGridCount + 1
            local cell = visionGridCache[visionGridCount]
            cell.dx, cell.dy, cell.tileFull, cell.tileLow = dx, dy, tileFull, tileLow

            local distSq = dx * dx + dy * dy
            local cellX = math.floor((marioX + dx) / 16) * 16 + 8 - marioX
            local cellY = math.floor((marioY + dy) / 16) * 16 + 8 - marioY

            if COIN_TILE_LOW_BYTES[tileLow] and (coinBestSq == nil or distSq < coinBestSq) then
                coinCount = coinCount + 1
                coinBestSq = distSq
                coinDx, coinDy = cellX, cellY
            end
            if COIN_BLOCK_LOW_BYTES[tileLow] and (blockBestSq == nil or distSq < blockBestSq) then
                blockCount = blockCount + 1
                blockBestSq = distSq
                blockDx, blockDy = cellX, cellY
            end
            if DIALOG_TILE_FULL[tileFull] and (dialogBestSq == nil or distSq < dialogBestSq) then
                dialogCount = dialogCount + 1
                dialogBestSq = distSq
                dialogDx, dialogDy = cellX, cellY
            end
            if PIPE_ENTRANCE_TILE_FULL[tileFull] and (pipeBestSq == nil or distSq < pipeBestSq) then
                pipeCount = pipeCount + 1
                pipeBestSq = distSq
                pipeDx, pipeDy = cellX, cellY
            end
        end
    end

    return {
        tilesStr = table.concat(tiles, ","),
        coinsStr = coinCount .. ";" .. coinDx .. ";" .. coinDy .. ";" .. blockCount .. ";" .. blockDx .. ";" .. blockDy,
        dialogStr = dialogCount .. ";" .. dialogDx .. ";" .. dialogDy,
        pipeStr = pipeCount .. ";" .. pipeDx .. ";" .. pipeDy,
    }
end

local function buildCliffSignals(marioX, marioY)
    local feetRow = math.floor((marioY + 16) / 16)
    local marioCol = math.floor(marioX / 16)
    local gaps = {}
    local inGap = false
    local gapStart = 0

    for col = 0, GRID_RADIUS do
        local tx = marioCol + col
        local open = true
        for row = feetRow, feetRow + 4 do
            if isSolidTile(tx, row) then
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

local lastWallDistance, lastAboveDistance, lastBelowDistance = 0, 0, 0

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

    lastWallDistance, lastAboveDistance, lastBelowDistance = wallDistance, aboveDistance, belowDistance
    return wallDistance .. ";" .. aboveDistance .. ";" .. belowDistance
end

local function buildSpriteList()
    local sprites = {}
    local count = 0
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
            count = count + 1
            sprites[count] = x .. "," .. y .. "," .. spriteType .. "," .. vx .. "," .. vy .. "," .. direction .. "," .. blocked .. "," .. offscreen .. "," .. subX .. "," .. subY .. "," .. status .. "," .. stun .. "," .. props .. "," .. misc1 .. "," .. misc2 .. "," .. misc3 .. "," .. offscreenFull .. "," .. eaten .. "," .. objectInteraction .. "," .. spinTimer
        end
    end
    return table.concat(sprites, ";")
end

local function buildClusterList()
    local clusters = {}
    local count = 0
    for slot = 0, 19 do
        local x = memory.readbyte(0x1E16 + slot) + memory.readbyte(0x1E3E + slot) * 256
        local y = memory.readbyte(0x1E02 + slot) + memory.readbyte(0x1E2A + slot) * 256
        if x ~= 0 or y ~= 0 then
            count = count + 1
            clusters[count] = x .. "," .. y
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

local lastMarioX, lastMarioY, lastCameraX, lastCameraY = 0, 0, 0, 0

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
    local vision = buildVisionData(marioX, marioY)
    local sprites = buildSpriteList()
    local clusters = buildClusterList()
    local grounded = isGrounded(marioVY) and "1" or "0"
    local powerup = marioPowerup()

    lastMarioX, lastMarioY, lastCameraX, lastCameraY = marioX, marioY, cameraX, cameraY

    return table.concat({
        gameFrame,
        marioX,
        marioY,
        marioVX,
        marioVY,
        dead,
        lives,
        vision.tilesStr,
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
        vision.coinsStr,
        vision.dialogStr,
        buildCliffSignals(marioX, marioY),
        carryingFlag,
        holdingObjectFlag,
        (midwayFlag ~= 0 and midwaySuppressed == 0) and 1 or 0,
        yoshiCoins,
        buildWallSignals(marioX, marioY),
        isVerticalLevel(),
        vision.pipeStr
    }, "|")
end

local function loadLevel(index)
    currentLevelIndex = index
    savestate.load(currentSavestateFile())
    beginResetGrace()
end

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
    if visionGridMarioX == marioX and visionGridMarioY == marioY then
        for i = 1, visionGridCount do
            local cell = visionGridCache[i]
            local cellWorldX = math.floor((marioX + cell.dx) / 16) * 16
            local cellWorldY = math.floor((marioY + cell.dy) / 16) * 16
            local screenX = cellWorldX - cameraX
            local screenY = cellWorldY - cameraY

            gui.drawRectangle(screenX, screenY, 16, 16, GRID_LINE_COLOR, tileFillColor(cell.tileFull, cell.tileLow))

            if cell.tileLow ~= 0 then
                gui.drawText(screenX + 1, screenY + 4, string.format("%02X", cell.tileLow), TILE_ID_LABEL_COLOR, nil, 8)
            end

            if cell.dx == 0 and cell.dy == 0 then
                gui.drawRectangle(screenX, screenY, 16, 16, MARIO_CELL_BORDER, nil)
            end
        end
    end

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

    for slot = 0, 19 do
        local x = memory.readbyte(0x1E16 + slot) + memory.readbyte(0x1E3E + slot) * 256
        local y = memory.readbyte(0x1E02 + slot) + memory.readbyte(0x1E2A + slot) * 256
        if x ~= 0 or y ~= 0 then
            local screenX = x - cameraX
            local screenY = y - cameraY
            gui.drawRectangle(screenX, screenY, 16, 16, CLUSTER_BOX_COLOR, nil)
        end
    end

    local marioScreenX = marioX - cameraX
    local marioScreenY = marioY - cameraY
    if lastWallDistance > 0 then
        gui.drawLine(marioScreenX + 16, marioScreenY + 8, marioScreenX + 16 + lastWallDistance * 16, marioScreenY + 8, WALL_SIGNAL_COLOR)
    end
    if lastAboveDistance > 0 then
        gui.drawLine(marioScreenX + 8, marioScreenY, marioScreenX + 8, marioScreenY - lastAboveDistance * 16, WALL_SIGNAL_COLOR)
    end
    if lastBelowDistance > 0 then
        gui.drawLine(marioScreenX + 8, marioScreenY + 16, marioScreenX + 8, marioScreenY + 16 + lastBelowDistance * 16, WALL_SIGNAL_COLOR)
    end

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

local consecutiveLoopErrors = 0
local MAX_LOOP_ERROR_LOGS = 5

while true do
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
            console.log("MarioBridge: error en el ciclo principal (frame " .. tostring(emu.framecount()) .. "), se salta este frame: " .. tostring(errOrStop))
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

    local shouldDrawOverlay = visionOverlayEnabled

    if shouldDrawOverlay or visionWasEnabled then
        local clearOk, clearErr = pcall(gui.clearGraphics)
        if not clearOk and not visionClearWarned then
            visionClearWarned = true
            console.log("MarioBridge: no se pudo limpiar el overlay de vision (" .. tostring(clearErr) .. ").")
        end
    end

    if shouldDrawOverlay then
        local ok2, err2 = pcall(drawVisionOverlay, lastMarioX, lastMarioY, lastCameraX, lastCameraY)
        if not ok2 and not visionDrawWarned then
            visionDrawWarned = true
            console.log("MarioBridge: no se pudo dibujar el overlay de vision (" .. tostring(err2) .. ").")
        end
    end

    visionWasEnabled = shouldDrawOverlay

    emu.frameadvance()

    if resetGraceFramesRemaining > 0 then
        resetGraceFramesRemaining = resetGraceFramesRemaining - 1
        if resetGraceFramesRemaining == 0 then
            loggedGraceSuppression = false
        end
    end
end
