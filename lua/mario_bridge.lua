local SAVESTATE_DIR = "D:/_CODE_/BizHawk/"

local SAVESTATE_FILES = {
    [0] = SAVESTATE_DIR .. "DP1.state",
}

local currentLevelIndex = 0
local captureMode = false

local RESET_COMMAND = "RESET"
local STOP_COMMAND = "STOP"
local CAPTURE_COMMAND = "CAPTURE"
local GRID_RADIUS = 6
local BUTTON_NAMES = { "A", "B", "X", "Y", "Up", "Down", "Left", "Right", "L", "R", "Select", "Start" }
local MESSAGE_BOX_ADDR = 0x1426
local MESSAGE_BOX_HOLD_FRAMES = 4
local MESSAGE_BOX_RELEASE_FRAMES = 4
local MAX_DISMISS_ATTEMPTS = 90
local LEVEL_END_ADDR = 0x1493
local MANUAL_RESET_KEY = "Insert"
local manualResetKeyWasDown = false

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

local function getTileFull(marioX, marioY, dx, dy)
    local x = math.floor((marioX + dx) / 16)
    local y = math.floor((marioY + dy) / 16)
    local idx = math.floor(x / 0x10) * 0x1B0 + y * 0x10 + x % 0x10
    local lo = memory.readbyte(0x1C800 + idx)
    local hi = memory.readbyte(0x1D800 + idx)
    return hi * 256 + lo
end

local function getTile(marioX, marioY, dx, dy)
    return getTileFull(marioX, marioY, dx, dy) % 256
end

local COIN_TILE_LOW_BYTES = { [0x25] = true, [0x2B] = true, [0x5B] = true, [0x6B] = true }
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
            local lo = memory.readbyte(0x1C800 + math.floor(tx / 0x10) * 0x1B0 + row * 0x10 + tx % 0x10)
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
    return memory.readbyte(0x1C800 + math.floor(tx / 0x10) * 0x1B0 + ty * 0x10 + tx % 0x10) ~= 0
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
    local dead = isMarioDead() and "1" or "0"
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
        return false
    end

    local captureIndex = tonumber(string.match(response, "^CAPTURE:(%d+)$"))
    if captureIndex ~= nil then
        captureMode = true
        loadLevel(captureIndex)
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

while true do
    local forcedReset = false
    if isMessageBoxActive() then
        forcedReset = dismissMessageBox()
    end

    local manualReset = manualResetPulse() or forcedReset
    comm.socketServerSend(buildState(isLevelComplete(), manualReset) .. "\n")
    local response = comm.socketServerResponse()
    local shouldStop = applyAction(response)
    if shouldStop then
        console.log("MarioBridge: finalizado, deteniendo script.")
        break
    end
    emu.frameadvance()
end