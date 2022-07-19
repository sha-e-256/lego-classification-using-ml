-- Lua script used to generate training dataset
-- Each model (Lego piece) will 360 degrees in all three axes
-- until n x n x n images are developed

-- Define properties of the animation
function register()

    n = 4 -- Dimensions of training dataset
    local ani = ldc.animation('training_dataset_collection')

    -- LDCad has a cap of 100 FPS
    aniDuration = n -- Total length in s of animation
    aniFPS = n * n -- Frames per second
    -- Training dataset size = aniDuration*aniFPS = n^3

    ani:setLength(aniDuration) 
    ani:setFPS(aniFPS) 

    ani:setEvent('start', 'setup') -- function 'setup' runs once before animation
    ani:setEvent('frame', 'loop') -- function 'loop' runs during animation

end

-- Will run once before animation starts
function setup()

    local cam = ldc.camera()  -- Create a camera object

    camLookAtPos = ldc.vector()
    camLookAtPos:set(0, 0, 0) -- Position where the camera points towards

    cam:setMode(3) -- Camera is placed in third-person view
    camDistanceFromPos = 800 -- Camera distance from point where camera points towards
   

    angleInterval = 360 / n

    counterTilt = 0  -- After a 360 degree tilt is completed (when counterTilt == aniFPS)...
    counterYaw = 0   -- Rotate by angleInterval, then tilt again
    counterRoll = 0

    -- **Debug statements
    i = 1

end

-- Runs throughout duration of animation
function loop()

    local ani = ldc.animation.getCurrent()
    curTime = ani:getFrameTime()
 
    local cam = ldc.camera()
    cam:setThirdPerson(
    camLookAtPos,
    camDistanceFromPos,
    -counterYaw*angleInterval, -- Yaw (left & right)
    counterTilt*angleInterval, -- Tilt (up & down)
    counterRoll*angleInterval  -- Roll (rotation)
    )
    cam:apply(0)

    -- **Debug statements
    -- print( string.format( "%d: counterTilt = %.2f, counterYaw = %.2f, counterRoll = %.2f", i, counterTilt*angleInterval, counterYaw*angleInterval, counterRoll*angleInterval))

    -- for loops don't work; they timeout; use nested if-statements instead

    -- **Debug statements
    -- i = i + 1
    counterTilt = counterTilt + 1           -- Rotate model down
    if (counterTilt == n) then              -- When a full 360 degree tilt is completed
        counterTilt = 0                     -- Reset counterTilt
        counterYaw = counterYaw + 1         -- And rotate model to the right 
        if(counterYaw == n) then            -- When a full 360 degree yaw is completed
            counterYaw = 0                  -- Reset counterYaw
            counterRoll = counterRoll + 1   -- And roll model 
            if (counterRoll == n) then      -- When a full 360 degree roll is completed
                counterRoll = 0             -- Stop
            end
        end
    end
end

register()
