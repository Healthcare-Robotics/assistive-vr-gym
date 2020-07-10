from gym.envs.registration import register

# Human Testing
register(
    id='HumanTesting-v0',
    entry_point='assistive_gym.envs:HumanTestingEnv',
    max_episode_steps=200,
)

# Scratch Itch PR2
register(
    id='ScratchItchPR2-v0',
    entry_point='assistive_gym.envs:ScratchItchPR2Env',
    max_episode_steps=200,
)

# Scratch Itch Jaco
register(
    id='ScratchItchJaco-v0',
    entry_point='assistive_gym.envs:ScratchItchJacoEnv',
    max_episode_steps=200,
)

# Scratch Itch PR2 Human
register(
    id='ScratchItchPR2Human-v0',
    entry_point='assistive_gym.envs:ScratchItchPR2HumanEnv',
    max_episode_steps=200,
)

# Scratch Itch Jaco Human
register(
    id='ScratchItchJacoHuman-v0',
    entry_point='assistive_gym.envs:ScratchItchJacoHumanEnv',
    max_episode_steps=200,
)

# Scratch Itch PR2 New
register(
    id='ScratchItchPR2New-v0',
    entry_point='assistive_gym.envs:ScratchItchPR2NewEnv',
    max_episode_steps=200,
)

# Scratch Itch Jaco New
register(
    id='ScratchItchJacoNew-v0',
    entry_point='assistive_gym.envs:ScratchItchJacoNewEnv',
    max_episode_steps=200,
)

# Scratch Itch VR PR2
register(
    id='ScratchItchVRPR2-v0',
    entry_point='assistive_gym.envs:ScratchItchVRPR2Env',
    max_episode_steps=9999999999,
)

# Scratch Itch VR Jaco
register(
    id='ScratchItchVRJaco-v0',
    entry_point='assistive_gym.envs:ScratchItchVRJacoEnv',
    max_episode_steps=9999999999,
)

# Scratch Itch VR PR2 Human
register(
    id='ScratchItchVRPR2Human-v0',
    entry_point='assistive_gym.envs:ScratchItchVRPR2HumanEnv',
    max_episode_steps=9999999999,
)

# Scratch Itch VR Jaco Human
register(
    id='ScratchItchVRJacoHuman-v0',
    entry_point='assistive_gym.envs:ScratchItchVRJacoHumanEnv',
    max_episode_steps=9999999999,
)

# Scratch Itch VR PR2 New
register(
    id='ScratchItchVRPR2New-v0',
    entry_point='assistive_gym.envs:ScratchItchVRPR2NewEnv',
    max_episode_steps=9999999999,
)

# Scratch Itch VR Jaco New
register(
    id='ScratchItchVRJacoNew-v0',
    entry_point='assistive_gym.envs:ScratchItchVRJacoNewEnv',
    max_episode_steps=9999999999,
)

# Bed Bathing PR2
register(
    id='BedBathingPR2-v0',
    entry_point='assistive_gym.envs:BedBathingPR2Env',
    max_episode_steps=200,
)

# Bed Bathing Jaco
register(
    id='BedBathingJaco-v0',
    entry_point='assistive_gym.envs:BedBathingJacoEnv',
    max_episode_steps=200,
)

# Bed Bathing PR2 Human
register(
    id='BedBathingPR2Human-v0',
    entry_point='assistive_gym.envs:BedBathingPR2HumanEnv',
    max_episode_steps=200,
)

# Bed Bathing Jaco Human
register(
    id='BedBathingJacoHuman-v0',
    entry_point='assistive_gym.envs:BedBathingJacoHumanEnv',
    max_episode_steps=200,
)

# Bed Bathing PR2 New
register(
    id='BedBathingPR2New-v0',
    entry_point='assistive_gym.envs:BedBathingPR2NewEnv',
    max_episode_steps=200,
)

# Bed Bathing Jaco New
register(
    id='BedBathingJacoNew-v0',
    entry_point='assistive_gym.envs:BedBathingJacoNewEnv',
    max_episode_steps=200,
)

# BedBathing VR PR2
register(
    id='BedBathingVRPR2-v0',
    entry_point='assistive_gym.envs:BedBathingVRPR2Env',
    max_episode_steps=9999999999,
)

# BedBathing VR Jaco
register(
    id='BedBathingVRJaco-v0',
    entry_point='assistive_gym.envs:BedBathingVRJacoEnv',
    max_episode_steps=9999999999,
)

# BedBathing VR PR2 Human
register(
    id='BedBathingVRPR2Human-v0',
    entry_point='assistive_gym.envs:BedBathingVRPR2HumanEnv',
    max_episode_steps=9999999999,
)

# BedBathing VR Jaco Human
register(
    id='BedBathingVRJacoHuman-v0',
    entry_point='assistive_gym.envs:BedBathingVRJacoHumanEnv',
    max_episode_steps=9999999999,
)

# BedBathing VR PR2 New
register(
    id='BedBathingVRPR2New-v0',
    entry_point='assistive_gym.envs:BedBathingVRPR2NewEnv',
    max_episode_steps=9999999999,
)

# BedBathing VR Jaco New
register(
    id='BedBathingVRJacoNew-v0',
    entry_point='assistive_gym.envs:BedBathingVRJacoNewEnv',
    max_episode_steps=9999999999,
)

# Drinking PR2
register(
    id='DrinkingPR2-v0',
    entry_point='assistive_gym.envs:DrinkingPR2Env',
    max_episode_steps=200,
)

# Drinking Jaco
register(
    id='DrinkingJaco-v0',
    entry_point='assistive_gym.envs:DrinkingJacoEnv',
    max_episode_steps=200,
)

# Drinking PR2 Human
register(
    id='DrinkingPR2Human-v0',
    entry_point='assistive_gym.envs:DrinkingPR2HumanEnv',
    max_episode_steps=200,
)

# Drinking Jaco Human
register(
    id='DrinkingJacoHuman-v0',
    entry_point='assistive_gym.envs:DrinkingJacoHumanEnv',
    max_episode_steps=200,
)

# Drinking PR2 New
register(
    id='DrinkingPR2New-v0',
    entry_point='assistive_gym.envs:DrinkingPR2NewEnv',
    max_episode_steps=200,
)

# Drinking Jaco New
register(
    id='DrinkingJacoNew-v0',
    entry_point='assistive_gym.envs:DrinkingJacoNewEnv',
    max_episode_steps=200,
)

# Drinking VR PR2
register(
    id='DrinkingVRPR2-v0',
    entry_point='assistive_gym.envs:DrinkingVRPR2Env',
    max_episode_steps=9999999999,
)

# Drinking VR Jaco
register(
    id='DrinkingVRJaco-v0',
    entry_point='assistive_gym.envs:DrinkingVRJacoEnv',
    max_episode_steps=9999999999,
)

# Drinking VR PR2 Human
register(
    id='DrinkingVRPR2Human-v0',
    entry_point='assistive_gym.envs:DrinkingVRPR2HumanEnv',
    max_episode_steps=9999999999,
)

# Drinking VR Jaco Human
register(
    id='DrinkingVRJacoHuman-v0',
    entry_point='assistive_gym.envs:DrinkingVRJacoHumanEnv',
    max_episode_steps=9999999999,
)

# Drinking VR PR2 New
register(
    id='DrinkingVRPR2New-v0',
    entry_point='assistive_gym.envs:DrinkingVRPR2NewEnv',
    max_episode_steps=9999999999,
)

# Drinking VR Jaco New
register(
    id='DrinkingVRJacoNew-v0',
    entry_point='assistive_gym.envs:DrinkingVRJacoNewEnv',
    max_episode_steps=9999999999,
)

# Feeding PR2
register(
    id='FeedingPR2-v0',
    entry_point='assistive_gym.envs:FeedingPR2Env',
    max_episode_steps=200,
)

# Feeding Jaco
register(
    id='FeedingJaco-v0',
    entry_point='assistive_gym.envs:FeedingJacoEnv',
    max_episode_steps=200,
)

# Feeding PR2 Human
register(
    id='FeedingPR2Human-v0',
    entry_point='assistive_gym.envs:FeedingPR2HumanEnv',
    max_episode_steps=200,
)

# Feeding Jaco Human
register(
    id='FeedingJacoHuman-v0',
    entry_point='assistive_gym.envs:FeedingJacoHumanEnv',
    max_episode_steps=200,
)

# Feeding PR2 New
register(
    id='FeedingPR2New-v0',
    entry_point='assistive_gym.envs:FeedingPR2NewEnv',
    max_episode_steps=200,
)

# Feeding Jaco New
register(
    id='FeedingJacoNew-v0',
    entry_point='assistive_gym.envs:FeedingJacoNewEnv',
    max_episode_steps=200,
)

# Feeding VR PR2
register(
    id='FeedingVRPR2-v0',
    entry_point='assistive_gym.envs:FeedingVRPR2Env',
    max_episode_steps=9999999999,
)

# Feeding VR Jaco
register(
    id='FeedingVRJaco-v0',
    entry_point='assistive_gym.envs:FeedingVRJacoEnv',
    max_episode_steps=9999999999,
)

# Feeding VR PR2 Human
register(
    id='FeedingVRPR2Human-v0',
    entry_point='assistive_gym.envs:FeedingVRPR2HumanEnv',
    max_episode_steps=9999999999,
)

# Feeding VR Jaco Human
register(
    id='FeedingVRJacoHuman-v0',
    entry_point='assistive_gym.envs:FeedingVRJacoHumanEnv',
    max_episode_steps=9999999999,
)

# Feeding VR PR2 New
register(
    id='FeedingVRPR2New-v0',
    entry_point='assistive_gym.envs:FeedingVRPR2NewEnv',
    max_episode_steps=9999999999,
)

# Feeding VR Jaco New
register(
    id='FeedingVRJacoNew-v0',
    entry_point='assistive_gym.envs:FeedingVRJacoNewEnv',
    max_episode_steps=9999999999,
)